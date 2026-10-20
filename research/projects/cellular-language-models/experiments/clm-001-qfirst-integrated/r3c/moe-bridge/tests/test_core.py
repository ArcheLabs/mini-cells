import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from minicells_r3c_moe_bridge.model import BridgeModel
from minicells_r3c_moe_bridge.instrumentation import MatmulAudit, norm, parameters
from minicells_r3c_moe_bridge.training import Trainer
from minicells_r3c_moe_bridge.data import synthetic_batches


def sample():
    return torch.tensor([[2, 3, 4, 5]]), torch.tensor([[3, 4, 5, 2]])


def grads(model):
    x, y = sample()
    out = model(x, route_rng=torch.Generator().manual_seed(91))
    ce = F.cross_entropy(out.logits.flatten(0, 1), y.flatten())
    task = (
        torch.autograd.grad(ce, list(model.gate.parameters()), allow_unused=True, retain_graph=True)
        if model.arm not in ("A", "B")
        else []
    )
    bal = (
        torch.autograd.grad(out.balance, list(model.gate.parameters()), allow_unused=True)
        if model.arm not in ("A", "B")
        else []
    )
    return out, norm(task).item(), norm(bal).item()


def test_c_d_forward_and_gradients(config):
    c, ct, cb = grads(BridgeModel(config, "C", 123))
    d, dt, db = grads(BridgeModel(config, "D", 123))
    assert torch.equal(c.logits, d.logits)
    assert ct > 0 and dt == 0 and cb > 0 and db > 0
    e, et, eb = grads(BridgeModel(config, "E", 123))
    assert torch.equal(c.logits, e.logits) and ct == et


@pytest.mark.parametrize("arm", list("ABCDE"))
def test_real_sparse_calls(config, arm):
    m = BridgeModel(config, arm, 123)
    x, y = sample()
    with MatmulAudit(m) as audit:
        out = m(x, route_rng=torch.Generator().manual_seed(2))
    cost = audit.finish(out)
    assert cost["cell_rows"] == 8 * x.numel()
    assert cost["context_rows"] == x.numel() and cost["gate_rows"] == 4 * x.numel()
    assert out.operator_rows.sum().item() == 8 * x.numel()
    assert torch.isfinite(out.logits).all()
    for stage in out.stages:
        assert (stage.indices[:, 0] != stage.indices[:, 1]).all()
    assert not hasattr(m, "controller") and not hasattr(m, "stop") and len(m.cells) == 4
    assert (
        parameters(m)["nominal_parameters"]
        == parameters(BridgeModel(config, "C", 123))["nominal_parameters"]
    )


def test_stage_state_and_causality(config):
    m = BridgeModel(config, "C", 123)
    x, y = sample()
    out = m(x)
    for t in range(1, 4):
        assert torch.equal(
            out.stages[t].inputs,
            out.stages[t - 1].inputs + config["model"]["alpha"] * out.stages[t - 1].delta,
        )
        assert torch.equal(out.stages[t].logits, m.gate(out.stages[t].inputs, t))
    alt = x.clone()
    alt[0, -1] = 12
    assert torch.equal(m(x).logits[:, :-1], m(alt).logits[:, :-1])
    memory = None
    parts = []
    for j in range(x.shape[1]):
        o = m(x[:, j : j + 1], past=memory)
        memory = o.memory
        parts.append(o.logits)
    torch.testing.assert_close(out.logits, torch.cat(parts, 1), rtol=1e-5, atol=1e-7)


def test_identical_cells_zero_gate_task_gradient(config):
    m = BridgeModel(config, "C", 123)
    for cell in m.cells[1:]:
        cell.load_state_dict(m.cells[0].state_dict())
    x, y = sample()
    out = m(x)
    loss = F.cross_entropy(out.logits.flatten(0, 1), y.flatten())
    g = torch.autograd.grad(loss, list(m.gate.parameters()), allow_unused=True)
    assert norm(g).item() < 1e-8


def test_logits_change_output_and_early_state_changes_later_gate(config):
    m = BridgeModel(config, "C", 123)
    x, y = sample()
    base = m(x)
    with torch.no_grad():
        m.gate.net[-1].bias[0] += 10
    changed = m(x)
    assert not torch.equal(base.logits, changed.logits)
    assert not torch.equal(base.stages[1].inputs, changed.stages[1].inputs)


def test_selected_cells_task_grad_only(config):
    m = BridgeModel(config, "A", 123)
    x, y = sample()
    x = x[:, :1]

    # Force every stage to the same pair, retaining sparse execution.
    def pair(t, h, z, ids, w):
        return torch.tensor([[0, 1]]), w

    out = m(x, intervention=pair)
    out.logits.square().sum().backward()
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0
        for cell in m.cells[:2]
        for p in cell.parameters()
    )
    assert all(p.grad is None for cell in m.cells[2:] for p in cell.parameters())


def test_joint_update_matches_direct_backward(config):
    m = BridgeModel(config, "C", 123).double()
    reference = copy.deepcopy(m)
    tr = Trainer(m)
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in reference.named_parameters() if not n.startswith("gate.")],
                "lr": 0.01,
            },
            {"params": list(reference.gate.parameters()), "lr": 0.01},
        ],
        weight_decay=config["training"]["weight_decay"],
    )
    batches = synthetic_batches(config, np.random.default_rng(1), "cpu")
    for x, y in batches:
        o = reference(x)
        loss = (F.cross_entropy(o.logits.flatten(0, 1), y.flatten()) + 0.01 * o.balance) / len(
            batches
        )
        loss.backward()
    for group in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], config["training"]["clip"])
    opt.step()
    tr.update(batches)
    for a, b in zip(m.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-7, atol=3e-12)
        # Reused ContextRead computes softmax in float32, including a double fixture.
        # Split CE/balance backward differs at ~1e-12; Adam magnifies near-zero entries.
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("arm", list("ABCDE"))
def test_synthetic_positive_control(config, arm):
    tr = Trainer(BridgeModel(config, arm, 123))
    rng = np.random.default_rng(1)
    fixed = synthetic_batches(config, np.random.default_rng(2), "cpu")

    def loss():
        with torch.no_grad():
            return sum(
                F.cross_entropy(
                    tr.model(x, route_rng=torch.Generator().manual_seed(3)).logits.flatten(0, 1),
                    y.flatten(),
                ).item()
                for x, y in fixed
            )

    before = loss()
    for _ in range(15):
        tr.update(synthetic_batches(config, rng, "cpu"))
    assert loss() < before


@pytest.mark.parametrize("cells", [2, 6])
def test_pool_sizes_and_shared_cost(config, cells):
    config["routing"].update(cells=cells, fixed_pair_offset=1)
    x, y = sample()
    costs = []
    for arm in "ABCDE":
        m = BridgeModel(config, arm, 123)
        with MatmulAudit(m) as audit:
            o = m(x, route_rng=torch.Generator().manual_seed(1))
        cost = audit.finish(o)
        costs.append(cost["per_token_forward_matmul_flops"])
        assert cost["cell_rows"] == 8 * x.numel()
    assert len(set(costs)) == 1
