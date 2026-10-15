from copy import deepcopy
import pytest
import torch
from clm001_r3.model import Model
from clm001_r3.training import Trainer
from clm001_r3.probing import EventSampler
from clm001_r3.competitive_credit import subset_loss, loss


def batch():
    x = torch.tensor([[1, 2, 3, 4]])
    return [(x, x + 1)]


def test_analytic_and_finite_difference():
    z = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64, requires_grad=True)
    costs = torch.tensor([-2.0, 0.0, 1.0], dtype=torch.float64)
    value = subset_loss(z, costs, 0.7)
    g = torch.autograd.grad(value, z)[0]
    p = (z / 0.7).softmax(-1)
    expected = p / 0.7 * (costs - (p * costs).sum())
    assert torch.allclose(g, expected)
    for i in range(3):
        step = torch.zeros(3, dtype=z.dtype)
        step[i] = 1e-5
        fd = (
            subset_loss(z.detach() + step, costs, 0.7) - subset_loss(z.detach() - step, costs, 0.7)
        ) / 2e-5
        assert abs(float(fd - g[i])) < 1e-8
    assert g[0] < 0  # gradient descent increases preference for the helpful Cell


@pytest.mark.parametrize("arm", ["Q-R3", "QC-R3", "QS-R3"])
def test_capture_preserves_hard_forward(config, arm):
    model = Model(config, arm)
    x, y = batch()[0]
    uniforms = torch.rand(1, 4, 3, generator=torch.Generator().manual_seed(9))
    original = model(x, uniforms=uniforms)
    sampler = EventSampler(4, torch.Generator().manual_seed(8), 1)
    captured = model(x, uniforms=uniforms, competition_sampler=sampler)
    for name in ("logits", "states", "actions", "cost_units", "actual_stop", "forced_end"):
        assert torch.equal(getattr(original, name), getattr(captured, name))
    torch.nn.functional.cross_entropy(captured.logits.reshape(-1, 16), y.reshape(-1)).backward()
    assert sampler.events and all(e.gradient is not None for e in sampler.events)
    assert len(sampler.events) <= 4
    sampler.close()


def test_router_only_competition_and_unselected_cell_no_grad(config):
    model = Model(config, "QC-R3")
    x, y = batch()[0]
    replay = torch.tensor([[[0, 0, 4]] * 4])
    sampler = EventSampler(4, torch.Generator().manual_seed(5), 1)
    out = model(x, replay=replay, competition_sampler=sampler)
    torch.nn.functional.cross_entropy(out.logits.reshape(-1, 16), y.reshape(-1)).backward()
    continuous = {
        n: p.grad.clone() if p.grad is not None else None
        for n, p in model.named_parameters()
        if not n.startswith("controller.")
    }
    credit, records, calls = loss(model, sampler.events, torch.Generator().manual_seed(6))
    credit.backward()
    assert records and calls == len(sampler.events)
    for n, p in model.named_parameters():
        if n in continuous:
            assert p.grad is None if continuous[n] is None else torch.equal(p.grad, continuous[n])
    assert all(p.grad is None for p in model.cells[1].parameters())
    sampler.close()


def test_three_arms_initialization_and_sham_exact_trajectory(config):
    trainers = {arm: Trainer(Model(config, arm)) for arm in config["arms"]}
    initial = trainers["Q-R3"].model.state_dict()
    assert all(
        torch.equal(v, t.model.state_dict()[n])
        for t in trainers.values()
        for n, v in initial.items()
    )
    rows = {arm: t.update(batch())[0] for arm, t in trainers.items()}
    q = trainers["Q-R3"].model.state_dict()
    sham = trainers["QS-R3"].model.state_dict()
    assert all(torch.equal(v, sham[n]) for n, v in q.items())
    assert (
        rows["QC-R3"]["lcg_candidate_cell_calls"] > 0
        and rows["Q-R3"]["lcg_candidate_cell_calls"] == 0
    )
    assert rows["QS-R3"]["lcg_grad_norm"] is not None


def test_credit_nan_rolls_back(config):
    trainer = Trainer(Model(config, "QC-R3"))
    before = deepcopy(trainer.model.state_dict())
    hook = trainer.model.controller.scorer[0].weight.register_hook(
        lambda g: torch.full_like(g, float("nan"))
    )
    with pytest.raises(Exception):
        trainer.update(batch())
    hook.remove()
    assert trainer.step == 0 and all(
        torch.equal(v, trainer.model.state_dict()[n]) for n, v in before.items()
    )


def test_full_parameter_shape_has_real_local_gradient():
    from clm001_r3.common import load

    c = load()
    c["synthetic"] = True
    c["competition"]["beta"] = 0.1
    trainer = Trainer(Model(c, "QC-R3"))
    gen = torch.Generator().manual_seed(140)
    data = [
        (
            torch.randint(0, 4096, (1, 64), generator=gen),
            torch.randint(0, 4096, (1, 64), generator=gen),
        )
        for _ in range(8)
    ]
    row, _, _ = trainer.update(data)
    assert sum(p.numel() for p in trainer.model.parameters()) == 1075492
    assert 0 < row["lcg_events"] <= 32 and row["lcg_candidate_cell_calls"] == 2 * row["lcg_events"]
    assert row["lcg_grad_norm"] > 0 and row["q_grad_norm"] > 0


def test_real_rng_and_causal_incremental_decode_parity(config):
    model = Model(config, "QC-R3")
    x, y = batch()[0]
    one = torch.Generator().manual_seed(50)
    two = torch.Generator().manual_seed(50)
    ordinary = model(x, generator=one)
    sampler = EventSampler(4, torch.Generator().manual_seed(3), 1)
    observed = model(x, generator=two, competition_sampler=sampler)
    assert torch.equal(ordinary.actions, observed.actions) and torch.equal(
        one.get_state(), two.get_state()
    )
    sampler.close()
    replay = torch.tensor([[[3, 0, 4]] * 4])
    full = model(x, replay=replay)
    changed = x.clone()
    changed[0, -1] = 8
    assert torch.equal(full.logits[:, :-1], model(changed, replay=replay).logits[:, :-1])
    pieces = []
    memory = None
    for i in range(4):
        out = model(x[:, i : i + 1], past=memory, replay=replay[:, i : i + 1])
        memory = out.memory
        pieces.append(out.logits)
    assert torch.allclose(full.logits, torch.cat(pieces, 1), atol=1e-6, rtol=1e-5)
