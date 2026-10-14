import copy
import torch
import pytest
from pd001.models import PDCLM, make_model
from pd001.accounting import Counter
from pd001.losses import joint_loss, sample_paths
from pd001.runner import fixture_config
from pd001.configs import ARMS, count
from pd001.evaluation import intervention_pair


def path(c, actions, n=1):
    row = actions + [-1] * (c["budget"] - len(actions))
    return torch.tensor([[row] * n])


def test_actual_ratios_and_same_architecture(proposal):
    rows = count(proposal)
    assert rows["M0"]["total"] == 16483968
    assert all(r["ratio_pass"] for r in rows.values())
    assert (
        rows["P50-F16"]["total"]
        == rows["P50-F32"]["total"]
        == rows["P50-A32"]["total"]
        == rows["P50-A32-J1"]["total"]
    )
    assert rows["P25-A64"]["width"] == 256


@pytest.mark.parametrize("arm", ARMS)
def test_all_cpu_joint_fixtures(proposal, arm):
    from pd001.runner import dry_run

    r = dry_run(proposal, arm)
    assert r["gpu_runs"] == 0 and r["status"] == "CPU_FIXTURE_PASS"
    assert r["cost"]["resident_parameters"] > 0


def test_initial_stop_runs_no_cell_or_context(tiny):
    m = PDCLM(tiny, "P50-A32").train()
    with Counter(m) as counter:
        out = m(torch.tensor([[1, 2]]), replay=path(tiny, [3], 2))
    assert out.logits.shape == (1, 2, 23)
    assert out.lengths.sum() == 0 and out.decision_mask.sum() == 2
    assert not out.forced_stop.any()
    assert counter.calls["controller.scorer.0"] == 2
    assert not any(v for k, v in counter.calls.items() if k.startswith(("cells.", "context.")))


def test_self_loop_repeats_real_work_and_state(tiny):
    c = copy.deepcopy(tiny)
    c.update(budget=10, adaptive=False)
    m = PDCLM(c, "P50-F32").eval()
    calls = []
    hook = m.cells[0].register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad(), Counter(m) as counter:
        out = m(torch.tensor([[1]]), replay=path(c, [0] * 10), capture=True)
    hook.remove()
    assert len(calls) == 10 and counter.calls["cells.0.up"] == 10
    assert out.lengths.item() == 10 and out.forced_stop.item()
    assert out.decision_mask.sum() == 10  # no virtual STOP at bound
    assert all(r["delta"].norm() > 0 for r in out.state_records)
    assert "cells.1.up" not in counter.calls
    with torch.no_grad(), Counter(m) as short:
        m(torch.tensor([[1]]), stop_after=1, action_overrides={0: 0})
    assert counter.by_module["cells.0.up"] == 10 * short.by_module["cells.0.up"]


def test_stopped_positions_never_updated(tiny):
    c = copy.deepcopy(tiny)
    c["budget"] = 3
    m = PDCLM(c, "P50-A32").eval()
    replay = torch.tensor([[[3, -1, -1], [0, 0, 3]]])
    with torch.no_grad():
        out = m(torch.tensor([[1, 2]]), replay=replay)
        initial = m(torch.tensor([[1, 2]]), replay=torch.tensor([[[3, -1, -1], [3, -1, -1]]]))
    assert torch.equal(out.logits[:, 0], initial.logits[:, 0])
    assert out.lengths.tolist() == [[0, 2]]
    assert out.decision_mask.tolist() == [[[True, False, False], [True, True, True]]]


def test_illegal_stop_fixed_and_replay_padding(tiny):
    c = copy.deepcopy(tiny)
    c["adaptive"] = False
    m = PDCLM(c, "P50-F32")
    with pytest.raises(ValueError, match="illegal"):
        m(torch.tensor([[1]]), replay=path(c, [3]))
    m = PDCLM(tiny, "P50-A32")
    with pytest.raises(ValueError, match="padded"):
        m(torch.tensor([[1]]), replay=path(tiny, [3, 0]))


def test_self_only_mask_allows_repeat_and_ctx(tiny):
    c = copy.deepcopy(tiny)
    c["self_only"] = True
    c["budget"] = 4
    m = PDCLM(c, "P50-SELF32")
    m(torch.tensor([[1]]), replay=path(c, [1, 2, 1, 3]))
    with pytest.raises(ValueError, match="illegal"):
        m(torch.tensor([[1]]), replay=path(c, [1, 2, 0, 3]))


def test_causality_and_full_incremental_replay(tiny):
    c = copy.deepcopy(tiny)
    c["budget"] = 4
    m = PDCLM(c, "P50-A32").eval()
    x = torch.tensor([[1, 2, 3]])
    replay = path(c, [2, 0, 0, 3], 3)
    with torch.no_grad():
        full = m(x, replay=replay)
        changed = m(torch.tensor([[1, 2, 8]]), replay=replay)
        assert torch.allclose(full.logits[:, :2], changed.logits[:, :2], atol=1e-6)
        memory = None
        logits = []
        for i in range(3):
            out = m(x[:, i : i + 1], past=memory, replay=replay[:, i : i + 1])
            memory = out.memory
            logits.append(out.logits)
        assert torch.allclose(full.logits, torch.cat(logits, 1), atol=1e-6)


def test_joint_cell_connection_gradient_and_critic_detach(tiny, proposal):
    c = copy.deepcopy(tiny)
    c["budget"] = 3
    m = PDCLM(c, "P50-A32").double().train()
    x = torch.tensor([[1]])
    out = m(x, replay=path(c, [0, 0, 3]))
    loss, _ = joint_loss([out], torch.tensor([[2]]), proposal["training"])
    loss.backward()
    assert m.cells[0].up.weight.grad.norm() > 0
    assert m.controller.scorer[0].weight.grad.norm() > 0
    assert m.controller.source.weight.grad.norm() > 0
    assert m.cells[1].up.weight.grad is None
    m.zero_grad(set_to_none=True)
    out = m(x, replay=path(c, [0, 3]))
    out.values.sum().backward()
    assert m.init.weight.grad is None
    assert m.cells[0].up.weight.grad is None
    assert m.controller.step.weight.grad is None
    assert m.controller.critic[0].weight.grad.norm() > 0


def test_joint_j1_parameter_identity_and_rng_independence(tiny, proposal):
    m = PDCLM(tiny, "P50-A32-J1").train()
    x = torch.tensor([[1, 2]])
    g1, g2 = [torch.Generator().manual_seed(s) for s in (7100, 7101)]
    before = m.state_dict()
    paths = sample_paths(m, x, [g1, g2], "J1")
    loss, _ = joint_loss(paths, torch.tensor([[2, 3]]), proposal["training"], "J1")
    loss.backward()
    assert torch.isfinite(loss)
    assert paths[0].states.data_ptr() != paths[1].states.data_ptr()
    assert all(torch.equal(before[n], p) for n, p in m.state_dict().items())
    with pytest.raises(ValueError, match="independent"):
        sample_paths(m, x, [g1, g1], "J1")


def test_seeded_replay_and_64step_bptt(tiny, proposal):
    c = copy.deepcopy(tiny)
    c.update(budget=64, adaptive=False)
    m = PDCLM(c, "P25-F32").train()
    x = torch.tensor([[1]])
    a = m(x, generator=torch.Generator().manual_seed(7), policy="sampled")
    b = m(x, generator=torch.Generator().manual_seed(7), policy="sampled")
    assert torch.equal(a.actions, b.actions) and torch.equal(a.logits, b.logits)
    out = m(x, replay=path(c, [0] * 64), capture=True)
    out.logits.sum().backward()
    assert m.cells[0].up.weight.grad.norm() > 0
    assert out.state_records[0]["router_state"].requires_grad


@pytest.mark.parametrize(
    "kind",
    [
        "stop_early",
        "extra_steps",
        "edge_zero",
        "edge_shuffle",
        "order_permutation",
        "self_loop_replacement",
        "context_zero_replay",
    ],
)
def test_interventions_real_reexecute(tiny, kind):
    m = PDCLM(tiny, "P50-A32").eval()
    result = intervention_pair(m, torch.tensor([[1, 2]]), torch.tensor([[2, 3]]), kind)
    assert result["paired_tokens"] == 2
    assert result["semantics"] in ("policy_replan", "fixed_action_replay")
    assert "delta_forward_gemm_flops" in result
    assert result["scientific_decision"] is False


def test_moe_native_attention_decode_parity(proposal):
    m = make_model(fixture_config(proposal, "M0"), "M0", 17).eval()
    with torch.no_grad():
        x = torch.tensor([[1, 2, 3]])
        full = m(x)
        prefix = m.prefill(x[:, :2])
        out = m.decode_step(x[:, -1], prefix.memory)
    assert torch.allclose(full.logits[:, -1], out.logits[:, 0], atol=1e-6)


def test_j1_two_rollouts_are_billed_twice(tiny):
    from pd001.accounting import TrainingGemmCounter

    c = copy.deepcopy(tiny)
    c["adaptive"] = False
    c["budget"] = 3
    m = PDCLM(c, "P50-F32").train()
    x = torch.tensor([[1]])
    replay = path(c, [0, 2, 0])
    with Counter(m) as one:
        m(x, replay=replay)
    with Counter(m) as two:
        m(x, replay=replay)
        m(x, replay=replay)
    assert two.total == 2 * one.total
    assert two.work == 2 * one.work
    m.zero_grad(set_to_none=True)
    with TrainingGemmCounter() as observed:
        out = m(x, replay=replay)
        out.logits.square().sum().backward()
    assert observed.flops > one.total
    assert observed.by_operator["aten.mm.default"] > 0


def test_cpu_bfloat16_replay_dtype_consistency(tiny):
    c = copy.deepcopy(tiny)
    c["budget"] = 3
    m = PDCLM(c, "P50-A32").train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(torch.tensor([[1]]), replay=path(c, [0, 2, 3]))
        loss = out.logits.float().square().mean() + out.log_prob.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert out.log_prob.dtype == torch.float32


def test_shared_j0_j1_initial_parameters(tiny):
    a = PDCLM(tiny, "P50-A32", 17)
    b = PDCLM(tiny, "P50-A32-J1", 17)
    assert all(torch.equal(p, b.state_dict()[name]) for name, p in a.state_dict().items())


def test_microbatch_accumulation_matches_joint_token_objective(tiny, proposal):
    c = copy.deepcopy(tiny)
    c["budget"] = 3
    a = PDCLM(c, "P50-A32", 17).train()
    b = PDCLM(c, "P50-A32", 17).train()
    x = torch.tensor([[1], [2]])
    y = torch.tensor([[2], [3]])
    replay = torch.tensor([[[3, -1, -1]], [[0, 0, 3]]])
    full = a(x, replay=replay)
    loss, _ = joint_loss([full], y, proposal["training"], "J0")
    loss.backward()
    for i in range(2):
        out = b(x[i : i + 1], replay=replay[i : i + 1])
        partial, _ = joint_loss([out], y[i : i + 1], proposal["training"], "J0")
        (partial / 2).backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-5), name
