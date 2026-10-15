from copy import deepcopy
import itertools
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from clm001_r2.model import Model
from clm001_r2.oracle import Snapshot, paired, OracleUntrusted
from clm001_r2.sampling import distribution, proposals
from clm001_r2.training import Trainer, grouped_clip, NumericalFailure, relative_loss
from clm001.q_connection import Connection as RawConnection


def batch(c):
    x = torch.tensor([[1, 2, 3, 4]])[:, : c["training"]["sequence_length"]]
    return [(x, (x + 1) % c["model"]["vocab_size"])]


@pytest.mark.parametrize("arm", ["QF-R2", "J0-R2"])
def test_inventory_and_critic(config, arm):
    m = Model(config, arm)
    other = Model(config, "J0-R2" if arm == "QF-R2" else "QF-R2")
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in other.parameters())
    assert all(p.requires_grad == (arm == "J0-R2") for p in m.controller.critic.parameters())
    assert all(torch.equal(p, other.state_dict()[n]) for n, p in m.state_dict().items())


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_zero_reference_order_mask(config, depth):
    m = Model(config, "QF-R2")
    raw = RawConnection(config, m.registry.actions)
    raw.load_state_dict(m.controller.state_dict())
    h = torch.randn(2, 8)
    prev = torch.zeros(2, dtype=torch.long)
    legal = m.registry.legal(depth, 3, 1, "cpu")
    q, _ = m.controller(h, prev, depth, legal)
    r, _ = raw(h, prev, depth, legal)
    assert torch.equal(q[:, -1], torch.zeros(2))
    assert torch.equal(
        distribution(q, legal, 0, 1, "greedy"), distribution(r, legal, 0, 1, "greedy")
    )
    assert torch.allclose(distribution(q, legal, 0.4, 1), distribution(r, legal, 0.4, 1), atol=1e-7)
    assert not distribution(q, legal, 0.4, 1)[:, ~legal].any()


def test_stop_tie(config):
    m = Model(config, "QF-R2")
    q = torch.zeros(1, m.registry.actions)
    assert (
        distribution(q, m.registry.legal(1, 3, 1, "cpu"), 0, 1, "greedy").argmax()
        == m.registry.stop
    )
    assert (
        distribution(q, m.registry.legal(0, 3, 1, "cpu"), 0, 1, "greedy").argmax()
        != m.registry.stop
    )


@pytest.mark.parametrize("arm", ["QF-R2", "J0-R2"])
def test_causal_replay_decode(config, arm):
    m = Model(config, arm)
    x = batch(config)[0][0]
    trace = torch.tensor([[[3, 0, 4]] * 4])
    full = m(x, replay=trace)
    change = x.clone()
    change[0, -1] = 7
    assert torch.allclose(full.logits[:, :-1], m(change, replay=trace).logits[:, :-1])
    memory = None
    parts = []
    for i in range(4):
        out = m(x[:, i : i + 1], past=memory, replay=trace[:, i : i + 1])
        memory = out.memory
        parts.append(out.logits)
    assert torch.allclose(full.logits, torch.cat(parts, 1), atol=1e-6)


def test_cell_has_no_history_channel(config):
    m = Model(config, "QF-R2")
    x = torch.tensor([[1, 2, 3, 4]])
    changed = x.clone()
    changed[0, 0] = 9
    replay = torch.tensor([[[0, 0, 4]] * 4])
    assert torch.equal(m(x, replay=replay).logits[:, 1:], m(changed, replay=replay).logits[:, 1:])


@pytest.mark.parametrize("depth", [0, 1])
def test_stop_baseline_and_relative_exact(config, depth):
    m = Model(config, "QF-R2")
    x, y = batch(config)[0]
    out = m(x, probe_coordinates=[(0, depth)], replay=torch.tensor([[[0, 0, 4]] * 4]))
    s = out.probes[0]
    snap = Snapshot(m)
    u = torch.rand(8, 3)
    lab = snap.relative(s, 0, y[0, 0], u)
    assert (
        lab.stop_loss == F.cross_entropy(snap.model.head(s.hidden[None]), y[0, 0].reshape(1)).item()
    )
    assert lab.value == pytest.approx(np.mean(lab.samples), abs=1e-7)
    assert lab.stop_legal == (depth >= 1)
    assert lab.value == pytest.approx(lab.absolute.value - lab.stop_loss)
    lab.verify(snap, s, y[0, 0])
    lab.value += 1
    with pytest.raises(OracleUntrusted):
        lab.verify(snap, s, y[0, 0])


def test_budget_and_pairing(config):
    m = Model(config, "QF-R2")
    x, y = batch(config)[0]
    state = m(x, probe_coordinates=[(0, 0)]).probes[0]
    snap = Snapshot(m)
    legal = m.registry.legal(0, 3, 1, "cpu")
    candidates = proposals(
        legal, state.chosen, m.registry.stop, 2, torch.Generator().manual_seed(5)
    )
    config["oracle"]["pair_tolerance"] = 1e9  # Force ambiguity to exercise budget truncation.
    labels, info = snap.estimate(state, y[0, 0], candidates, torch.rand(8, 3), config["oracle"], 18)
    assert info["operator_calls"] <= 18 and info["samples"] == 2 and info["budget_truncated"]
    assert len(set(len(label.samples) for label in labels)) == 1
    assert len(set(label.absolute.uniforms_sha for label in labels)) == 1


@pytest.mark.parametrize("n,expected", [(2, False), (4, True), (8, True)])
def test_confidence(n, expected, config):
    assert paired([0.0] * n, [1.0] * n, config["oracle"])["identifiable"] == expected
    assert not paired([1.0] * n, [1.0] * n, config["oracle"])["identifiable"]


def test_ht_exact_expectation():
    # Actual action forced, 2 extras sampled without replacement from 3 alternatives.
    losses = [2.0, 3.0, 5.0, 7.0]
    estimate = []
    for extras in itertools.combinations(range(1, 4), 2):
        estimate.append((losses[0] + sum(losses[a] / (2 / 3) for a in extras)) / 4)
    assert np.mean(estimate) == pytest.approx(np.mean(losses))


def test_visited_early_stop(config):
    m = Model(config, "QF-R2")
    m.registry.plan = [0, m.registry.stop]
    out = m(
        batch(config)[0][0],
        probe_coordinates="reservoir",
        generator=torch.Generator().manual_seed(7),
    )
    assert all(s.depth in (0, 1) for s in out.probes)
    assert sum(s.population for s in out.probes) == 8
    assert all(s.depth < 2 for s in out.probes)
    assert out.actual_stop.all() and not out.forced_end.any()


@pytest.mark.parametrize("arm", ["QF-R2", "J0-R2"])
def test_update_gradient_contract(config, arm):
    m = Model(config, arm)
    t = Trainer(m)
    critic = deepcopy(m.controller.critic.state_dict())
    row, snap, labels = t.update(batch(config))
    assert row["oracle_states"] > 0 and labels and t.data_position == 4
    assert (
        row["counterfactual_operator_calls"]
        <= config["oracle"]["max_counterfactual_operator_calls_per_step"]
    )
    assert all(torch.isfinite(p).all() for p in m.parameters())
    if arm == "QF-R2":
        assert all(torch.equal(v, m.controller.critic.state_dict()[k]) for k, v in critic.items())
        assert all(p.grad is None for p in m.controller.critic.parameters())
    else:
        assert any(p.grad is not None for p in m.controller.critic.parameters())


def test_group_clip_independence(config):
    m = Model(config, "QF-R2")
    for name, p in m.named_parameters():
        p.grad = torch.ones_like(p) * (1e6 if name.startswith("controller.") else 0.001)
    before = {n: p.grad.clone() for n, p in m.named_parameters() if not n.startswith("controller.")}
    grouped_clip(m, config)
    assert all(torch.equal(p.grad, before[n]) for n, p in m.named_parameters() if n in before)


def test_rollback_all_rng_optimizer(config):
    t = Trainer(Model(config, "QF-R2"))
    weights = deepcopy(t.model.state_dict())
    rng = {k: g.get_state().clone() for k, g in t.rng.items()}

    def corrupt():
        with torch.no_grad():
            next(t.model.parameters()).fill_(float("nan"))

    with pytest.raises(NumericalFailure):
        t.update(batch(config), failure=corrupt)
    assert t.step == 0 and t.data_position == 0 and not t.journal and not t.opt.state
    assert all(torch.equal(v, t.model.state_dict()[k]) for k, v in weights.items())
    assert all(torch.equal(g.get_state(), rng[k]) for k, g in t.rng.items())


def test_short_world_exact_suffix_enumeration(config):
    m = Model(config, "QF-R2")
    x, y = batch(config)[0]
    state = m(x, probe_coordinates=[(0, 1)], replay=torch.tensor([[[0, 0, 4]] * 4])).probes[0]
    snap = Snapshot(m)
    a = 0
    h = snap.advance(state.hidden, a, state)
    probabilities = snap.probabilities(h, a, 2)
    weighted = 0.0
    left = 0.0
    for action, p in enumerate(probabilities.tolist()):
        if p:
            u = torch.zeros(3)
            u[2] = left + p / 2
            value, trace, work = snap.suffix(state, a, y[0, 0], u)
            assert trace[-1] == action
            weighted += p * value
        left += p
    assert weighted == pytest.approx(snap.exact(state, a, y[0, 0]), abs=2e-7)
    baseline = snap.terminal(state.hidden, y[0, 0])
    assert weighted - baseline == pytest.approx(snap.exact(state, a, y[0, 0]) - baseline, abs=2e-7)


def test_real_q_gradients_do_not_touch_cells(config):
    m = Model(config, "QF-R2")
    x, y = batch(config)[0]
    state = m(x, probe_coordinates=[(0, 0)]).probes[0]
    snap = Snapshot(m)
    label = snap.relative(state, 0, y[0, 0], torch.rand(4, 3))
    m.zero_grad(set_to_none=True)
    q, rank, _ = relative_loss(m, [(state, y[0, 0], label)], snap, config)
    (q + 0.1 * rank).backward()
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith("controller."))
    assert any(p.grad is not None for p in m.controller.scorer.parameters())


def test_stop_and_forced_end_distinct(config):
    m = Model(config, "QF-R2")
    x = batch(config)[0][0]
    out = m(x, replay=torch.tensor([[[0, 0, 0]] * 4]))
    assert out.forced_end.all() and not out.actual_stop.any() and (out.lengths == 3).all()
    out = m(x, replay=torch.tensor([[[0, 4, -1]] * 4]))
    assert out.actual_stop.all() and not out.forced_end.any() and (out.lengths == 1).all()


def test_full_inventory_real_shape_cpu_update():
    from clm001_r2.common import load

    c = load()
    c["synthetic"] = True
    c["training"].update(
        steps=2, effective_batch=1, microbatch=1, sequence_length=4, target_tokens=8
    )
    t = Trainer(Model(c, "QF-R2"))
    assert sum(p.numel() for p in t.model.parameters()) == 1075492
    row, _, _ = t.update(batch(c))
    assert np.isfinite(row["live_ce"]) and row["oracle_states"] > 0


def test_context_aliasing_is_visible_not_hidden(config):
    from copy import copy
    from exp006.memory import Memory

    m = Model(config, "QF-R2")
    x, y = batch(config)[0]
    state = m(x, probe_coordinates=[(0, 1)], replay=torch.tensor([[[0, 0, 4]] * 4])).probes[0]
    other = copy(state)
    other.memory = Memory(state.memory.key, state.memory.value + 3)
    snap = Snapshot(m)
    original = snap.advance(state.hidden, m.registry.ctx, state)
    changed = snap.advance(other.hidden, m.registry.ctx, other)
    assert not torch.equal(original, changed)
    legal = m.registry.legal(1, 3, 1, "cpu")
    previous = torch.tensor([state.previous])
    assert torch.equal(
        m.controller(state.hidden[None], previous, 1, legal)[0],
        m.controller(other.hidden[None], previous, 1, legal)[0],
    )
    assert torch.equal(snap.advance(state.hidden, 0, state), snap.advance(other.hidden, 0, other))
