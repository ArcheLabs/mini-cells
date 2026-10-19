import copy
import pytest
import torch
from minicells_runtime.store import ContinuationError
from r3b_cf001.inference import rollout, census, statistics, reconstruct, greedy, action_selection
from clm001_r3b.analysis_runner import fingerprint
from clm001_r3b.instrumentation import isolated
from r3b_cf001.counterfactual import GreedySnapshot, audit_state, uniforms
from clm001_r3.oracle import Snapshot
from r3b_cf001.common import configuration


def test_noop_native_actions_states_and_rng(case):
    m, x, y = case
    before = fingerprint(m.state_dict())
    rng = torch.get_rng_state().clone()
    with isolated(m), torch.no_grad():
        native = m(x, policy="greedy", generator=torch.Generator().manual_seed(4))
        other = rollout(m, x)
        assert torch.equal(native.actions, other.actions)
        assert torch.equal(native.states, other.states)
        assert torch.equal(native.lengths, other.lengths)
        assert torch.equal(native.actual_stop, other.actual_stop)
        assert statistics(native, y, m.registry)["nll"] == statistics(other, y, m.registry)["nll"]
    assert fingerprint(m.state_dict()) == before
    assert torch.equal(rng, torch.get_rng_state())


def test_census_reconstruct_real_batch_shapes(routed):
    m, x, y = routed
    with torch.no_grad():
        out, records = census(m, x, y, "story", "window")
    assert len(records) == 20
    for record in records:
        state = reconstruct(m, x, out.actions.reshape(-1, 5).tolist(), record)
        assert state.fingerprint() == record["state_fingerprint"]
    tampered = copy.deepcopy(records[-1])
    tampered["state_fingerprint"] = "0" * 64
    with pytest.raises(ContinuationError, match="REPLAY"):
        reconstruct(m, x, out.actions.reshape(-1, 5).tolist(), tampered)


def test_masks_are_transient_q_unmodified_and_stop_ties(routed):
    m, x, y = routed
    r = m.registry
    q = torch.zeros(2, r.actions)
    before = q.clone()
    legal = r.legal(1, 5, 1, "cpu")
    chosen, _ = action_selection(
        m, q, legal, 1, torch.tensor([0, 1]), torch.tensor([1, 0]), "CTX_ONCE_AND_NO_REPEAT"
    )
    assert chosen.tolist() == [r.stop, r.stop]
    assert torch.equal(q, before)
    with pytest.raises(ContinuationError, match="NO_LEGAL"):
        greedy(q, torch.zeros_like(legal))


@pytest.mark.parametrize("policy", ["CTX_ONCE", "NO_ADJACENT_SAME_CELL", "CTX_ONCE_AND_NO_REPEAT"])
def test_repeat_interventions(routed, policy):
    m, x, y = routed
    out = rollout(m, x, policy)
    for path in out.actions.reshape(-1, 5).tolist():
        active = [v for v in path if 0 <= v < m.registry.stop]
        if "CTX_ONCE" in policy:
            assert active.count(m.registry.ctx) <= 1
        if policy in ("NO_ADJACENT_SAME_CELL", "CTX_ONCE_AND_NO_REPEAT"):
            assert not any(a == b and a < m.registry.slots for a, b in zip(active, active[1:]))
    assert out.touched.all()


def test_forced_stop_is_not_budget_end(routed):
    m, x, _ = routed
    out = rollout(m, x, "FORCE_STOP_D1")
    assert out.lengths.tolist() == [[1] * 4]
    assert out.actual_stop.all() and not out.forced_end.any()
    assert out.actions[0, 0].tolist() == [0, m.registry.stop, -1, -1, -1]
    with pytest.raises(ContinuationError, match="STOP_ILLEGAL"):
        rollout(m, x, "FORCE_STOP_D0")


def test_frozen_plan_never_silently_overridden(routed):
    m, x, _ = routed
    m.registry.plan = [0, 0, m.registry.ctx]
    with torch.no_grad():
        native = m(x, policy="greedy", generator=torch.Generator().manual_seed(1))
    assert torch.equal(native.actions, rollout(m, x).actions)
    with pytest.raises(ContinuationError, match="PLAN_CONFLICT"):
        rollout(m, x, "NO_ADJACENT_SAME_CELL")
    with pytest.raises(ContinuationError, match="PLAN_CONFLICT"):
        rollout(m, x, "FORCE_STOP_D1")


def test_stop_oracle_no_operator_crn_and_chunk_independence(routed):
    m, x, y = routed
    with torch.no_grad():
        out, records = census(m, x, y, "s", "w")
    r = next(r for r in records if r["depth_zero_based"] == 1)
    r.update(design_weight=2.0, inclusion_probability=0.5)
    state = reconstruct(m, x, out.actions.reshape(-1, 5).tolist(), r)
    a = Snapshot(m)
    b = GreedySnapshot(m)
    c = configuration()
    rng = torch.get_rng_state().clone()
    u, seed = uniforms(r, c["mc_seed"], 8, 5, "cpu")
    value, trace, count = a.suffix(state, m.registry.stop, y.reshape(-1)[0], u[0])
    assert (
        count == 0
        and trace == [m.registry.stop]
        and value == a.terminal(state.hidden, y.reshape(-1)[0])
    )
    first = audit_state(b, a, state, y.reshape(-1)[0], r, c)
    uniforms({**r, "story_id": "different"}, c["mc_seed"], 8, 5, "cpu")
    second = audit_state(b, a, state, y.reshape(-1)[0], r, c)
    assert first == second and first["frozen_sampled_oracle"]["samples"] == 8
    assert torch.equal(rng, torch.get_rng_state())
    a.assert_integrity()
    a.assert_current(m)
