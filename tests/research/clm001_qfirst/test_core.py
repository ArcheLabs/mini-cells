import pytest
import torch
from clm001.model import Model
from clm001.oracle import Snapshot, OracleUntrusted
from clm001.training import Trainer, q_loss
from clm001.common import hash_accepted_state
from clm001.shadows import Nursery
from clm001.admission import assess, promote


def batch():
    return torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]])


def probe(model):
    x, y = batch()
    r = torch.zeros(1, 4, 3, dtype=torch.long)
    out = model(x, replay=r, probe_coordinates=[(0, 1)], generator=torch.Generator().manual_seed(1))
    return out.probes[0], y[0, 0]


def test_matched_inventory(config):
    a, b = Model(config, "QF-FULL"), Model(config, "J0-FULL")
    assert hash_accepted_state(a) == hash_accepted_state(b)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_stop_and_forced(config):
    m = Model(config, "QF-FULL")
    x, _ = batch()
    r = torch.zeros(1, 4, 3, dtype=torch.long)
    r[:, :, 1] = m.registry.stop
    r[:, :, 2] = -1
    out = m(x, replay=r)
    assert out.actual_stop.all() and not out.forced_end.any()
    assert out.lengths.eq(1).all()
    assert out.cost_units.eq(1 / 3).all()
    out = m(x, replay=torch.zeros_like(r))
    assert out.forced_end.all() and not out.actual_stop.any()
    assert out.lengths.eq(3).all()
    r[:, :, 0] = m.registry.stop
    with pytest.raises(ValueError):
        m(x, replay=r)


def test_nonselected_never_execute(config):
    m = Model(config, "QF-FULL")
    calls = [0] * 3
    handles = [
        cell.register_forward_hook(lambda a, b, c, i=i: calls.__setitem__(i, calls[i] + 1))
        for i, cell in enumerate(m.cells)
    ]
    m(batch()[0], replay=torch.zeros(1, 4, 3, dtype=torch.long))
    assert calls == [3, 0, 0]
    for h in handles:
        h.remove()


def test_oracle_stop_identity_and_full_suffix(config):
    m = Model(config, "QF-FULL")
    s, y = probe(m)
    snap = Snapshot(m)
    u = torch.rand(100, 3)
    label = snap.label(s, m.registry.stop, y, u)
    assert label.value == pytest.approx(snap.terminal(s.hidden, y))
    assert all(t == [m.registry.stop] for t in label.traces)
    a = snap.label(s, 0, y, u)
    assert a.token_work >= 100
    assert any(len(t) > 1 for t in a.traces)
    snap.assert_current(m)


def test_exact_mc(config):
    m = Model(config, "QF-FULL")
    s, y = probe(m)
    snap = Snapshot(m)
    # Depth 1 leaves two total decisions: enumerate every possible suffix next action.
    exact = snap.exact(s, 0, y)
    values = []
    for u in torch.linspace(0.0005, 0.9995, 1000):
        stream = torch.full((3,), float(u))
        values.append(snap.suffix(s, 0, y, stream)[0])
    assert abs(sum(values) / len(values) - exact) < 0.002


def test_q_gradient_isolated(config):
    m = Model(config, "QF-FULL")
    s, y = probe(m)
    snap = Snapshot(m)
    lab = snap.label(s, 0, y, torch.rand(2, 3))
    m.zero_grad()
    loss, critic = q_loss(m, [(s, y, lab)], snap)
    (loss + critic).backward()
    assert any(p.grad is not None for p in m.controller.parameters())
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith("controller."))


def test_label_tamper_and_stale(config):
    m = Model(config, "QF-FULL")
    s, y = probe(m)
    snap = Snapshot(m)
    lab = snap.label(s, 0, y, torch.rand(2, 3))
    lab.value += 1
    with pytest.raises(OracleUntrusted):
        lab.verify(snap, s, y)
    m.registry.version += 1
    with pytest.raises(OracleUntrusted):
        snap.assert_current(m)


@pytest.mark.parametrize("arm", ["QF-FULL", "J0-FULL"])
def test_training_and_independent_rng(config, arm):
    m = Model(config, arm)
    t = Trainer(m)
    rng = t.rng["evaluation"].get_state().clone()
    before = hash_accepted_state(m)
    row, _, labels = t.update([batch()])
    assert hash_accepted_state(m) != before
    assert torch.equal(rng, t.rng["evaluation"].get_state()) and row["candidate_labels"] <= 8
    assert (row["q_loss"] is None) == (arm == "J0-FULL")
    assert m.registry.model_version == 1 and not t.cache
    with pytest.raises(ValueError):
        t.update([batch()], role="admission")


def test_shadow_and_atomic_rollback(config):
    m = Model(config, "QF-FULL")
    t = Trainer(m)
    nursery = Nursery()
    before = hash_accepted_state(m)
    candidate = nursery.propose(m, "cell", t.opt)
    nursery.probation(candidate, m, [batch()], 1)
    assert hash_accepted_state(m) == before and candidate.model.registry.activated == [2]
    base = {"paired_identity": "fixed", "nll": [3.0, 3.0], "cost": [1.0, 1.0]}
    new = {"paired_identity": "fixed", "nll": [2.0, 2.0], "cost": [1.0, 1.0]}
    assert assess(candidate, m, base, new, [0, 1], config)

    def fail():
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError):
        promote(candidate, m, t, failure=fail)
    assert (
        hash_accepted_state(m) == before
        and m.registry.activated == []
        and candidate.state == "EVALUATED"
    )
    promote(candidate, m, t)
    assert m.registry.activated == [2] and candidate.state == "PROMOTED"
    with pytest.raises(ValueError):
        nursery.propose(m, "cell")


def test_causal_incremental(config):
    m = Model(config, "QF-FULL").eval()
    x, _ = batch()
    r = torch.zeros(1, 4, 3, dtype=torch.long)
    r[:, :, 0] = m.registry.ctx
    with torch.no_grad():
        full = m(x, replay=r)
        memory = None
        pieces = []
        for i in range(4):
            out = m(x[:, i : i + 1], past=memory, replay=r[:, i : i + 1])
            memory = out.memory
            pieces.append(out.logits)
        assert torch.allclose(full.logits, torch.cat(pieces, 1), atol=1e-6)
        x2 = x.clone()
        x2[0, -1] = 6
        assert torch.allclose(full.logits[:, :-1], m(x2, replay=r).logits[:, :-1], atol=1e-6)


def test_proposed_full_size_sixteen_real_calls():
    from clm001.common import load
    from torch.nn import functional as F

    c = load()
    m = Model(c, "QF-FULL")
    assert sum(p.numel() for p in m.parameters()) == 1075492
    x = torch.tensor([[1, 2]])
    r = torch.zeros(1, 2, 16, dtype=torch.long)
    out = m(x, replay=r)
    assert out.lengths.eq(16).all() and out.forced_end.all() and not out.actual_stop.any()
    F.cross_entropy(out.logits.flatten(0, 1), torch.tensor([2, 3])).backward()
    assert m.cells[0].up.weight.grad.abs().sum() > 0
    assert m.cells[1].up.weight.grad is None
    assert all(p.grad is None for p in m.controller.parameters())


def test_shadow_corruption_and_insufficient_evidence_fail_closed(config):
    m = Model(config, "QF-FULL")
    t = Trainer(m)
    n = Nursery()
    candidate = n.propose(m, "cell", t.opt)
    n.probation(candidate, m, [batch()], 1)
    with torch.no_grad():
        candidate.model.embedding.weight[0, 0] += 1
    base = {"paired_identity": "fixed", "nll": [3.0, 3.0], "cost": [1.0, 1.0]}
    challenger = {"paired_identity": "fixed", "nll": [2.0, 2.0], "cost": [1.0, 1.0]}
    with pytest.raises(ValueError):
        assess(candidate, m, base, challenger, [0, 1], config)
    assert candidate.state == "PROBATION"
