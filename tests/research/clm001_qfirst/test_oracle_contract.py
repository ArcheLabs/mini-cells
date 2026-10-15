from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from clm001.model import Model
from clm001.training import q_loss
from clm001.oracle import Snapshot, OracleUntrusted
from clm001.sampling import inverse_cdf, proposals, distribution
from test_core import probe, batch


@pytest.mark.parametrize("action", [0, 1, 3, 4])
def test_all_actions_exact_and_crn(config, action):
    model = Model(config, "QF-FULL")
    state, y = probe(model)
    snap = Snapshot(model)
    g = torch.Generator().manual_seed(190)
    u = torch.rand(1000, 3, generator=g)
    lab = snap.label(state, action, y, u)
    exact = snap.exact(state, action, y)
    stderr = (lab.variance / 1000) ** 0.5
    assert abs(lab.value - exact) < 5 * stderr + 1e-5
    again = snap.label(state, action, y, u)
    assert lab.record_sha == again.record_sha
    assert lab.uniforms_sha == snap.label(state, 0, y, u).uniforms_sha


def test_bellman(config):
    m = Model(config, "QF-FULL")
    s, y = probe(m)
    snap = Snapshot(m)
    a = 0
    h = snap.advance(s.hidden, a, s)
    depth = s.depth + 1
    p = snap.probabilities(h, a, depth)
    future = replace(s, hidden=h, previous=a, depth=depth)
    rhs = 0.01 / 3 + sum(
        float(p[j]) * snap.exact(future, j, y) for j in range(m.registry.actions) if p[j] > 0
    )
    assert snap.exact(s, a, y) == pytest.approx(rhs, abs=1e-6)


@pytest.mark.parametrize("tamper", ["policy", "weight", "registry", "state", "target"])
def test_fail_closed(config, tamper):
    m = Model(config, "QF-FULL")
    s, y = probe(m)
    snap = Snapshot(m)
    u = torch.rand(2, 3)
    lab = snap.label(s, 0, y, u)
    if tamper == "policy":
        snap.policy["epsilon"] = 1.0
        with pytest.raises(OracleUntrusted):
            snap.label(s, 0, y, u)
    elif tamper == "weight":
        with torch.no_grad():
            next(snap.model.parameters()).add_(1)
        with pytest.raises(OracleUntrusted):
            snap.label(s, 0, y, u)
    elif tamper == "registry":
        snap.model.registry.activate(2)
        with pytest.raises(OracleUntrusted):
            snap.label(s, 0, y, u)
    elif tamper == "state":
        s.hidden = s.hidden + 1
        with pytest.raises(OracleUntrusted):
            lab.verify(snap, s, y)
    else:
        with pytest.raises(OracleUntrusted):
            lab.verify(snap, s, y + 1)


def test_fake_stop_and_reserved(config):
    m = Model(config, "QF-FULL")
    s, y = probe(m)
    snap = Snapshot(m)
    with pytest.raises(OracleUntrusted):
        snap.label(s, m.registry.cells, y, torch.rand(2, 3))
    fake = replace(s, depth=0)
    with pytest.raises(OracleUntrusted):
        snap.label(fake, m.registry.stop, y, torch.rand(2, 3))


def test_q_cost_orientation_and_coverage(config):
    cost = torch.tensor([[3.0, 1.0, 9.0]])
    legal = torch.tensor([True, True, False])
    assert distribution(cost, legal, 0.15, 1.0, "greedy").argmax() == 1
    probabilities = distribution(cost, legal, 0.15, 1.0)
    assert (
        probabilities[0, 0] >= 0.075 and probabilities[0, 1] >= 0.075 and probabilities[0, 2] == 0
    )
    p = proposals(torch.tensor([True, True, True, True]), 0, 3, 1, torch.Generator().manual_seed(1))
    assert p[:2] == [(0, 1.0), (3, 1.0)] and p[2][1] == 0.5


def test_crn_redecides_not_action_reuse(config):
    a = torch.tensor([[0.8, 0.2]])
    b = torch.tensor([[0.2, 0.8]])
    u = torch.tensor([0.5])
    assert inverse_cdf(a, u).item() == 0 and inverse_cdf(b, u).item() == 1


def test_continuous_finite_difference(config):
    model = Model(config, "QF-FULL").double()
    x, y = batch()
    replay = torch.zeros(1, 4, 3, dtype=torch.long)
    p = model.cells[0].up.weight
    model.zero_grad()
    loss = F.cross_entropy(model(x, replay=replay).logits.flatten(0, 1), y.flatten())
    loss.backward()
    index = tuple(torch.unravel_index(p.grad.abs().argmax(), p.shape))
    analytic = float(p.grad[index])
    original = float(p[index])
    eps = 1e-5
    with torch.no_grad():
        p[index] = original + eps
        hi = float(F.cross_entropy(model(x, replay=replay).logits.flatten(0, 1), y.flatten()))
        p[index] = original - eps
        lo = float(F.cross_entropy(model(x, replay=replay).logits.flatten(0, 1), y.flatten()))
        p[index] = original
    assert (hi - lo) / (2 * eps) == pytest.approx(analytic, rel=1e-3, abs=1e-6)


def test_conditional_q_learning_not_false_specialization(config):
    # Fixed non-language two-state heterogeneous cost table: input state alone identifies the best action.
    model = Model(config, "QF-FULL")
    connection = model.controller
    h = torch.zeros(2, 8)
    h[0, 0] = -1
    h[1, 0] = 1
    legal = torch.ones(model.registry.actions, dtype=torch.bool)
    prev = torch.tensor([model.registry.start] * 2)
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    opt = torch.optim.Adam(connection.parameters(), lr=0.03)
    for _ in range(150):
        q, _ = connection(h, prev, 1, legal)
        loss = F.smooth_l1_loss(q[:, :2], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    q, _ = connection(h, prev, 1, legal)
    assert q[:, :2].argmin(-1).tolist() == [0, 1]
    # Truly homogeneous actions: exact counterfactual costs are equal, independent of learned Q noise.
    m = Model(config, "QF-FULL")
    m.cells[1].load_state_dict(m.cells[0].state_dict())
    s, y = probe(m)
    snap = Snapshot(m)
    # Equal Cell functions alone aren't enough if predecessor-conditioned suffix policy differs: force uniform suffix scores.
    with torch.no_grad():
        for p in m.controller.parameters():
            p.zero_()
    s, y = probe(m)
    snap = Snapshot(m)
    assert snap.exact(s, 0, y) == pytest.approx(snap.exact(s, 1, y), abs=1e-7)


def test_real_oracle_state_conditional_action_ranking_learns(config):
    from torch import nn

    class ConstantCell(nn.Module):
        def __init__(self, sign):
            super().__init__()
            v = torch.zeros(8)
            v[0] = sign * 24
            self.register_buffer("delta", v)

        def forward(self, h):
            return self.delta.expand_as(h)

    m = Model(config, "QF-FULL")
    m.cells[0] = ConstantCell(1)
    m.cells[1] = ConstantCell(-1)
    with torch.no_grad():
        m.embedding.weight.zero_()
        m.embedding.weight[0, 0] = 2
        m.embedding.weight[1, 0] = -2
    state, _ = probe(m)
    ha = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hb = torch.tensor([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    states = [replace(state, hidden=ha, depth=2), replace(state, hidden=hb, depth=2)]
    targets = [torch.tensor(0), torch.tensor(1)]
    snap = Snapshot(m)
    rows = []
    for s, y in zip(states, targets):
        labels = [snap.label(s, a, y, torch.rand(2, 3)) for a in (0, 1)]
        assert labels[y.item()].value < labels[1 - y.item()].value
        rows.extend((s, y, label) for label in labels)
    opt = torch.optim.Adam(m.controller.parameters(), lr=0.025)
    for _ in range(150):
        loss, _ = q_loss(m, rows, snap)
        opt.zero_grad()
        loss.backward()
        opt.step()
    legal = m.registry.legal(2, 3, 1, "cpu")
    for s, y in zip(states, targets):
        q, _ = m.controller(
            s.hidden[None], torch.tensor([s.previous]), 2, legal, m.registry.version
        )
        assert int(q[0, :2].argmin()) == int(y)
