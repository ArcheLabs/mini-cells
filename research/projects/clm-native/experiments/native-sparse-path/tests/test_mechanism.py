from dataclasses import replace
import numpy as np
import pytest
import torch
from native_sparse_path.config import load, authorize_sealed
from native_sparse_path.generator import generate, permute, FEATURES
from native_sparse_path.reference_solver import solve
from native_sparse_path.models import NativeModel, ARMS
from native_sparse_path.candidate import CandidateProvider
from native_sparse_path.accounting import Ledger
from native_sparse_path.route import EdgeBatch
from native_sparse_path.intervention import rewire_slots


@pytest.mark.parametrize('task', ['A', 'B', 'C'])
def test_independent_numpy_solver_against_literal_torch_executor(task):
    for index in range(6):
        instance = generate(task, 'dev', index, depth=3)
        x, lists = instance.x.double(), instance.candidates
        state = x[:, 7:9].clone()
        active = instance.roots.tolist()
        for _ in range(3):
            packets = []
            for source in active:
                value = torch.tanh(x[source, :2] * state[source] + x[source, 2:4])
                ids = lists[source]
                score = -x[ids, 6] if task == 'C' else x[ids, 4:6] @ value
                chosen = score.topk(2 if task == 'B' else 1).indices
                for slot in chosen:
                    packets.append((source, int(ids[slot]), value / len(chosen)))
            active = sorted({j for _, j, _ in packets}, key=lambda j: float(x[j, 6]))[:8]
            old = state.clone()
            for target in active:
                incoming = sum((v for _, j, v in packets if j == target), torch.zeros(2))
                state[target] = torch.tanh(0.3 * old[target] + incoming)
        expected = ((state[active] * x[active, 4:6]).sum(0) / len(active) ** 0.5
                    if task == 'B' else state[active].mean(0))
        np.testing.assert_allclose(solve(instance), expected.numpy(), atol=1e-7)


@pytest.mark.parametrize('task', ['A', 'B', 'C'])
def test_solver_permutation_determinism_and_split_isolation(task):
    a = generate(task, 'train', 5)
    duplicate = generate(task, 'train', 5)
    dev = generate(task, 'dev', 5)
    assert a.instance_id == duplicate.instance_id != dev.instance_id
    assert a.template_id != dev.template_id
    p = np.random.default_rng(7).permutation(64)
    np.testing.assert_allclose(solve(a), solve(permute(a, p)), atol=1e-7)
    assert a.candidates.shape == (64, 8)
    assert all(len(set(row.tolist())) == 8 for row in a.candidates)


@pytest.mark.parametrize('arm', ['B0', 'C0', 'B3'])
def test_model_permutation_and_true_hard_budget(arm):
    torch.manual_seed(42)
    model = NativeModel(load(), arm).eval()
    instance = generate('B', 'dev', 3)
    p = np.random.default_rng(9).permutation(64)
    with torch.no_grad():
        a, cost, trace, _ = model(instance, trace=True)
        b, _, _, _ = model(permute(instance, p))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert cost['global_pair_scores'] == 0
    assert cost['messages'] <= 4 * 8 * 2
    assert cost['cell_updates'] <= 4 * 8
    assert cost['candidate_reads'] <= 4 * 8 * 8
    for row in trace:
        for source, target in row['edges']:
            assert target in instance.candidates[source]
        assert len(row['targets']) <= 8


def test_candidate_private_state_not_scanned():
    instance = generate('B', 'dev', 0)
    active = instance.roots
    allowed = torch.cat([active, instance.candidates[active].flatten()]).unique()
    state = torch.full((64, 32), float('nan'))
    state[allowed] = 1
    candidates = CandidateProvider(8).candidates(active, state, instance, Ledger())
    assert torch.isfinite(candidates.states).all()
    bad = replace(instance, candidates=torch.cat([instance.candidates, instance.candidates[:, :1]], 1))
    with pytest.raises(ValueError, match='candidate limit'):
        CandidateProvider(8).candidates(active, state, bad, Ledger())


def test_edge_changes_only_actual_receivers_and_changes_downstream():
    model = NativeModel(load())
    instance = generate('B', 'dev', 0)
    state = torch.randn(64, 32)
    src = instance.roots[:1]
    target_a, target_b = instance.candidates[src[0], :2]
    ea = EdgeBatch(src, target_a.view(1), torch.ones(1), torch.zeros(1, 1, dtype=torch.long), torch.tensor(0.))
    eb = replace(ea, targets=target_b.view(1))
    sa = model.cell.step(state, ea, instance, Ledger(), active_limit=8)[0]
    sb = model.cell.step(state, eb, instance, Ledger(), active_limit=8)[0]
    mask = torch.ones(64, dtype=torch.bool)
    mask[target_a] = False
    torch.testing.assert_close(sa[mask], state[mask])
    assert not torch.allclose(sa[target_a], state[target_a])
    assert not torch.allclose(sa, sb)
    # Changed message source also changes the receiving state.
    altered = state.clone()
    altered[src] += 5
    sc = model.cell.step(altered, ea, instance, Ledger(), active_limit=8)[0]
    assert not torch.allclose(sc[target_a], sa[target_a])


def test_sparse_compute_scales_with_frontier_not_number_of_cells():
    model = NativeModel(load()).eval()
    small = generate('B', 'dev', 4)
    large = replace(small, x=torch.cat([small.x, torch.zeros(64, FEATURES)]),
                    candidates=torch.cat([small.candidates, small.candidates]))
    with torch.no_grad():
        _, a, _, _ = model(small)
        _, b, _, _ = model(large)
    assert a['candidate_reads'] == b['candidate_reads']
    assert a['messages'] == b['messages']
    assert a['cell_updates'] == b['cell_updates']
    assert b['linear_ops'] - a['linear_ops'] == 2 * 64 * FEATURES * 32
    assert a['route_score_ops'] == b['route_score_ops']
    assert b['node_state_bytes'] == 2 * a['node_state_bytes']


def test_parameter_matching_without_unused_filler():
    counts = [sum(p.numel() for p in NativeModel(load(), arm).parameters()) for arm in ARMS]
    assert max(counts) < 1_000_000
    assert max(counts) / min(counts) < 1.05
    for arm in ARMS:
        model = NativeModel(load(), arm)
        output = model(generate('B', 'train', 5))[0]
        output.square().sum().backward()
        assert all(p.grad is not None for p in model.parameters())


def test_static_learned_router_is_optimized_not_random_fixed():
    torch.manual_seed(13)
    model = NativeModel(load(), 'B2')
    prediction = model(generate('B', 'train', 1))[0]
    prediction.sum().backward()
    assert model.router.source_bias.weight.grad.abs().sum() > 0
    assert model.router.target_bias.weight.grad.abs().sum() > 0


def test_no_oracle_model_dependency(monkeypatch):
    import native_sparse_path.reference_solver as solver
    monkeypatch.setattr(solver, 'solve', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('oracle forbidden')))
    instance = generate('B', 'dev', 0)
    assert not hasattr(instance, 'target') and not hasattr(instance, 'correct_path')
    output = NativeModel(load())(instance)[0]
    assert torch.isfinite(output).all()


def test_sealed_access_and_malformed_registration_are_blocked():
    with pytest.raises(PermissionError):
        generate('B', 'sealed_test', 0)
    with pytest.raises(PermissionError):
        authorize_sealed(load(), None, 'a' * 64)
    with pytest.raises(PermissionError):
        authorize_sealed(load(), {'approval': 'APPROVED'}, 'a' * 64)


def test_joint_cell_and_router_gradients_are_finite():
    torch.manual_seed(2)
    model = NativeModel(load(), 'C0')
    output, _, _, aux = model(generate('B', 'train', 3))
    (output.square().mean() - 0.002 * aux['entropy']).backward()
    for group in (model.router, model.cell, model.encoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters())
        assert sum(float(p.grad.abs().sum()) for p in group.parameters()) > 0


def test_mixed_precision_receiver_dtype():
    model = NativeModel(load()).eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(generate('B', 'dev', 0))[0]
    assert torch.isfinite(output).all()


def test_degree_preserving_rewire_fixture():
    trace = [{'candidate_ids': [[3, 4, 5], [3, 4, 6]], 'slots': [[0], [1]], 'targets': [3, 4]}]
    slots, changed = rewire_slots(trace, 7)
    assert changed == 2
    assert slots == [[[1], [0]]]


def test_fixed_graph_subcondition_is_actually_absolute_static():
    a = generate('B', 'train', 0, fixed_candidates=True)
    b = generate('B', 'dev', 17, fixed_candidates=True)
    torch.testing.assert_close(a.candidates, b.candidates)
    torch.testing.assert_close(a.roots, b.roots)
    assert not torch.equal(a.x, b.x)


def test_common_operator_initialization_is_identical():
    torch.manual_seed(8)
    local = NativeModel(load(), 'C0')
    torch.manual_seed(8)
    central = NativeModel(load(), 'B3')
    for a, b in zip(local.cell.parameters(), central.cell.parameters()):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(local.encoder.weight, central.encoder.weight)
    torch.testing.assert_close(local.readout.weight, central.readout.weight)


def test_straight_through_training_forward_matches_hard_eval():
    torch.manual_seed(3)
    model = NativeModel(load(), 'C0')
    instance = generate('B', 'train', 0)
    a = model(instance)[0].detach()
    model.eval()
    with torch.no_grad():
        b = model(instance, force_hard=True)[0]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
