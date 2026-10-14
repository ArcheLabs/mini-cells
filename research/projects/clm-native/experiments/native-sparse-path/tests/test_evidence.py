import copy
import json
import numpy as np
import pytest
import torch
from native_sparse_path.config import load
from native_sparse_path.generator import generate
from native_sparse_path.models import NativeModel
from native_sparse_path.profile import profile
from native_sparse_path.reuse import write_json, canonical_sha256, save_checkpoint, restore_checkpoint
from native_sparse_path.train import train
from native_sparse_path.report import paired_interval, analyze
from native_sparse_path.intervention import paired_rewire


def test_atomic_writer_creates_missing_parent(tmp_path):
    path = tmp_path / 'absent/deep/micro.json'
    write_json(path, {'ok': True})
    assert json.loads(path.read_text()) == {'ok': True}
    assert not path.with_suffix('.tmp').exists()


@pytest.mark.parametrize('arm', ['B0', 'B1', 'B2', 'C0'])
def test_cpu_short_train_checkpoint_and_resume_identity(tmp_path, arm):
    config = load()
    config.update(steps=3, batch_size=2, train_examples=8, eval_examples=4, checkpoint_every=2)
    result = train(config, arm, 'B', 0, tmp_path / arm, 'a' * 40)
    assert result['train_steps'] == 3 and result['scientific_decision'] is False
    checkpoint = tmp_path / arm / 'checkpoint.pt'
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['identity']['config_sha256'] == canonical_sha256(config)
    assert payload['step'] == 3
    assert payload['optimizer'] and payload['numpy_rng'] and payload['torch_rng'] is not None
    changed = copy.deepcopy(config)
    changed['dim'] = 33
    with pytest.raises(ValueError, match='identity mismatch'):
        train(changed, arm, 'B', 0, tmp_path / arm, 'a' * 40, resume=True)


def test_profiler_cross_checks_executed_linear_ops():
    model = NativeModel(load()).eval()
    instance = generate('B', 'dev', 0)
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                with_flops=True, record_shapes=True) as prof:
        _, ledger, _, _ = model(instance)
    observed = sum(e.flops or 0 for e in prof.key_averages()
                   if e.key in ('aten::mm', 'aten::addmm', 'aten::bmm'))
    assert observed == ledger['linear_ops']
    assert ledger['route_score_ops'] > 0
    assert ledger['total_ops_estimated'] >= observed


def test_paired_intervention_budget_and_restoration():
    result = paired_rewire(NativeModel(load()).eval(), load(), 'B', count=6)
    assert all(row['replay_parity'] and row['restoration_parity'] for row in result['rows'])
    assert all(row['equal_budget'] for row in result['rows'])


def test_analysis_rejects_unpaired_data_and_incomplete_matrix(tmp_path):
    with pytest.raises(ValueError, match='IDs differ'):
        paired_interval([{'id': 'a', 'mse': 1}], [{'id': 'b', 'mse': 1}])
    with pytest.raises(FileNotFoundError, match='incomplete screening'):
        analyze(tmp_path, load())


def test_reused_checkpoint_restores_rng_and_rejects_source_change(tmp_path):
    c = load()
    model = NativeModel(c)
    optimizer = torch.optim.AdamW(model.parameters())
    generator = np.random.default_rng(7)
    identity = {'source_commit': 'a' * 40, 'config_sha256': canonical_sha256(c)}
    save_checkpoint(tmp_path / 'saved.pt', model=model, optimizer=optimizer,
        step=2, scaler=None, identity=identity, generator=generator)
    expected = generator.integers(1000, size=8)
    torch_expected = torch.rand(8)
    step, _ = restore_checkpoint(tmp_path / 'saved.pt', model=model, optimizer=optimizer,
        scaler=None, expected_identity=identity, generator=generator)
    assert step == 2
    np.testing.assert_equal(generator.integers(1000, size=8), expected)
    torch.testing.assert_close(torch.rand(8), torch_expected)
    with pytest.raises(ValueError, match='identity mismatch'):
        restore_checkpoint(tmp_path / 'saved.pt', model=model, optimizer=optimizer, scaler=None,
            expected_identity={**identity, 'source_commit': 'b' * 40}, generator=generator)


def test_positive_single_seed_cannot_be_a_scientific_claim(tmp_path):
    config = load()
    for task in config['tasks']:
        for arm in config['screening_arms']:
            mse = 0.1 if arm == 'C0' else 0.3
            rows = [{'id': f'{task}-{i}', 'mse': mse, 'target': [0.2, 0.3]} for i in range(8)]
            identity = {'config_sha256': canonical_sha256(config), 'source_commit': 'a' * 40,
                        'code_sha256': 'b' * 64, 'data_sha256': task}
            metrics = {'identity': identity, 'arm': arm,
                'mse': mse, 'examples': rows, 'accuracy': 0.3, 'parameters_total': 9500,
                'train_seconds': 1, 'latency_p50_seconds': 1, 'latency_p95_seconds': 1,
                'elapsed_wall_time': 8, 'examples_per_sec': 1, 'checkpoint_sha256': 'c' * 64}
            write_json(tmp_path / task / arm / 'metrics.json', metrics)
            write_json(tmp_path / task / arm / 'profile.json', {'identity': identity,
                'checkpoint_sha256': 'c' * 64, 'uninstrumented': metrics})
    write_json(tmp_path / 'B/pooling-shortcut.json', {'accuracy': 0.2})
    decision = analyze(tmp_path, config)
    assert decision['m3_authorized_by_screening']
    assert decision['scientific_decision'] is False and decision['H1'] == 'INCONCLUSIVE'
    # Easy global shortcut invalidates the task rather than authorizing more compute.
    write_json(tmp_path / 'B/pooling-shortcut.json', {'accuracy': 0.9})
    decision = analyze(tmp_path, config)
    assert decision['status'] == 'IMPLEMENTATION_FAIL'
    assert not decision['m3_authorized_by_screening']


def test_publication_reuses_historical_authentication_components():
    import native_sparse_path.publish as native
    import publish_core_validation_007 as prior
    assert native.preflight_push is prior.preflight_push
    assert native.push_branch is prior.push_branch
