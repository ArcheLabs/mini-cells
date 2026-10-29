"""Release evidence and tiny complete workflow checks; no private data, network or GPU."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_protocol import PROJECT, ROOT
from minicells_r3me import data, evaluation, protocol, release, reports, runner, study
from minicells_r3me.common import atomic, canonical, digest, fingerprint, read
from minicells_r3m import anchor
from minicells_r3h.routing import Model
from minicells_r3h.trainer import Trainer
from minicells_r3c_moe_bridge.common import synthetic_configuration


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def released():
    return read(PROJECT / 'config/exploratory.released.json')


def fixture():
    c = released()
    runtime = anchor.runtime(c)
    runtime['engine'] = synthetic_configuration()
    runtime['engine']['routing']['cells'] = 6
    runtime['training']['tokens_per_step'] = 8
    c.update(tokens_per_step=8, continuation_steps=2, proposal_at=1, shadow_steps=1, barriers=[0, 1, 2])
    c['model']['sequence_length'] = 4
    c['training'].update(tokens_per_step=8, effective_batch=2, microbatch=1, sequence_length=4,
                         logical_updates=24, logical_tokens=192)
    c['evaluation'].update(arithmetic_groups=2, autoregressive_answer_token_cap=2,
                           cell_connection_grid_relative_steps=[1, 2])
    assets = {'train': np.tile(np.arange(2, 12, dtype=np.int32), 4),
              'windows': {'train': {'starts': [0, 4, 8, 12]}}}
    facts = [{'group': '11+12', 'carry': False, 'ids': [2, 3, 4, 5], 'prompt_ids': [2, 3],
              'answer_positions': [2, 3], 'answer': 23},
             {'group': '18+15', 'carry': True, 'ids': [6, 7, 8, 9], 'prompt_ids': [6, 7],
              'answer_positions': [2, 3], 'answer': 33}]
    research = {'story': {'development': [{'group': 'synthetic', 'windows': [[2, 3, 4, 5, 6]]}]},
                'arithmetic': {'train': facts, 'dev': facts}}
    # Synthetic fixture only; real fact Train/Dev isolation is independently covered.
    tokenizer = SimpleNamespace(decode=lambda ids: '0')
    return c, runtime, assets, research, tokenizer


def parent(runtime, seed):
    model = Model(runtime, seed, 'B-H')
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    tr = Trainer(model, runtime, 'B-H')
    tr.parent_identity = {'scope': 'SYNTHETIC_CPU_ONLY'}
    tr.step, tr.data_cursor = 300, 2400
    return tr


def test_release_retains_science_and_exact_reconstructed_hf_evidence(tmp_path):
    c = released()
    protocol.validate(c)
    assert runner.cpu_gate(c)['full_gpu_training_authorized'] is True
    assert c['authorization']['formal_confirmation'] is False
    local = read(PROJECT / 'release/gate2/SESSION_B.json')
    ref = c['release']['hf_reference']
    manifest_path = PROJECT / 'release/gate2/HF_REVIEW_MANIFEST.json'
    assert digest(manifest_path) == ref['manifest_sha256']
    remote_review = dict(local)
    remote_review.pop('hf_reference')
    payload = tmp_path / 'remote-review.json'
    atomic(payload, remote_review)
    manifest = read(manifest_path)
    row = manifest['files']['GATE2_REVIEW.json']
    assert digest(payload) == row['sha256']
    accesses = []
    def get(path, revision):
        accesses.append((path, revision))
        assert revision == ref['hf_revision']
        return manifest_path if path == ref['manifest_path'] else payload
    backend = SimpleNamespace(repo=ref['hf_repo'], get=get)
    proof = release.verify(c, backend)
    assert proof['status'] == 'EXPLORATORY_RELEASE_VERIFIED'
    assert proof['repeated_gpu_updates'] == 0
    assert len(accesses) == 2
    remote_review['comparison']['exact_rng_cursor_identity'] = False
    atomic(payload, remote_review)
    with pytest.raises(RuntimeError, match='REMOTE_BYTES_CHANGED'):
        release.verify(c, backend)


def test_twelve_job_small_workflow_and_json_report_barrier(tmp_path):
    c, runtime, assets, research, tokenizer = fixture()
    ident = dict(experiment_id='SYNTHETIC_R3ME', run_id='cpu-matrix', source_sha='cpu', config_sha256=canonical(c))
    for seed in c['seeds']:
        tr = parent(runtime, seed)
        state = copy.deepcopy(tr.state())
        for arm in c['arms']:
            frozen = SimpleNamespace(state=lambda: copy.deepcopy(state))
            aid = dict(ident, seed=seed, arm=arm)
            root = study.location(tmp_path, ident, seed, arm)
            execute = study.training_runner(c, tmp_path, ident, seed, arm, durable=lambda: None)
            final, aux = execute(c, runtime, frozen, aid, root, assets, research,
                'N' if arm == 'E0-STORY-STABLE' else 'C', runner._arm_runtime_name(arm), 'cpu')
            assert aux['main_updates'] == 2
            assert len(aux['transactions']) == (0 if arm in c['arms'][:2] else 1)
            assert study.completed(c, tmp_path, ident, seed, arm)
    study.training_barrier(c, tmp_path, ident)
    for seed, arm in study.jobs(c):
        result = runner.analyze_job(c, tmp_path, runtime, assets, research, tokenizer, ident, seed, arm, 'cpu')
        assert result['status'] == 'COMPLETE'
        assert result['checkpoint_receipts'][-1]['step'] == 2
        assert result['cost']['main_tokens'] == 16
        if arm == 'E1-MIX-STABLE':
            assert set(result['four_grid']['2']) == {'OO', 'ON', 'NO', 'NN'}
    report = reports.write(c, tmp_path, ident, {'anchors': []})
    assert report['status'] == 'COMPLETE_EXPLORATORY'
    assert report['completed_jobs'] == 12
    assert all((tmp_path / 'reports' / name).is_file() for name in reports.ARTIFACTS)
    # Exercise the existing Incremental publisher, namespace and selective restore.
    import sys
    sys.path.insert(0, str(ROOT / 'tests/engineering'))
    from test_hf_storage import FakeHub
    from minicells_r3me.storage import SnapshotPublisher
    from minicells_r3m.storage import restore as restore_private
    backend = FakeHub(tmp_path / 'fake-hub')
    writer = SnapshotPublisher(backend, tmp_path / 'private-checkpoints' / ident['run_id'],
                               tmp_path, ident, 'writer-A')
    ref = writer.publish(True)
    assert backend.optional('r3m-e/cpu-matrix/latest.json', ref['hf_revision']) is not None
    restored = tmp_path / 'restored'
    restored.mkdir()
    restore_private(backend, ref, restored / 'private-checkpoints' / ident['run_id'],
                    restored, ident, 'reader-B', c['barriers'])
    study.training_barrier(c, restored, ident)
    assert reports.write(c, restored, ident, {})['status'] == 'COMPLETE_EXPLORATORY'
    # Dropping one job must never produce a complete scientific report.
    missing = study.location(tmp_path, ident, c['seeds'][0], c['arms'][0]) / 'EVALUATION.json'
    missing.unlink()
    partial = reports.write(c, tmp_path, ident, {})
    assert partial['status'] == 'PARTIAL'
    assert read(tmp_path / 'reports/LONG_HORIZON_VALUE.json')['decision']['status'] == 'INCONCLUSIVE'


def test_path_diversity_counts_individual_token_paths():
    out = SimpleNamespace(actions=torch.tensor([[[1, 2], [2, 3]], [[1, 2], [4, 5]]]))
    assert set(evaluation._path_keys(out)) == {(1, 2), (2, 3), (4, 5)}


def test_gradient_probe_handles_cells_used_by_only_one_task_and_preserves_state(monkeypatch):
    import types
    from minicells_r3c_moe_bridge import data as existing_data
    c, runtime, assets, research, tokenizer = fixture()
    tr = parent(runtime, 77)
    original_forward, original_select = tr.model.forward, tr.model.select
    task = [0]
    def forward(self, tokens, **kwargs):
        task[0] = 0 if int(tokens[0, 0]) == 2 else 1
        return original_forward(tokens, **kwargs)
    def select(self, h, t, previous, **kwargs):
        kwargs['forced'] = torch.full((len(h),), task[0], device=h.device, dtype=torch.long)
        return original_select(h, t, previous, **kwargs)
    tr.model.forward = types.MethodType(forward, tr.model)
    tr.model.select = types.MethodType(select, tr.model)
    batch = lambda token: [(torch.full((1, 4), token, dtype=torch.long), torch.ones((1, 4), dtype=torch.long))]
    monkeypatch.setattr(existing_data, 'train_batches', lambda *args: batch(2))
    monkeypatch.setattr(data, 'arithmetic_batches', lambda *args: batch(6))
    before = fingerprint(tr.state())
    result = evaluation.gradient_conflict(tr, assets, research, 77, c)
    assert result['cells'][0]['story_norm'] > 0
    assert result['cells'][0]['arithmetic_norm'] == 0
    assert result['cells'][1]['story_norm'] == 0
    assert result['cells'][1]['arithmetic_norm'] > 0
    assert fingerprint(tr.state()) == before


def test_result_publication_uses_current_main_and_only_aggregate_files(tmp_path, monkeypatch):
    import subprocess
    from minicells_r3m import publication as inherited
    bare, seed = tmp_path / 'remote.git', tmp_path / 'main-seed'
    def command(*args, cwd=None):
        return subprocess.run(list(map(str, args)), cwd=cwd, text=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command('git', 'init', '--bare', bare)
    command('git', 'init', seed)
    command('git', 'symbolic-ref', 'HEAD', 'refs/heads/main', cwd=seed)
    command('git', 'config', 'user.name', 'CPU Test', cwd=seed)
    command('git', 'config', 'user.email', 'cpu@example.invalid', cwd=seed)
    (seed / 'README.md').write_text('Synthetic publication fixture\n')
    command('git', 'add', 'README.md', cwd=seed)
    command('git', 'commit', '-m', 'main fixture', cwd=seed)
    command('git', 'remote', 'add', 'origin', bare, cwd=seed)
    command('git', 'push', 'origin', 'main', cwd=seed)
    main_sha = command('git', 'rev-parse', 'HEAD', cwd=seed).stdout.strip()
    output = tmp_path / 'result'
    for name in reports.ARTIFACTS:
        path = output / 'reports' / name
        if path.suffix == '.json':
            atomic(path, {'scope': 'SYNTHETIC_CPU_ONLY', 'status': 'COMPLETE_EXPLORATORY'})
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Synthetic CPU report\n')
    (output / 'reports/not-for-publication.pt').write_bytes(b'private fixture')
    def git(checkout, *args):
        values = list(args)
        if values[:3] == ['remote', 'add', 'origin']:
            values[3] = str(bare)
        return command('git', *values, cwd=checkout)
    monkeypatch.setattr(inherited, 'git', git)
    monkeypatch.setattr(inherited, 'push_results',
                        lambda checkout, branch: git(checkout, 'push', 'origin', 'HEAD:refs/heads/' + branch))
    published = study.publication(output)
    assert published['status'] == 'PUBLISHED'
    remote_sha = command('git', '--git-dir', bare, 'rev-parse', published['branch']).stdout.strip()
    assert remote_sha == published['remote_commit_sha']
    assert command('git', '--git-dir', bare, 'rev-parse', remote_sha + '^').stdout.strip() == main_sha
    changed = command('git', '--git-dir', bare, 'diff', '--name-only', main_sha, remote_sha).stdout.splitlines()
    assert {Path(path).name for path in changed} == set(reports.ARTIFACTS)
    assert all('/r3m-e/reports/published/' in path for path in changed)
    assert study.publication(output)['remote_commit_sha'] == remote_sha
