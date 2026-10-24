import json
from pathlib import Path
import runpy

import numpy as np
import torch
import pytest


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-e"
import sys
runpy.run_path(str(PROJECT / "run.py"), run_name="r3me_test_bootstrap")
sys.path.insert(0, str(PROJECT / "implementation"))

from minicells_r3me import arithmetic, data, protocol, runner  # noqa: E402


class CharacterTokenizer:
    class Encoded:
        def __init__(self, text):
            self.ids = [2 + (ord(ch) % 127) for ch in text]
            self.offsets = [(i, i + 1) for i in range(len(text))]

    def encode(self, text):
        return self.Encoded(text)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_frozen_config_cpu_gate_and_fixed_matrix():
    config = json.loads((PROJECT / "config/exploratory.frozen.json").read_text())
    protocol.validate(config)
    gate = runner.cpu_gate(config)
    assert gate["status"] == "CPU_GATE_PASS"
    assert gate["task_counts_per_phase"] == {"0": 10, "100": 30, "200": 100, "400": 100}
    assert gate["logical_updates"] == 7200
    assert gate["logical_tokens"] == 3_686_400
    assert gate["confirmation"] == "SEALED"


def test_arithmetic_fact_level_split_is_disjoint_and_balanced():
    rows, audit = arithmetic.generate(CharacterTokenizer(), max_length=64)
    train = {row["group"] for row in rows["train"]}
    dev = {row["group"] for row in rows["dev"]}
    assert len(train) + len(dev) == 4095
    assert not train.intersection(dev)
    assert audit["train_dev_fact_overlap"] == 0
    assert set(audit["carry_strata"]["train"]) == {"carry", "no_carry"}
    assert set(audit["carry_strata"]["dev"]) == {"carry", "no_carry"}
    assert all(row["answer_tokens"] for row in rows["dev"])
    assert all(row["ids"][-1] != 1 for row in rows["dev"])


def test_task_schedule_has_exact_per_phase_allocation():
    assert sum(protocol.task_for(i) == "arithmetic" for i in range(0, 100)) == 10
    assert sum(protocol.task_for(i) == "arithmetic" for i in range(100, 200)) == 30
    assert sum(protocol.task_for(i) == "arithmetic" for i in range(200, 400)) == 100
    assert sum(protocol.task_for(i) == "arithmetic" for i in range(400, 600)) == 100
    with pytest.raises(ValueError, match="R3ME_RELATIVE_STEP_INVALID"):
        protocol.task_for(600)


def test_arithmetic_batches_add_delimiters_and_exact_token_budget():
    config = json.loads((PROJECT / "config/exploratory.frozen.json").read_text())
    records = [{"ids": [2, 3, 4], "group": "10+10"}]
    batches = data.arithmetic_batches(np.random.default_rng(4), records, config, TinyModel())
    assert len(batches) == 8
    assert sum(x.numel() for x, _ in batches) == 512
    assert all(x.shape == (1, 64) and y.shape == (1, 64) for x, y in batches)
    assert all(int(y[0, 2]) == config["arithmetic"]["eos_token_id"] for _, y in batches)


def test_schedule_continues_anchor_without_warmup_reset():
    config = json.loads((PROJECT / "config/exploratory.frozen.json").read_text())
    lr0, eps0 = protocol.schedule(0.001, 0.06, 0, config)
    lr_end, eps_end = protocol.schedule(0.001, 0.06, 599, config)
    assert lr0 == 0.001
    assert eps0 == 0.06
    assert abs(lr_end - 0.0002) < 1e-12
    assert abs(eps_end - 0.02) < 1e-12


def test_arm_runtime_mapping_preserves_sham_vs_real_proposal():
    assert runner._arm_runtime_name("E0-STORY-STABLE") == "STABLE"
    assert runner._arm_runtime_name("E1-MIX-STABLE") == "STABLE"
    assert runner._arm_runtime_name("E2-MIX-REORG") == "P1"
    assert runner._arm_runtime_name("E3-MIX-SHAM") == "P0"


def test_full_training_remains_disabled_by_frozen_config(tmp_path):
    config = json.loads((PROJECT / "config/exploratory.frozen.json").read_text())
    assert config["authorization"]["full_training"] is False
    with pytest.raises(RuntimeError, match="NOT_AUTHORIZED"):
        runner.run_study(config, tmp_path / "output", tmp_path / "assets")
    assert not (tmp_path / "output").exists()


def test_gpu_authorization_cannot_be_enabled_by_editing_the_frozen_config():
    config = json.loads((PROJECT / "config/exploratory.frozen.json").read_text())
    config["authorization"]["full_training"] = True
    with pytest.raises(ValueError, match="FROZEN_CONFIG_HASH_MISMATCH"):
        protocol.validate(config)


def test_story_selection_uses_historical_legal_windows_not_single_fixed_window():
    metadata_path = PROJECT.parent / 'r3m/registration/artifact-recovery-001/independent-review-001/windows.json'
    metadata = json.loads(metadata_path.read_text())['development']
    # Original fixed evaluation has one window/story; the legal pool supports two.
    assert len(set(metadata['evaluation']['groups'])) == len(metadata['evaluation']['starts'])
    tokens = np.zeros(metadata['story_offsets'][-1], dtype=np.int32)
    assets = {'windows': {'development': metadata}, 'development': tokens}
    selected = data._story_eval_assets(assets, 64, 32, 2)
    assert len(selected) == 32
    legal = set(zip(metadata['starts'], metadata['groups']))
    for row in selected:
        assert len(row['windows']) == len(set(row['starts'])) == 2
        assert all((start, int(row['group'])) in legal for start in row['starts'])
        assert all(len(window) == 65 for window in row['windows'])
    assert selected == data._story_eval_assets(assets, 64, 32, 2)


def test_cpu_parent_inspection_preserves_original_cuda_state(monkeypatch):
    from types import SimpleNamespace
    from minicells_r3me import parent
    from minicells_r3me.common import fingerprint
    state = {'global_cuda_rng': [torch.tensor([1, 2], dtype=torch.uint8)],
             'model': {'weight': torch.ones(2)}, 'cell_optimizer': {'step': 300}}
    original = fingerprint(state)
    captured = {}
    def restore(runtime, copied, seed, arm, device):
        captured.update(copied)
        assert device == 'cpu'
        return SimpleNamespace()
    monkeypatch.setattr(parent, 'parent_restore', restore)
    trainer = parent._restore_for_cpu_inspection({}, state, 151001301, 'B-H', 'cpu')
    assert captured['global_cuda_rng'] == []
    assert fingerprint(captured['model']) == fingerprint(state['model'])
    assert captured['cell_optimizer'] == state['cell_optimizer']
    assert fingerprint(state) == original
    assert trainer.cpu_inspection_evidence['original_cuda_rng_count'] == 1
    assert trainer.cpu_inspection_evidence['gpu_resume_verified'] is False
    with pytest.raises(RuntimeError, match='CPU_ONLY'):
        parent._restore_for_cpu_inspection({}, state, 151001301, 'B-H', 'cuda:0')


def test_gate2_mid_shadow_restore_and_forced_rollback_use_real_engine(tmp_path):
    import copy
    from minicells_r3me import gate2
    from minicells_r3m.anchor import runtime
    from minicells_r3h.routing import Model
    from minicells_r3h.trainer import Trainer
    from minicells_r3c_moe_bridge.common import synthetic_configuration
    from minicells_r3c_moe_bridge.data import synthetic_batches
    from minicells_r3me.common import atomic, fingerprint

    c = json.loads((PROJECT / 'config/exploratory.frozen.json').read_text())
    r = runtime(c)
    r['engine'] = synthetic_configuration()
    r['engine']['routing']['cells'] = 6
    r['training']['tokens_per_step'] = 8
    c['tokens_per_step'] = 8
    model = Model(r, 77, 'B-H')
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    parent = Trainer(model, r, 'B-H')
    parent.parent_identity = {'synthetic': True}
    parent.step, parent.data_cursor = 300, 2400
    before = fingerprint(parent.model.state_dict()), fingerprint(parent.cell_opt.state_dict())
    def batches(tr, historical, research, environment, config):
        return synthetic_batches(tr.c['engine'], tr.data_rng)
    proposal_calls = []
    def proposal(tr, seen, root, identity, persist):
        proposal_calls.append(len(seen))
        from minicells_r3m.intervention import propose
        return propose(tr, seen, root, identity, persist)
    identity = {'experiment_id': 'SYNTHETIC_GATE2', 'run_id': 'cpu', 'seed': 77}
    # The engine identity must include its seed, as production identities do.
    local = gate2.run_transaction_checks(c, r, parent, None, None, tmp_path / 'first', identity,
                                         'cpu', batches, proposal)
    assert proposal_calls == [1]  # restored middle transaction must not regenerate Proposal.
    assert local['paused_transaction_done'] == 3
    assert local['rollback']['exact_rng_cursor_identity']
    assert local['local_optimizer_updates'] == 26
    trace = json.loads((tmp_path / 'first/relay/PROPOSAL_TRACE.json').read_text())
    assert len(trace['trace']['attempts']) == 22
    assert trace['trace']['data_role'] == 'ALREADY_OBSERVED_TRAIN'
    atomic(tmp_path / 'first/relay/LOCAL_GATE2.json', local)
    # Reuse the shared fake Hub and actual immutable HF segment protocol.
    sys.path.insert(0, str(ROOT / 'tests/engineering'))
    from test_hf_storage import FakeHub
    from minicells_r3c_moe_bridge.checkpoint import publish_segment, restore_segment
    backend = FakeHub(tmp_path / 'hub')
    ref = publish_segment(backend, tmp_path / 'first/relay', identity, 'writer-A',
                          pointer_path='engineering/gate2-test/latest.json')
    proof = restore_segment(backend, ref, tmp_path / 'separate-directory', identity, 'reader-B')
    assert backend.writes == 1
    assert proof['cross_session_verified'] is True  # simulated transport, not real Kaggle evidence.
    with pytest.raises(Exception, match='INDEPENDENT_SESSION_REQUIRED'):
        restore_segment(backend, ref, tmp_path / 'same-writer', identity, 'writer-A')
    resumed = gate2.resume_and_compare(c, r, parent, None, None, tmp_path / 'separate-directory',
                                       identity, 'cpu', batches)
    assert resumed['status'] == 'GATE2_PASS'
    assert resumed['resumed_main_updates'] == resumed['resumed_shadow_updates'] == 7
    assert before == (fingerprint(parent.model.state_dict()), fingerprint(parent.cell_opt.state_dict()))
    assert proposal_calls == [1]
    # Local CPU restore does not create evidence of an actual cloud session transition.
    assert 'actual_cross_session_verified' not in resumed


def test_gate2_requires_exact_random_stream_even_with_float_tolerance():
    from minicells_r3me.gate2 import compare_states
    from minicells_r3m.training import semantic
    # A semantic-shaped fixture avoids model/training work for this corruption check.
    keys = ('model', 'cell_optimizer', 'data_rng', 'route_rng', 'cf_rng', 'step', 'data_cursor',
            'optimizer_identity', 'frozen_sha256', 'global_cpu_rng', 'global_cuda_rng',
            'python_rng', 'numpy_rng')
    state = {k: 0 for k in keys}
    state['model'] = {'w': torch.ones(2)}
    state['cell_optimizer'] = {}
    other = dict(state, data_cursor=1)
    with pytest.raises(RuntimeError, match='EXACT_STATE_MISMATCH:data_cursor'):
        compare_states(state, other, cuda=True)


def test_gate2_failed_publication_retries_without_optimizer_work(tmp_path):
    from minicells_r3me import gate2
    from minicells_r3me.common import atomic
    sys.path.insert(0, str(ROOT / 'tests/engineering'))
    from test_hf_storage import FakeHub
    backend = FakeHub(tmp_path / 'hub')
    output = tmp_path / 'output'
    export = output / 'export'
    atomic(export / 'review.json', {'scope': 'synthetic'})
    identity = {'experiment_id': 'CPU_UPLOAD_ONLY', 'run_id': 'retry'}
    result = {'status': 'LOCAL_GATE2_PASS_CROSS_SESSION_PENDING', 'identity': identity}
    backend.fail = True
    with pytest.raises(PermissionError):
        gate2.publish_review(backend, export, output, identity, 'engineering/retry/latest.json', result)
    backend.fail = False
    recovered = gate2.retry_upload(backend, output, identity)
    assert recovered['upload_retried_without_optimizer_updates'] is True
    assert recovered['status'] == result['status']
    assert backend.writes == 1
    assert gate2.retry_upload(backend, output, identity) is None
