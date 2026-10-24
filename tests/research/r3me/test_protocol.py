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
    assert runner._arm_runtime_name("E0-STORY-STABLE") == "S0"
    assert runner._arm_runtime_name("E1-MIX-STABLE") == "S0"
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
