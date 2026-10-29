import copy
from pathlib import Path
import pytest
import torch
from minicells_r3e.common import fingerprint
from minicells_jct001 import gpu_runner
from minicells_jct001.gpu_training import DeviceTrainer
from minicells_jct001.data import unit_data, feature_tensor
from minicells_jct001.common import read, atomic


def test_pilot_config_is_frozen_without_releasing_real_model():
    p, c = gpu_runner.pilot_configuration()
    assert p["seeds"] == [731001, 731002, 731003]
    assert p["logical_updates_per_arm"] == 200 and p["arms"] == ["A", "C"]
    assert p["fixture"] == c["fixture"]
    assert not p["real_model_authorized"] and not p["confirmation_authorized"]


def test_gpu_entry_has_no_cpu_scientific_fallback(config, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    p, _ = gpu_runner.pilot_configuration()
    with pytest.raises(RuntimeError, match="GPU_REQUIRED"):
        gpu_runner.run(p, config, tmp_path, device="cuda:0")
    assert not (tmp_path / "RUN_MANIFEST.json").exists()


def test_device_adapter_cpu_continuation(config, identity, tmp_path):
    config["fixture"]["logical_updates"] = 2
    base = DeviceTrainer(config, "BASE", identity, tmp_path / "base.jsonl", device="cpu").state()
    trainer = DeviceTrainer(config, "C", identity, tmp_path / "C.jsonl", base, device="cpu")
    data = unit_data(config["fixture"])
    data["reference_features"] = feature_tensor([dict(x=0, a=0, b=0)])
    trainer.advance(data, tmp_path / "state.pt")
    trainer.save(tmp_path / "boundary.pt")
    trainer.advance(data, tmp_path / "state.pt")
    expected = fingerprint(trainer.state())
    restored = DeviceTrainer(config, "C", identity, tmp_path / "C.jsonl", base, device="cpu")
    restored.restore(tmp_path / "boundary.pt")
    restored.advance(data, tmp_path / "state.pt")
    assert fingerprint(restored.state()) == expected
    assert restored.journal.summary()["completed_optimizer_calls"] == 6


def test_synthetic_gpu_flow_on_cpu_unit_only(config, identity, tmp_path, monkeypatch):
    class UnitBank:
        def __init__(self, fixture):
            self.fixture = fixture

        def registry(self):
            return dict(scope="CPU_UNIT_TEST_ONLY")

        def records(self, role):
            assert role in ("train", "calibration", "pilot_report")
            return [dict(x=0, a=0, b=0, target=32, canonical_id="unit-only", role=role)]

    monkeypatch.setattr(gpu_runner, "TaskBank", UnitBank)
    monkeypatch.setattr(gpu_runner, "run_identity", lambda c, r: copy.deepcopy(identity))
    p, _ = gpu_runner.pilot_configuration()
    p = copy.deepcopy(p)
    p["seeds"] = [919991]
    p["logical_updates_per_arm"] = 2
    result = gpu_runner.run(p, config, tmp_path, "cpu", publish_github=False, test_mode=True)
    root = Path(result["output"])
    manifest = read(root / "RUN_MANIFEST.json")
    assert result["status"] == "SYNTHETIC_PILOT_COMPLETE_EXPLORATORY_REVIEW_REQUIRED"
    assert not result["gpu_used"] and manifest["scope"] == "CPU_UNIT_TEST_ONLY"
    assert manifest["scientific_optimizer_updates"] == 4
    assert manifest["functional_preflight_optimizer_calls"] == 2
    assert read(root / "BUDGET_PROOF.json")["actual_optimizer_calls"] == 4
    assert not read(root / "SEED_SUMMARY.json")["formal_protocol_pilot_pass"]
    again = gpu_runner.run(p, config, tmp_path, "cpu", publish_github=False, test_mode=True)
    assert again["reused_completed_run"]
    manifest["run_status"] = "PILOT_STARTED"
    atomic(root / "RUN_MANIFEST.json", manifest)
    resumed = gpu_runner.run(p, config, tmp_path, "cpu", publish_github=False, test_mode=True)
    assert resumed["scientific_optimizer_updates"] == 4
    assert (
        read(root / "COMPUTE_COST.json")["seeds"]["919991"]["C"]["cuda_timing_coverage"]
        == "COMPLETE_RETAINED_INTERVALS"
    )
    from minicells_jct001.gpu_publication import payload

    with pytest.raises(RuntimeError, match="IDENTITY_INVALID"):
        payload(root)
    with pytest.raises(RuntimeError, match="CANNOT_PUBLISH"):
        gpu_runner.run(p, config, tmp_path, "cpu", publish_github=True, test_mode=True)


def test_oracle_feature_device_is_explicit(config, identity, tmp_path, monkeypatch):
    calls = []
    original = gpu_runner.feature_tensor

    def record(rows, device="cpu"):
        calls.append(str(device))
        return original(rows, device)

    monkeypatch.setattr(gpu_runner, "feature_tensor", record)
    model = DeviceTrainer(config, "BASE", identity, tmp_path / "base.jsonl", device="cpu").model
    result = gpu_runner.oracle_evaluate(
        model, [dict(x=0, a=0, b=0, target=32)], torch.device("cpu")
    )
    assert result["accuracy"] == 1 and calls == ["cpu"]
