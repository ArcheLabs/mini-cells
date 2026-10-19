"""Synthetic orchestration rehearsal; no real-history acceptance is inferred."""

import copy
from types import SimpleNamespace
import numpy as np
from minicells_runtime.store import canonical, digest, read
from clm001_r3.common import canonical_sha256
from r3b_cf001 import driver
from r3b_cf001.common import configuration


def test_one_launch_synthetic_pipeline_and_idempotent_restart(routed, tmp_path, monkeypatch):
    model, x, y = routed
    runtime = model.experiment_config
    root = tmp_path / "input"
    root.mkdir()
    for name in (
        "runtime-config.json",
        "continuation.json",
        "experiment.lock.json",
        "checkpoint.pt",
    ):
        (root / name).write_text("synthetic fixture only")
    c = configuration()
    c["checkpoint_sha256"]["QC-R3b"] = digest(root / "checkpoint.pt")
    c["cf1_state_candidates"] = [8, 16, 32]
    c["bootstrap_replicates"] = 10
    c["pilot_states"] = 1
    data = {
        "development": np.tile(np.arange(2, 10), 100),
        "calibration": np.tile(np.arange(2, 10), 4),
        "asset_sha256": "SYNTHETIC_ASSETS",
        "windows": {
            "development": {"evaluation": {"starts": list(range(128)), "groups": list(range(128))}},
            "calibration": {"evaluation": {"starts": [0], "groups": [999]}},
        },
    }
    pairing = canonical_sha256(
        {
            "role": "development",
            "windows": data["windows"]["development"]["evaluation"],
            "seed": runtime["evaluation"]["seed"],
            "budget": 16,
            "policy": "greedy",
            "cost_lambda": 0.01,
        }
    )
    reference = {
        "identity": {"dataset_window_sha256": canonical(data["windows"]["development"])},
        "result": {"quality": {"greedy": {"groups": list(range(128)), "paired_identity": pairing}}},
    }
    monkeypatch.setattr(driver, "configuration", lambda: copy.deepcopy(c))
    monkeypatch.setattr(driver, "KaggleOutputAPI", lambda: object())
    monkeypatch.setattr(driver, "discover", lambda *a, **kw: ([{"version_number": 65}], {}))
    monkeypatch.setattr(
        driver,
        "audit_old",
        lambda *a: (
            {"status": "OLD_ANALYSIS_PERSISTENCE_VERIFIED", "synthetic_fixture": True},
            reference,
        ),
    )
    monkeypatch.setattr(driver, "locate_training", lambda *a: ({}, root, {}, {}))
    monkeypatch.setattr(
        driver, "load_model", lambda *a: (model, runtime, {"status": "SYNTHETIC_ONLY"})
    )
    monkeypatch.setattr(driver, "load_assets", lambda *a: data)
    monkeypatch.setattr(driver, "restore_saved", lambda *a: 0)
    monkeypatch.setattr(
        driver, "baseline_parity", lambda *a: {"status": "SYNTHETIC_ONLY_NOT_REAL_PARITY"}
    )
    monkeypatch.setattr(
        driver,
        "choose_budget",
        lambda seconds, c: {
            "selected_states": 32,
            "mc_samples": 8,
            "selection_data": "CALIBRATION_ONLY",
        },
    )
    import minicells_runtime.store as storage

    monkeypatch.setattr(
        storage,
        "MountedInputStore",
        lambda root: SimpleNamespace(select=lambda *a: {"checkpoint": "checkpoint.pt"}),
    )
    args = SimpleNamespace(
        output=tmp_path / "output",
        cache=tmp_path / "cache",
        orchestrator_root=tmp_path,
        device="cpu",
        publish_github=False,
        segment_seconds=10000,
    )
    result = driver.run(args, "SYNTHETIC_SOURCE")
    assert result["status"] == "COMPUTED_AWAITING_SAVE" and result["training_updates"] == 0
    coverage = read(args.output / "public/coverage-ledger.json")
    assert coverage["baseline_stories"] == 128 and coverage["cf1_states"] == 32
    assert set(coverage["intervention_stories"].values()) == {128}
    files = {p.name: digest(p) for p in (args.output / "chunks/objects").glob("*.json")}
    # A local restart reuses chunks but cannot claim cross-session Saved Output verification.
    second = driver.run(args, "SYNTHETIC_SOURCE")
    assert second["status"] == "COMPUTED_AWAITING_SAVE" and second["NEXT_RUN_REQUIRED"]
    assert files == {p.name: digest(p) for p in (args.output / "chunks/objects").glob("*.json")}
    assert read(args.output / "public/coverage-ledger.json")["cross_session_verified_chunks"] == 0


def test_old_audit_failure_blocks_model_and_all_new_cf(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "KaggleOutputAPI", lambda: object())
    monkeypatch.setattr(driver, "discover", lambda *a, **kw: ([], {}))
    monkeypatch.setattr(
        driver,
        "audit_old",
        lambda *a: ({"status": "NOT_VERIFIED", "verified_saved_results": 58}, None),
    )

    def forbidden(*a, **kw):
        raise AssertionError("model restore happened after failed preservation audit")

    monkeypatch.setattr(driver, "load_model", forbidden)
    import pytest
    from minicells_runtime.store import ContinuationError

    args = SimpleNamespace(
        output=tmp_path / "output",
        cache=tmp_path / "cache",
        orchestrator_root=tmp_path,
        device="cpu",
        publish_github=False,
        segment_seconds=10000,
    )
    with pytest.raises(ContinuationError, match="OLD_ANALYSIS_NOT_VERIFIED"):
        driver.run(args, "SYNTHETIC_SOURCE")
    assert not (args.output / "chunks").exists()
    assert (
        read(args.output / "public/old-analysis-preservation.json")["verified_saved_results"] == 58
    )
