from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"


def _load_script(name: str, rel: str):
    path = ROOT / rel
    script_dir = str(path.parent)
    added = script_dir not in sys.path
    if added:
        sys.path.insert(0, script_dir)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            sys.path.remove(script_dir)


def test_confirmation_amendment_retires_exposed_seed_set_without_gate_changes() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    retired = set(amendment["retired_confirmation_seeds"])
    fresh = set(amendment["confirmation_seeds"])
    assert retired == {80711, 80712, 80713}
    assert fresh == {80721, 80722, 80723}
    assert retired.isdisjoint(fresh)
    assert amendment["scientific_invariants"] == {
        "winner_changed": False,
        "boundary_mechanism_changed": False,
        "model_or_data_changed": False,
        "gate_thresholds_changed": False,
        "core006_baselines_changed": False,
    }
    assert amendment["winner"] == "interference_cut"
    assert amendment["expected_data_manifest_sha256"] == (
        "d098f9172083b8de9f825b66de5277dde5b6ea0581b3a950b8f76e4f443546cc"
    )


def test_seed_checkpoint_identity_refuses_mismatch(tmp_path: Path) -> None:
    module = _load_script(
        "core007_seed_runner",
        "scripts/research/run_core_validation_007_confirmation_seed.py",
    )
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "seed-80721.json"
    payload = {
        "complete": True,
        "seed": 80721,
        "phase": "confirmation",
        "confirmation_protocol_sha256": "amend",
        "base_protocol_sha256": amendment["base_discovery_protocol_sha256"],
        "data_manifest_sha256": amendment["expected_data_manifest_sha256"],
        "winner": amendment["winner"],
        "scientific_code_sha256": "science",
    }
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    assert module._checkpoint_matches(
        checkpoint,
        seed=80721,
        amendment_sha="amend",
        scientific_sha="science",
        amendment=amendment,
    )
    assert not module._checkpoint_matches(
        checkpoint,
        seed=80721,
        amendment_sha="amend",
        scientific_sha="different",
        amendment=amendment,
    )


def test_partial_report_is_explicitly_non_scientific(tmp_path: Path) -> None:
    out = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/research/report_core_validation_007_confirmation.py"),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    decision = json.loads(
        (out / "confirmation" / "decision.json").read_text(encoding="utf-8")
    )
    assert decision["status"] == "CONFIRMATION_INCOMPLETE"
    assert decision["scientific_decision"] is False
    assert decision["completed_seeds"] == []
    assert decision["pending_seeds"] == [80721, 80722, 80723]


def test_hydrate_restores_seed_checkpoint_across_sessions(tmp_path: Path) -> None:
    module = _load_script(
        "core007_orchestrator",
        "scripts/research/orchestrate_core_validation_007_confirmation.py",
    )
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    source = artifacts / "confirmation" / "seeds"
    source.mkdir(parents=True)
    (source / "seed-80721.json").write_text('{"complete": true}\n', encoding="utf-8")
    module.ARTIFACTS = artifacts
    module.RESULTS = results
    module._hydrate_partial_results()
    restored = results / "confirmation" / "seeds" / "seed-80721.json"
    assert restored.is_file()
    assert json.loads(restored.read_text(encoding="utf-8"))["complete"] is True


def test_host_level_failure_record_survives_sigkill_style_exit(tmp_path: Path) -> None:
    module = _load_script(
        "core007_orchestrator_failure",
        "scripts/research/orchestrate_core_validation_007_confirmation.py",
    )
    module.RESULTS = tmp_path / "results"
    log_path = module.RESULTS / "confirmation" / "logs" / "seed-80721.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("killed\n", encoding="utf-8")
    module._record_host_failure(80721, -9, log_path)
    failure = module.RESULTS / "confirmation" / "failures" / "seed-80721.json"
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert payload["returncode"] == -9
    assert payload["failure_kind"] == "child_process_terminated_without_python_failure_record"


def test_partial_confirmation_requires_explicit_allow_partial(tmp_path: Path) -> None:
    module = _load_script(
        "core007_publisher",
        "scripts/research/publish_core_validation_007.py",
    )
    results = tmp_path / "results"
    artifacts = tmp_path / "artifacts"
    phase = results / "confirmation"
    seeds = phase / "seeds"
    seeds.mkdir(parents=True)
    (seeds / "seed-80721.json").write_text('{"complete": true}\n', encoding="utf-8")
    (phase / "decision.json").write_text(
        json.dumps(
            {
                "status": "CONFIRMATION_INCOMPLETE",
                "scientific_decision": False,
                "completed_seeds": [80721],
            }
        ),
        encoding="utf-8",
    )
    module.RESULTS = results
    module.ARTIFACTS = artifacts
    with pytest.raises(RuntimeError):
        module._copy_phase("confirmation", allow_partial=False)
    dest, decision = module._copy_phase("confirmation", allow_partial=True)
    assert decision["scientific_decision"] is False
    assert (dest / "seeds" / "seed-80721.json").is_file()
    assert (artifacts / "confirmation-protocol-v1.1.json").is_file()


def test_publisher_reuses_historical_github_token_secret_name() -> None:
    module = _load_script(
        "core007_publisher_secret",
        "scripts/research/publish_core_validation_007.py",
    )
    assert module.DEFAULT_SECRET_NAME == "GITHUB_TOKEN"


def test_retired_monolithic_confirmation_entrypoint_fails_before_gpu_work() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/research/run_core_validation_007.py"),
            "--phase",
            "confirmation",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "monolithic confirmation entrypoint is retired" in combined
    assert "orchestrate_core_validation_007_confirmation.py" in combined
