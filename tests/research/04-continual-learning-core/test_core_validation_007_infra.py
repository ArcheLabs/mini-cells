from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"


def _load_script(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_amendment_retires_exposed_seed_set_without_gate_changes() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    retired = set(amendment["retired_confirmation_seeds"])
    fresh = set(amendment["confirmation_seeds"])
    assert retired == {80711, 80712, 80713}
    assert fresh == {80721, 80722, 80723}
    assert retired.isdisjoint(fresh)
    invariants = amendment["scientific_invariants"]
    assert invariants == {
        "winner_changed": False,
        "boundary_mechanism_changed": False,
        "model_or_data_changed": False,
        "gate_thresholds_changed": False,
        "core006_baselines_changed": False,
    }
    assert amendment["winner"] == "interference_cut"


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


def test_partial_confirmation_can_be_copied_to_canonical_artifacts(tmp_path: Path) -> None:
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
    dest, decision = module._copy_phase("confirmation")
    assert decision["scientific_decision"] is False
    assert (dest / "seeds" / "seed-80721.json").is_file()
    assert (artifacts / "confirmation-protocol-v1.1.json").is_file()
