import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from minicells.clm04mini import M0_SEED, run_m0


ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_REQUIRED = {
    "transaction_id",
    "operation",
    "address_id",
    "train_manifest_ids",
    "validation_manifest_ids",
    "probe_manifest_ids",
    "state_hash_before",
    "base_route_cells_by_layer",
    "attempts",
    "final_decision",
    "growth_attempted",
    "cell_births",
    "cell_deletions",
    "state_hash_after",
    "active_cells_by_layer",
    "transaction_wall_seconds",
}

ATTEMPT_REQUIRED = {
    "attempt_index",
    "candidate_kind",
    "touched_cells",
    "touched_parameter_count",
    "touched_parameter_fraction",
    "local_dependency_probe_count",
    "local_dependency_coverage",
    "new_gain",
    "local_regression",
    "global_regression",
    "local_pass",
    "oracle_pass",
    "false_safe",
    "structural_escape_count",
    "structural_escape_rate",
    "training_tokens",
    "validation_tokens",
    "optimizer_steps",
    "candidate_wall_seconds",
    "validation_wall_seconds",
    "peak_gpu_memory_bytes",
}

CELL_REQUIRED = {
    "cell_id",
    "layer_id",
    "cell_type",
    "parameter_count",
    "state_hash",
    "activation_count",
    "dependency_probe_count",
    "accepted_updates",
    "rejected_updates",
}


@pytest.mark.smoke
def test_clm_0_4_mini_m0_execution_smoke(tmp_path):
    summary = run_m0(tmp_path, device="cpu", seed=M0_SEED)
    assert summary["decision"]["status"] == "SMOKE_ONLY"
    assert summary["decision"]["scientific_decision"] is False
    assert summary["decision"]["paths_exercised"] == [
        "growth-commit",
        "private-reuse-commit",
        "direct-commit",
    ]
    assert summary["replay_valid"] is True

    records = [
        json.loads(line)
        for line in (tmp_path / "transactions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    for record in records:
        assert TRANSACTION_REQUIRED <= set(record)
        assert record["smoke_only"] is True
        for attempt in record["attempts"]:
            assert ATTEMPT_REQUIRED <= set(attempt)
            assert attempt["structural_escape_count"] == 0

    assert records[0]["attempts"][0]["candidate_kind"] == "direct"
    assert records[0]["attempts"][1]["candidate_kind"] == "spawn"
    assert records[0]["attempts"][1]["zero_output_pretrain_max_logit_delta"] <= 1e-6
    assert records[0]["state_hash_after"] == records[1]["state_hash_before"]
    assert records[1]["state_hash_after"] == records[2]["state_hash_before"]

    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "SMOKE_ONLY"

    registry = json.loads((tmp_path / "cell-registry.json").read_text(encoding="utf-8"))
    for entry in registry:
        assert CELL_REQUIRED <= set(entry)
    private = [entry for entry in registry if entry["cell_type"] == "private-growth"]
    assert len(private) == 2
    assert {entry["owner_address_id"] for entry in private} == {"math/mul"}


@pytest.mark.smoke
def test_m0_rejects_development_and_formal_seeds(tmp_path):
    for seed in (90401, 90411, 90412, 90413):
        with pytest.raises(ValueError):
            run_m0(tmp_path / str(seed), device="cpu", seed=seed)


@pytest.mark.smoke
def test_m0_unified_runner_and_report(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run.py",
            "clm-0.4-mini-m0",
            "--device",
            "cpu",
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/research/report.py",
            "clm-0.4-mini-m0",
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "SMOKE_ONLY"
    assert decision["scientific_decision"] is False
    assert (tmp_path / "RESULTS.md").exists()
