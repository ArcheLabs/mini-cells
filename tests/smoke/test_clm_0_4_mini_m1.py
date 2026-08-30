import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from minicells.clm04mini.m1 import run_m1_infrastructure_smoke
from minicells.clm04mini.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT / "research" / "validations" / "clm-0.4-mini-language-validation" / "protocol.json"
)


@pytest.mark.smoke
def test_m1_infrastructure_smoke_keeps_scientific_seeds_closed(tmp_path):
    result = run_m1_infrastructure_smoke(
        protocol_path=PROTOCOL,
        out_dir=tmp_path,
        device="cpu",
        seed=90400,
    )
    assert result["status"] == "SMOKE_ONLY"
    assert result["scientific_decision"] is False
    assert result["development_seed_observed"] is False
    assert result["formal_seeds_observed"] is False
    assert 4_500_000 <= result["formal_model_parameter_count"] <= 5_500_000
    assert result["base_corpus_manifest"]["actual_tokens"] >= 6000
    assert len(result["registered_calibration_grid"]["direct"]) == 9
    assert len(result["registered_calibration_grid"]["growth"]) == 9
    assert len(result["transaction_projection_ids"]) == 5
    assert all(item["match"] for item in result["checkpoint_replay"].values())
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision == {
        "reason": (
            "M1 infrastructure smoke validates data/model/variant/checkpoint plumbing only; "
            "development and formal seeds remain unopened."
        ),
        "scientific_decision": False,
        "status": "SMOKE_ONLY",
    }
    for seed in (90401, 90411, 90412, 90413):
        with pytest.raises(ProtocolError):
            run_m1_infrastructure_smoke(
                protocol_path=PROTOCOL,
                out_dir=tmp_path / str(seed),
                device="cpu",
                seed=seed,
            )


@pytest.mark.smoke
def test_unified_m1_runner_and_report_emit_smoke_only(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run.py",
            "clm-0.4-mini-m1",
            "--smoke",
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
            "clm-0.4-mini-m1",
            "--results",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    assert (tmp_path / "RESULTS.md").exists()
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "SMOKE_ONLY"
    assert decision["scientific_decision"] is False
