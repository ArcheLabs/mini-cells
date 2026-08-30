import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from minicells.clm04mini.calibration import write_plan_only


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "clm-0.4-mini-language-validation"
PROTOCOL = VALIDATION / "protocol.json"
PLAN = VALIDATION / "calibration-plan.json"
ASSETS = VALIDATION / "calibration-assets.json"


@pytest.mark.smoke
def test_calibration_plan_only_keeps_all_scientific_seeds_closed(tmp_path):
    decision = write_plan_only(
        protocol_path=PROTOCOL,
        committed_plan_path=PLAN,
        expected_assets_path=ASSETS,
        out_dir=tmp_path,
    )
    assert decision["status"] == "CALIBRATION_PLAN_ONLY"
    assert decision["scientific_decision"] is False
    assert decision["development_seed_observed"] is False
    assert decision["formal_seeds_observed"] is False
    assert decision["candidate_count"] == 81


@pytest.mark.smoke
def test_unified_calibration_plan_runner_and_report_do_not_open_90401(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run.py",
            "clm-0.4-mini-calibration",
            "--plan-only",
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
            "clm-0.4-mini-calibration",
            "--results",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "CALIBRATION_PLAN_ONLY"
    assert decision["development_seed_observed"] is False
    assert decision["formal_seeds_observed"] is False
    assert (tmp_path / "RESULTS.md").is_file()
