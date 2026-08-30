"""Low-cost execution smoke for Core Validation 003."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_core_validation_003_smoke(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [sys.executable, "scripts/research/run.py", "core-validation-003", "--smoke", "--device", "cpu", "--out", str(tmp_path)],
        cwd=ROOT,
        env=env,
        check=True,
    )
