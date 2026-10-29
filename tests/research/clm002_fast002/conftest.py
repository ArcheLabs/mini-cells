import runpy
from pathlib import Path

BASE = (
    Path(__file__).resolve().parents[3]
    / "research/projects/cellular-language-models/experiments/clm-002-fast-002"
)
runpy.run_path(str(BASE / "run.py"), run_name="fast002_tests")["bootstrap"]()
