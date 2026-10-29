import runpy
from pathlib import Path

BASE = (
    Path(__file__).resolve().parents[3]
    / "research/projects/cellular-language-models/experiments/clm-003-routing-value-benchmark"
)
runpy.run_path(str(BASE / "run.py"), run_name="screening_tests")["bootstrap"]()
