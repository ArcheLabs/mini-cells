from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(
    str(
        ROOT
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3g/run.py"
    ),
    run_name="__main__",
)
