from pathlib import Path
import runpy

runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m/run.py"
    ),
    run_name="__main__",
)
