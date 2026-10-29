"""R3m-F entrypoint, using the experiment's original bootstrap."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f/run.py"
        ),
        run_name="__main__",
    )
