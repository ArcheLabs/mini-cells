from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3i/run.py"
        ),
        run_name="__main__",
    )
