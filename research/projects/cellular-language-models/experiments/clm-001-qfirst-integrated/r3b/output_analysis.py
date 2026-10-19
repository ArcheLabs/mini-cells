"""New transport bootstrap; measurements import only the immutable ed2accb checkout."""

from pathlib import Path
import argparse
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
ENGINE_SHA = "ed2accbc90f541325ca2f6c53229e9c81475768d"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine-root", type=Path, required=True)
    args, remainder = p.parse_known_args()
    engine = args.engine_root.resolve()
    worker = BASE / "implementation/kaggle_posthoc/worker.py"
    subprocess.run(
        [
            sys.executable,
            str(worker),
            "--engine-root",
            str(engine),
            "--orchestrator-root",
            str(ROOT),
            *remainder,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
