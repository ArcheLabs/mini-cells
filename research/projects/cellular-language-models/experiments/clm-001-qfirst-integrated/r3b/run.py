"""Reuse continuation import bootstrap, then select the distinct research launcher."""

from pathlib import Path
import runpy
import sys
import os

# Hide host CUDA before importing the historical bootstrap (which imports torch).
# Fresh spawned workers restore their one assigned physical device themselves.
if __name__ == "__main__":
    os.environ.setdefault("MINICELLS_WORKER_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
runpy.run_path(
    str(ROOT / "research/projects/cellular-language-models/engineering/kaggle-continuation/run.py"),
    run_name="r3b_bootstrap",
)
sys.path.insert(0, str(BASE / "implementation"))
from clm001_r3b.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
