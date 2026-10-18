"""Bounded one-update engineering verification; no formal experiment entry."""

from pathlib import Path
import runpy
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
runpy.run_path(
    str(ROOT / "research/projects/cellular-language-models/engineering/kaggle-continuation/run.py"),
    run_name="analysis_bootstrap",
)
sys.path.insert(0, str(BASE / "implementation"))
from clm001_r3b.engineering_delta import main  # noqa: E402

if __name__ == "__main__":
    main()
