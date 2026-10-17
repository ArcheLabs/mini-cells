"""Read-only analysis launcher; never imports a training CLI entry point."""

from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
runpy.run_path(
    str(ROOT / "research/projects/cellular-language-models/engineering/kaggle-continuation/run.py"),
    run_name="analysis_bootstrap",
)
sys.path.insert(0, str(BASE / "implementation"))
from clm001_r3b.analysis_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
