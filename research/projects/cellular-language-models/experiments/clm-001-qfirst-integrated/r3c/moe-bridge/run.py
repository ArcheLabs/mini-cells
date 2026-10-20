"""Isolated R3c entry point; synthetic CPU engineering is the default."""

from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
runpy.run_path(
    str(ROOT / "research/projects/cellular-language-models/engineering/kaggle-continuation/run.py"),
    run_name="r3c_bootstrap",
)
sys.path[:0] = [str(BASE / "implementation"), str(BASE.parents[1] / "r3b/implementation")]
if __name__ == "__main__":
    from minicells_r3c_moe_bridge.cli import main

    main()
