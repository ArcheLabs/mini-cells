"""Thin compatibility entry; frozen historical sources remain in place."""
from pathlib import Path
import runpy
import sys
ROOT = next(_root for _root in Path(__file__).resolve().parents if (_root / ".git").exists() or (_root / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
if __name__ == "__main__":
    runpy.run_module('minicells.training.evaluate_legacy', run_name="__main__")
