"""Thin compatibility entry; frozen historical sources remain in place."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
if __name__ == "__main__":
    runpy.run_module('minicells.training.evaluate_legacy', run_name="__main__")
