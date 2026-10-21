"""Bootstrap the existing research runtime; no training logic in Notebook."""

from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
runpy.run_path(str(BASE.parent / "r3c/moe-bridge/run.py"), run_name="r3d_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    from minicells_r3d.runner import main

    main()
