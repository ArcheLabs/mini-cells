"""Bootstrap the existing R3m-E/R3h runtime without executing a historical experiment."""

from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE.parent / "r3m-e/run.py"), run_name="r3mf_dependencies")
sys.path.insert(0, str(BASE.parent / "r3m-e/implementation"))
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    from minicells_r3mf.runner import main

    main()
