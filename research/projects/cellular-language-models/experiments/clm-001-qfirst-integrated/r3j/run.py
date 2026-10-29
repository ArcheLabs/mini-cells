from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE.parent / "r3i/run.py"), run_name="r3j_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    from minicells_r3j.runner import main

    main()
