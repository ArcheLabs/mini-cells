from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE.parent / "r3f-connection-admission/run.py"), run_name="r3h_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    from minicells_r3h.runner import main

    main()
