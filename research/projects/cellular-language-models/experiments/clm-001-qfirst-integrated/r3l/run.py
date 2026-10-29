from pathlib import Path
import sys
import runpy

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE.parent / "r3k/run.py"), run_name="r3l_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    from minicells_r3l.runner import main

    main()
