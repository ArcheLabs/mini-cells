from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE.parent / "r3h/run.py"), run_name="r3i_bootstrap")
runpy.run_path(str(BASE.parent / "r3f-v2/run.py"), run_name="r3i_proposal_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    from minicells_r3i.runner import main

    main()
