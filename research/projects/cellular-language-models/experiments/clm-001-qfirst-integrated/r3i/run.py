from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE.parent / "r3h/run.py"), run_name="r3i_bootstrap")
runpy.run_path(str(BASE.parent / "r3f-v2/run.py"), run_name="r3i_proposal_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    raise SystemExit("R3I_IMPLEMENTATION_IN_PROGRESS_NO_TRAINING_ENTRY_RELEASED")
