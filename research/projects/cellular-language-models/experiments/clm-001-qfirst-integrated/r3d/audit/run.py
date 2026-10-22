"""Read-only audit entry; bootstrap model/data primitives without a Trainer."""

from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE.parent / "run.py"), run_name="r3d_audit_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
if __name__ == "__main__":
    from minicells_r3d_audit.runner import main

    main()
