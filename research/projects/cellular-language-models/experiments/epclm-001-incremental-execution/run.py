"""EPCLM-001 independent inference-only entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "implementation"))
from epclm001.cli import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("retrain-"):
        from epclm001.retrain import main as retrain_main

        retrain_main()
    else:
        main()
