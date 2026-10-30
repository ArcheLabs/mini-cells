"""EPCLM-001 independent inference-only entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "implementation"))
from epclm001.cli import main

if __name__ == "__main__":
    main()
