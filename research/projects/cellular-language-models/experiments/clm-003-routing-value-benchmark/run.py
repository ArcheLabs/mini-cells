"""Phase 0 only. Bootstrap the existing native model and verified publication transport."""

import argparse
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def bootstrap():
    runpy.run_path(
        str(BASE.parent / "clm-002-fast-002-r1-stateful-calibration/run.py"),
        run_name="dependencies",
    )["bootstrap"]()
    sys.path.insert(0, str(BASE / "implementation"))


def main():
    bootstrap()
    from routing_value import engineering

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--publish-existing", action="store_true")
    parser.add_argument("--run-science", action="store_true")
    args = parser.parse_args()
    if args.run_science:
        raise RuntimeError("CLM003_PHASE0_ONLY_SCIENTIFIC_SPECIFICATIONS_UNRESOLVED")
    engineering(args)


if __name__ == "__main__":
    main()
