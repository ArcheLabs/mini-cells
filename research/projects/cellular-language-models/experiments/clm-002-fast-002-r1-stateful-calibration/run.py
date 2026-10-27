"""Engineering and read-only Gate 0 only; no implicit diagnostic/scientific execution."""

import argparse
import json
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def bootstrap():
    runpy.run_path(str(BASE.parent / "clm-002-fast-002/run.py"), run_name="r1_dependencies")[
        "bootstrap"
    ]()
    sys.path.insert(0, str(BASE / "implementation"))
    sys.path.insert(0, str(BASE.parent / "clm-001-qfirst-integrated/r3f-v2/implementation"))


def main():
    bootstrap()
    from minicells_fast_r1.runner import engineering

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--inspect-cloud-parents", action="store_true")
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--publish-existing", action="store_true")
    parser.add_argument("--run-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.run_diagnostic:
        raise RuntimeError("R1_GATE1_REQUIRES_INDEPENDENT_REVIEW_AND_EXPLICIT_EXECUTION_APPROVAL")
    print(json.dumps(engineering(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
