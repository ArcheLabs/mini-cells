"""Independent controlled-task study; the original CLM-002 registration is unchanged."""

import argparse
import json
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def bootstrap():
    runpy.run_path(
        str(BASE.parent / "clm-002-cell-differentiation-joint-value/run.py"),
        run_name="fast_dependencies",
    )["bootstrap"]()
    sys.path.insert(0, str(BASE / "implementation"))


def main():
    bootstrap()
    from minicells_fast001.runner import main as execute

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--run-science", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--phase", choices=("A", "B"))
    parser.add_argument("--publish-existing", action="store_true")
    result = execute(parser.parse_args())
    if result is not None:
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
