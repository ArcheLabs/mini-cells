"""FAST002 additive audit / bounded targeted repair. No historical result mutation."""

import argparse
import json
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def bootstrap():
    runpy.run_path(str(BASE.parent / "clm-002-fast-001/run.py"), run_name="fast002_dependencies")[
        "bootstrap"
    ]()
    sys.path.insert(0, str(BASE / "implementation"))


def main():
    bootstrap()
    from minicells_fast002.runner import main as execute

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run-repair", action="store_true")
    parser.add_argument(
        "--checkpoint-roots",
        type=Path,
        nargs="+",
        default=[Path("/kaggle/input"), Path("/kaggle/working")],
    )
    parser.add_argument(
        "--kaggle-version",
        type=int,
        help="Optional exact saved version; normally discovered automatically",
    )
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--publish-existing", action="store_true")
    result = execute(parser.parse_args())
    if result is not None:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
