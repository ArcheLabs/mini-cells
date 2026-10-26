"""Gate 0 CPU engineering entry. No implicit scientific release."""

import argparse
import json
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "implementation"))


def bootstrap():
    dependency = BASE.parent / "jct-001/run.py"
    runpy.run_path(str(dependency), run_name="clm002_dependencies")["bootstrap"]()
    sys.path.insert(0, str(BASE / "implementation"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "engineering",
            "stage-a-development",
            "stage-a-confirmation",
            "stage-b",
            "publish-existing",
            "inspect-parent",
            "analyze-existing",
        ),
        default="engineering",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--analysis-input", type=Path)
    args = parser.parse_args()
    from minicells_clm002.common import configuration

    c = configuration()
    if args.mode.startswith("stage-"):
        from minicells_clm002.gates import require_scientific

        require_scientific(c, args.mode)  # Stop before parent/data/model/GPU access.
    bootstrap()
    if args.mode == "publish-existing":
        from minicells_clm002.publication import publish_existing

        result = publish_existing(args.output)
    elif args.mode == "analyze-existing":
        from minicells_clm002.analysis import analyze_existing

        if args.analysis_input is None:
            parser.error("--analysis-input is required")
        result = analyze_existing(args.analysis_input, args.output)
    elif args.mode == "inspect-parent":
        from minicells_clm002.audit import inspect_parent

        if args.checkpoint is None:
            parser.error("--checkpoint is required; historical bytes are never substituted")
        result = inspect_parent(c, args.checkpoint, args.output)
    else:
        from minicells_clm002.engineering import run

        result = run(c, args.output, args.publish_github, parent_checkpoint=args.checkpoint)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
