"""JCT-001 independent engineering entry; no scientific stage is released."""

from pathlib import Path
import argparse
import json
import runpy
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "implementation"))


def bootstrap():
    parent = BASE.parent / "clm-001-qfirst-integrated/r3m-f-d1/run.py"
    namespace = runpy.run_path(str(parent), run_name="jct001_reused_dependencies")
    namespace["_bootstrap"]()
    sys.path.insert(0, str(BASE / "implementation"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "engineering",
            "synthetic-pilot",
            "real-model",
            "confirmation",
            "publish-existing",
        ),
        default="engineering",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-github", action="store_true")
    args = parser.parse_args()
    from minicells_jct001.common import configuration

    c = configuration()
    if args.mode not in ("engineering", "publish-existing"):
        from minicells_jct001.gates import require_scientific

        require_scientific(args.mode, c)  # Stop before Torch, data loaders or checkpoints.
    bootstrap()
    if args.mode == "publish-existing":
        from minicells_jct001.publication import publish_existing

        result = publish_existing(args.output)
    else:
        from minicells_jct001.runner import engineering

        result = engineering(c, args.output, publish_github=args.publish_github)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
