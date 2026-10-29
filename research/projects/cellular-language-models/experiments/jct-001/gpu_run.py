"""Explicitly authorized Stage 1 entry; one automatic preflight → Pilot → return."""

import argparse
import json
from pathlib import Path
from run import bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("run", "publish-existing"), default="run")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bootstrap()
    if args.mode == "publish-existing":
        from minicells_jct001.gpu_publication import publish_existing

        result = publish_existing(args.output)
    else:
        from minicells_jct001.gpu_runner import pilot_configuration, run

        p, c = pilot_configuration()
        result = run(p, c, args.output, args.device, publish_github=True)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
