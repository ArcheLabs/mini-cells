#!/usr/bin/env python3
"""Run exactly one PCU-LOCALITY-WIDTH-001 width in an isolated CUDA process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.pcu_kill_001.locality_width import DEFAULT_OUTPUT, ENGINEERING_SEED, LAYER_BASELINE_ROOT
from minicells.pcu_kill_001.locality_width_isolated import run_width_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=LAYER_BASELINE_ROOT)
    args = parser.parse_args()
    result = run_width_worker(
        width=args.width,
        device=args.device,
        output=args.out,
        baseline_root=args.baseline,
        seed=args.seed,
    )
    print(json.dumps({
        "width": args.width,
        "device": args.device,
        "direct_accuracy": result["direct_accuracy"],
        "passes": result["passes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
