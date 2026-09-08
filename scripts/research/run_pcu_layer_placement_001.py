#!/usr/bin/env python3
"""Run the engineering-only PCU layer placement diagnostic on two GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.pcu_kill_001.layer_placement import (
    BASELINE_ROOT,
    DEFAULT_OUTPUT,
    ENGINEERING_SEED,
    run_layer_placement_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--baseline", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()
    devices = tuple(part.strip() for part in str(args.devices).split(",") if part.strip())
    if len(devices) != 2:
        raise ValueError("--devices must contain exactly two comma-separated CUDA devices")
    decision = run_layer_placement_diagnostic(
        output=args.out,
        baseline_root=args.baseline,
        devices=(devices[0], devices[1]),
        seed=args.seed,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
