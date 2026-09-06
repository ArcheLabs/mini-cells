#!/usr/bin/env python3
"""Run PCU-LOCALITY-WIDTH-001 engineering diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.pcu_kill_001.locality_width import (
    DEFAULT_OUTPUT,
    ENGINEERING_SEED,
    LAYER_BASELINE_ROOT,
)
from minicells.pcu_kill_001.locality_width_isolated import (
    run_locality_width_diagnostic_isolated,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=LAYER_BASELINE_ROOT)
    parser.add_argument("--device0", default="cuda:0")
    parser.add_argument("--device1", default="cuda:1")
    args = parser.parse_args()
    decision = run_locality_width_diagnostic_isolated(
        output=args.out,
        baseline_root=args.baseline,
        devices=(args.device0, args.device1),
        seed=args.seed,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
