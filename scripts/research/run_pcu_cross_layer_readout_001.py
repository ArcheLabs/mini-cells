#!/usr/bin/env python3
"""Run PCU-CROSS-LAYER-READOUT-001 engineering diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicells.pcu_kill_001.cross_layer_readout import (  # noqa: E402
    DEFAULT_OUTPUT,
    HYBRID_BASELINE_ROOT,
    OBJECTIVE_BASELINE_ROOT,
    READOUT_BASELINE_ROOT,
    run_cross_layer_readout_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26090501)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objective-root", type=Path, default=OBJECTIVE_BASELINE_ROOT)
    parser.add_argument("--hybrid-root", type=Path, default=HYBRID_BASELINE_ROOT)
    parser.add_argument("--readout-root", type=Path, default=READOUT_BASELINE_ROOT)
    args = parser.parse_args()
    result = run_cross_layer_readout_diagnostic(
        output=args.output,
        objective_root=args.objective_root,
        hybrid_root=args.hybrid_root,
        readout_root=args.readout_root,
        device=args.device,
        seed=args.seed,
    )
    comparison = result["comparison"]
    print(
        f"PCU-CROSS-LAYER-READOUT-001 status={result['status']} "
        f"L7={comparison['l7_only_direct']:.6f} "
        f"L23={comparison['l23_only_direct']:.6f} "
        f"L7+L23={comparison['cross_layer_direct']:.6f} "
        f"ranking={comparison['cross_layer_ranking']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
