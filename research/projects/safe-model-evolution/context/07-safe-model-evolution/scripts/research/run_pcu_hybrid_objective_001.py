#!/usr/bin/env python3
"""Run the narrow PCU-HYBRID-OBJECTIVE-001 engineering diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.pcu_kill_001.hybrid_objective import (
    DEFAULT_OUTPUT,
    ENGINEERING_SEED,
    OBJECTIVE_BASELINE_ROOT,
    run_hybrid_objective_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", type=Path, default=OBJECTIVE_BASELINE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_hybrid_objective_diagnostic(
        output=args.out,
        baseline_root=args.baseline,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps({
        "experiment": result["experiment"],
        "status": result["status"],
        "ce_weight": result["ce_weight"],
        "ranking_eval_accuracy": result["ranking"]["eval"]["accuracy"],
        "direct_accuracy": result["direct_accuracy"],
        "formal_execution_not_started": result["formal_execution_not_started"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
