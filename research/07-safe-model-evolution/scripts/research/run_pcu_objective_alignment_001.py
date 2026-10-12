#!/usr/bin/env python3
"""Run PCU-OBJECTIVE-ALIGNMENT-001 engineering diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.pcu_kill_001.objective_alignment import (
    DEFAULT_OUTPUT,
    ENGINEERING_SEED,
    LOCALITY_BASELINE_ROOT,
    run_objective_alignment_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=LOCALITY_BASELINE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_objective_alignment_diagnostic(
        output=args.out,
        baseline_root=args.baseline,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps({
        "status": result["status"],
        "ranking_train_accuracy": result["ranking"]["train"]["accuracy"],
        "ranking_eval_accuracy": result["ranking"]["eval"]["accuracy"],
        "direct_accuracy": result["direct_accuracy"],
        "formal_execution_not_started": result["formal_execution_not_started"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
