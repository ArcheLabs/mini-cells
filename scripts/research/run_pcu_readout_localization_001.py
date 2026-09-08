#!/usr/bin/env python3
"""Run PCU-READOUT-LOCALIZATION-001 on the engineering seed only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.pcu_kill_001.readout_localization import (
    DEFAULT_OUTPUT,
    HYBRID_BASELINE_ROOT,
    OBJECTIVE_BASELINE_ROOT,
    run_readout_localization_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26090501)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hybrid-baseline", type=Path, default=HYBRID_BASELINE_ROOT)
    parser.add_argument("--objective-baseline", type=Path, default=OBJECTIVE_BASELINE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_readout_localization_diagnostic(
        output=args.out,
        hybrid_root=args.hybrid_baseline,
        objective_root=args.objective_baseline,
        device=args.device,
        seed=args.seed,
    )
    decision = json.loads((args.out / "DECISION.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "status": result["status"],
        "first_token_top1_accuracy": decision["first_token_top1_accuracy"],
        "later_token_top1_accuracy": decision["later_token_top1_accuracy"],
        "force1_suffix_exact_accuracy": decision["force1_suffix_exact_accuracy"],
        "force2_suffix_exact_accuracy": decision["force2_suffix_exact_accuracy"],
        "minimal_forced_tokens_reaching_floor": decision["minimal_forced_tokens_reaching_floor"],
        "hybrid_reproduction_exact": decision["hybrid_reproduction_exact"],
        "formal_execution_not_started": decision["formal_execution_not_started"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
