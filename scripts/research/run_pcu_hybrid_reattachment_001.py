#!/usr/bin/env python3
"""Run PCU-HYBRID-REATTACHMENT-001 on the pinned engineering mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicells.pcu_kill_001.hybrid_reattachment import (  # noqa: E402
    DEFAULT_OUTPUT,
    run_hybrid_reattachment_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the published L7/K64 PCU mutation and compare final Granite "
            "outputs with the exact same Cell deltas ON, OFF and RESTORED."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-direct-generation",
        action="store_true",
        help="Skip the secondary greedy-generation diagnostic; causal ranking/logit gates still run.",
    )
    args = parser.parse_args()
    result = run_hybrid_reattachment_diagnostic(
        output=args.output,
        device=args.device,
        run_direct_generation=not args.skip_direct_generation,
    )
    print(json.dumps({
        "experiment": result["experiment"],
        "status": result["status"],
        "valid_run": result["valid_run"],
        "scientific_evidence": result["scientific_evidence"],
        "formal_execution_not_started": result["formal_execution_not_started"],
        "ranking_on": result["causal_effect"]["ranking_on"],
        "ranking_off": result["causal_effect"]["ranking_off"],
        "ranking_gain": result["causal_effect"]["ranking_gain"],
        "answer_margin_gain": result["causal_effect"]["answer_margin_gain"],
        "B_control_answer_nll_increase": result["causal_effect"]["B_control_answer_nll_increase"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
