#!/usr/bin/env python3
"""Run the low-cost Experiment 026a granularity screening protocol."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORMAL_RUNNER = ROOT / "scripts" / "run_experiment_026_cell_granularity.py"
SCREENING_REPORT = ROOT / "scripts" / "report_experiment_026a_granularity_screening.py"
SCREENING_PROTOCOL = ROOT / "research" / "stages" / "03-routing-and-growth" / "sources" / "experiment-026a-protocol.json"
SCREENING_OUT = ROOT / "results" / "experiment-026a-granularity-screening"


def _load_formal_runner():
    spec = importlib.util.spec_from_file_location("experiment_026_formal_runner", FORMAL_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FORMAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_formal_runner()
    module.GRANULARITIES = (1, 4, 8)
    module.OUT = SCREENING_OUT
    module.REPORT = SCREENING_REPORT
    module.FROZEN_PROTOCOL = SCREENING_PROTOCOL
    module.DEFAULT_TOTAL_WALL_HOURS = 4.0
    module.DEFAULT_FINALIZATION_RESERVE_MINUTES = 15.0
    module.DEFAULT_WORKER_SLICE_HOURS = 1.5

    # Reuse the tested Experiment-026 data preparation, worker, checkpoint/resume,
    # and dual-GPU orchestration. Only the screening budget and arm set change.
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--continuation-tokens",
                "5000000",
                "--total-wall-hours",
                "4",
                "--finalization-reserve-minutes",
                "15",
                "--worker-slice-hours",
                "1.5",
            ]
        )
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
