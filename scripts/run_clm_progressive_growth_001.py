#!/usr/bin/env python3
"""Parent runner for CLM-0.3; prints the planned matrix before execution."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (55031, 55032, 55033)
ARMS = ("fixed4", "pressure_growth", "random_growth")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CLM-0.3 three-arm matrix")
    parser.add_argument("--output-root", type=Path, default=Path("results/clm-0.3-progressive-growth"))
    parser.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = [(replicate, arm) for replicate in range(3) for arm in ARMS]
    for replicate, arm in jobs:
        command = [
            sys.executable, str(Path(__file__).with_name("run_clm_progressive_growth_001_worker.py")),
            "--release-dir", str(args.release_dir), "--output-dir", str(args.output_root / f"r{replicate}-{arm}"),
            "--arm", arm, "--replicate", str(replicate),
        ]
        print("PLAN", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
