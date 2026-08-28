#!/usr/bin/env python3
"""Launch and monitor the paired nine-worker CLM-0.3 matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


ARMS = ("fixed4", "pressure_growth", "random_growth")


def jobs() -> list[tuple[int, str]]:
    return [(replicate, arm) for replicate in range(3) for arm in ARMS]


def command(args: argparse.Namespace, replicate: int, arm: str) -> list[str]:
    result = [sys.executable, str(Path(__file__).with_name("run_clm_progressive_growth_001_worker.py")),
              "--release-dir", str(args.release_dir), "--output-dir", str(args.output_root / f"r{replicate}-{arm}"),
              "--arm", arm, "--replicate", str(replicate), "--target-tokens", str(args.target_tokens)]
    if args.execute:
        result.append("--execute")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CLM-0.3 three-arm, three-replicate matrix")
    parser.add_argument("--output-root", type=Path, default=Path("results/clm-0.3-progressive-growth"))
    parser.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    parser.add_argument("--target-tokens", type=int, default=1_500_000)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--execute", action="store_true", help="start formal workers; default is plan-only")
    args = parser.parse_args()
    planned = [(item, command(args, *item)) for item in jobs()]
    for _, cmd in planned:
        print("PLAN", " ".join(cmd), flush=True)
    if not args.execute:
        print("PREFLIGHT ONLY: pass --execute to launch the formal matrix.", flush=True)
        return 0
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("formal CLM-0.3 execution requires at least one CUDA GPU")
    capacity = min(args.max_workers or gpu_count, gpu_count)
    pending = list(planned)
    active: list[tuple[tuple[int, str], int, subprocess.Popen[bytes]]] = []
    while pending or active:
        while pending and len(active) < capacity:
            item, cmd = pending.pop(0)
            used = {gpu for _, gpu, _ in active}
            gpu = next(index for index in range(capacity) if index not in used)
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            active.append((item, gpu, subprocess.Popen(cmd, env=env)))
        states = []
        for (replicate, arm), _, process in active:
            progress = args.output_root / f"r{replicate}-{arm}" / "progress.json"
            if progress.exists():
                record = json.loads(progress.read_text())
                states.append(f"r{replicate}/{arm}: {record.get('consumed_tokens', 0):,}/{record.get('target_tokens', 0):,}")
        if states:
            print("DASHBOARD | " + " | ".join(states), flush=True)
        time.sleep(2)
        remaining = []
        for item, gpu, process in active:
            code = process.poll()
            if code is None:
                remaining.append((item, gpu, process))
            elif code != 0:
                for _, _, other in remaining:
                    other.terminate()
                raise subprocess.CalledProcessError(code, process.args)
        active = remaining
    print("All nine formal workers completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
