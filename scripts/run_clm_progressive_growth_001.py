#!/usr/bin/env python3
"""Launch, monitor, and aggregate the paired nine-worker CLM-0.3 matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from minicells.growth_reporting import (  # noqa: E402
    FORMAL_TARGET_TOKENS,
    aggregate_formal_results,
)


ARMS = ("fixed4", "pressure_growth", "random_growth")


def jobs() -> list[tuple[int, str]]:
    return [(replicate, arm) for replicate in range(3) for arm in ARMS]


def command(args: argparse.Namespace, replicate: int, arm: str) -> list[str]:
    result = [
        sys.executable,
        str(Path(__file__).with_name("run_clm_progressive_growth_001_worker.py")),
        "--release-dir", str(args.release_dir),
        "--output-dir", str(args.output_root / f"r{replicate}-{arm}"),
        "--arm", arm,
        "--replicate", str(replicate),
        "--target-tokens", str(args.target_tokens),
    ]
    if args.execute:
        result.append("--execute")
    return result


def _dashboard_line(
    replicate: int,
    arm: str,
    gpu: int,
    record: dict[str, object],
) -> str:
    consumed = int(record.get("consumed_tokens", 0) or 0)
    target = int(record.get("target_tokens", 0) or 0)
    phase = str(record.get("phase", "starting"))
    ppl = record.get("ppl")
    throughput = float(record.get("tokens_per_second", 0.0) or 0.0)
    ppl_text = "--" if ppl is None else f"{float(ppl):.4f}"
    return (
        f"r{replicate} {arm:<15} | GPU{gpu} | {consumed/1e6:.2f}/{target/1e6:.2f}M "
        f"| {phase:<12} | PPL {ppl_text} | {throughput/1000:.1f}K tok/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CLM-0.3 three-arm, three-replicate matrix")
    parser.add_argument("--output-root", type=Path, default=Path("results/clm-0.3-progressive-growth"))
    parser.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    parser.add_argument("--target-tokens", type=int, default=FORMAL_TARGET_TOKENS)
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
    completed_count = 0

    while pending or active:
        while pending and len(active) < capacity:
            item, cmd = pending.pop(0)
            used = {gpu for _, gpu, _ in active}
            gpu = next(index for index in range(capacity) if index not in used)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            # Workers are standalone Python processes. Notebook-level sys.path
            # changes do not propagate across exec(), so explicitly expose the
            # repository's research package root to every worker.
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                str(RESEARCH_ROOT)
                if not existing_pythonpath
                else str(RESEARCH_ROOT) + os.pathsep + existing_pythonpath
            )
            active.append((item, gpu, subprocess.Popen(cmd, env=env, cwd=REPO_ROOT)))

        states = []
        for (replicate, arm), gpu, _process in active:
            progress = args.output_root / f"r{replicate}-{arm}" / "progress.json"
            if progress.exists():
                try:
                    record = json.loads(progress.read_text(encoding="utf-8"))
                    states.append(_dashboard_line(replicate, arm, gpu, record))
                except (OSError, json.JSONDecodeError, ValueError):
                    states.append(f"r{replicate} {arm:<15} | GPU{gpu} | progress updating")
            else:
                states.append(f"r{replicate} {arm:<15} | GPU{gpu} | starting")
        if states:
            print(
                "DASHBOARD | " + " || ".join(states)
                + f" | completed {completed_count}/9 | pending {len(pending)}",
                flush=True,
            )

        time.sleep(2)
        remaining: list[tuple[tuple[int, str], int, subprocess.Popen[bytes]]] = []
        failed: tuple[subprocess.Popen[bytes], int] | None = None
        for item, gpu, process in active:
            code = process.poll()
            if code is None:
                remaining.append((item, gpu, process))
            elif code != 0:
                failed = (process, code)
                break
            else:
                completed_count += 1
        if failed is not None:
            for _, _, process in active:
                if process.poll() is None:
                    process.terminate()
            process, code = failed
            raise subprocess.CalledProcessError(code, process.args)
        active = remaining

    formal_run = args.target_tokens == FORMAL_TARGET_TOKENS
    aggregate = aggregate_formal_results(
        args.output_root,
        formal_gpu_experiment_run=formal_run,
        expected_target_tokens=args.target_tokens,
    )
    decision = aggregate["decision"]
    print("All nine workers completed and matched results were aggregated.", flush=True)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print(f"decision: {args.output_root / 'decision.json'}", flush=True)
    print(f"formal history: {args.output_root / 'formal-ppl-history.csv'}", flush=True)
    if not formal_run:
        print("NOTE: shortened matrix; formal_gpu_experiment_run remains false.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
