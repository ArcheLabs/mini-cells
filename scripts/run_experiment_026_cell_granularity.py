#!/usr/bin/env python3
"""Orchestrate formal Experiment 026 on one or two CUDA GPUs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from minicells.granularity_30m import (  # noqa: E402
    CONTINUATION_TOKENS,
    GRANULARITIES,
    RESULT_DIR_NAME,
    prepare_domain_corpora,
    protocol_manifest,
    schedule_manifest,
)
from minicells.language_30m import prepare_30m_corpus  # noqa: E402
from minicells.language_data import load_tokenizer  # noqa: E402


SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
OUT = ROOT / "results" / RESULT_DIR_NAME
WORKER = ROOT / "scripts" / "run_experiment_026_cell_granularity_worker.py"
REPORT = ROOT / "scripts" / "report_experiment_026_cell_granularity.py"
FROZEN_PROTOCOL = ROOT / "research" / "experiment-026-protocol.json"

DEFAULT_TOTAL_WALL_HOURS = 8.0
DEFAULT_FINALIZATION_RESERVE_MINUTES = 30.0
DEFAULT_WORKER_SLICE_HOURS = 2.25
MIN_LAUNCH_WINDOW_MINUTES = 20.0


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="Run Experiment 026 — cell granularity")
    result.add_argument("--continuation-tokens", type=int, default=CONTINUATION_TOKENS)
    result.add_argument("--total-wall-hours", type=float, default=DEFAULT_TOTAL_WALL_HOURS)
    result.add_argument(
        "--finalization-reserve-minutes",
        type=float,
        default=DEFAULT_FINALIZATION_RESERVE_MINUTES,
    )
    result.add_argument("--worker-slice-hours", type=float, default=DEFAULT_WORKER_SLICE_HOURS)
    result.add_argument("--reset", action="store_true")
    return result.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(granularity: int) -> dict[str, object] | None:
    path = OUT / f"g{granularity}" / "worker-summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(granularity: int) -> bool:
    summary = _summary(granularity)
    return bool(summary and summary.get("complete"))


def _worker_command(
    granularity: int,
    *,
    corpus,
    synthetic,
    continuation_tokens: int,
    max_wall_hours: float,
) -> list[str]:
    math = synthetic["math"]
    symbolic = synthetic["symbolic"]
    facts = synthetic["facts"]
    return [
        sys.executable,
        str(WORKER),
        "--granularity",
        str(granularity),
        "--tokenizer-path",
        str(corpus.tokenizer_path),
        "--story-train",
        str(corpus.train_path),
        "--story-validation",
        str(corpus.validation_path),
        "--math-train",
        str(math["train_path"]),
        "--math-validation",
        str(math["validation_path"]),
        "--symbolic-train",
        str(symbolic.train_path),
        "--symbolic-validation",
        str(symbolic.validation_path),
        "--facts-train",
        str(facts.train_path),
        "--facts-validation",
        str(facts.validation_path),
        "--output-dir",
        str(OUT),
        "--continuation-tokens",
        str(continuation_tokens),
        "--max-wall-hours",
        str(max_wall_hours),
    ]


def _run_batch(
    granularities: list[int],
    *,
    corpus,
    synthetic,
    continuation_tokens: int,
    worker_slice_hours: float,
    batch_index: int,
) -> None:
    active: list[tuple[int, int, subprocess.Popen[str], Path]] = []
    for gpu_index, granularity in enumerate(granularities):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        log_path = OUT / f"g{granularity}.batch-{batch_index}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            _worker_command(
                granularity,
                corpus=corpus,
                synthetic=synthetic,
                continuation_tokens=continuation_tokens,
                max_wall_hours=worker_slice_hours,
            ),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.close()
        active.append((granularity, gpu_index, process, log_path))
        print(f"started G={granularity} on physical GPU {gpu_index} / batch {batch_index}")

    failures: list[str] = []
    for granularity, gpu_index, process, log_path in active:
        code = process.wait()
        print(f"--- G={granularity} / GPU {gpu_index} / batch {batch_index} ---")
        print(log_path.read_text(encoding="utf-8").rstrip())
        if code != 0:
            failures.append(f"G={granularity} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> int:
    args = parser()
    if args.continuation_tokens <= 0 or args.continuation_tokens > CONTINUATION_TOKENS:
        raise ValueError("--continuation-tokens must be in (0, 20M]")
    if args.total_wall_hours <= 0 or args.total_wall_hours > 8.0:
        raise ValueError("--total-wall-hours must be in (0, 8]")
    if args.finalization_reserve_minutes < 10:
        raise ValueError("reserve at least 10 minutes for finalization")
    if args.worker_slice_hours <= 0:
        raise ValueError("--worker-slice-hours must be positive")

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Experiment 026 requires a clean tracked Git tree")

    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 026 requires at least one CUDA GPU")
    used = min(2, available)

    job_started = time.monotonic()
    job_deadline = job_started + args.total_wall_hours * 3600.0
    training_deadline = job_deadline - args.finalization_reserve_minutes * 60.0
    if training_deadline <= job_started:
        raise ValueError("finalization reserve consumes the whole job budget")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.reset:
        for granularity in GRANULARITIES:
            arm_dir = OUT / f"g{granularity}"
            if arm_dir.exists():
                shutil.rmtree(arm_dir)
        for path in OUT.glob("g*.batch-*.log"):
            path.unlink()

    if not FROZEN_PROTOCOL.is_file():
        raise FileNotFoundError(FROZEN_PROTOCOL)
    shutil.copy2(FROZEN_PROTOCOL, OUT / "protocol.json")

    print("preparing/reusing Experiment-007 TinyStories corpus")
    corpus = prepare_30m_corpus(ROOT, source_006_dir=SOURCE_006)
    tokenizer = load_tokenizer(corpus.tokenizer_path)
    print("preparing/reusing Experiment-026 heterogeneous synthetic domains")
    synthetic = prepare_domain_corpora(OUT / "cache", tokenizer)

    provenance = {
        "format": "minicells.cell-granularity-30m-run.v1",
        "code_commit": _git("rev-parse", "HEAD"),
        "code_tree_sha": _git("rev-parse", "HEAD^{tree}"),
        "tracked_tree_dirty": False,
        "gpu_count_visible": available,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(available)],
        "gpu_count_used": used,
        "continuation_tokens": args.continuation_tokens,
        "schedule": schedule_manifest(args.continuation_tokens),
        "protocol_runtime_manifest": protocol_manifest(),
        "story_corpus_manifest": corpus.manifest,
        "synthetic_manifests": {
            "math": synthetic["math"]["manifest"],
            "symbolic": synthetic["symbolic"].manifest,
            "facts": synthetic["facts"].manifest,
        },
        "budget": {
            "global_wall_hours": args.total_wall_hours,
            "finalization_reserve_minutes": args.finalization_reserve_minutes,
            "worker_slice_hours": args.worker_slice_hours,
            "minimum_launch_window_minutes": MIN_LAUNCH_WINDOW_MINUTES,
            "automatic_resume": True,
            "queue_policy": "round-robin across incomplete granularity arms",
        },
    }
    _json_write(OUT / "run-provenance.json", provenance)

    queue: deque[int] = deque(granularity for granularity in GRANULARITIES if not _complete(granularity))
    batch_index = 0
    while queue:
        remaining = training_deadline - time.monotonic()
        if remaining < MIN_LAUNCH_WINDOW_MINUTES * 60.0:
            print("stopping automatic resume: remaining training budget is below launch window")
            break
        selected: list[int] = []
        while queue and len(selected) < used:
            granularity = queue.popleft()
            if not _complete(granularity):
                selected.append(granularity)
        if not selected:
            break
        batch_index += 1
        worker_slice_hours = min(args.worker_slice_hours, remaining / 3600.0)
        _run_batch(
            selected,
            corpus=corpus,
            synthetic=synthetic,
            continuation_tokens=args.continuation_tokens,
            worker_slice_hours=worker_slice_hours,
            batch_index=batch_index,
        )
        for granularity in selected:
            if not _complete(granularity):
                queue.append(granularity)
        for granularity in GRANULARITIES:
            if not _complete(granularity) and granularity not in queue and granularity not in selected:
                queue.append(granularity)

    summaries = {
        f"g{granularity}": _summary(granularity)
        or {
            "granularity": granularity,
            "complete": False,
            "reason": "worker did not emit a summary before global deadline",
        }
        for granularity in GRANULARITIES
    }
    complete = all(bool(summary.get("complete")) for summary in summaries.values())
    _json_write(
        OUT / "worker-summary.json",
        {
            "format": "minicells.cell-granularity-30m-workers.v1",
            "complete": complete,
            "automatic_batches": batch_index,
            "global_elapsed_seconds": time.monotonic() - job_started,
            "arms": summaries,
        },
    )

    if complete:
        subprocess.run(
            [sys.executable, str(REPORT), "--results-dir", str(OUT)],
            cwd=ROOT,
            check=True,
        )
        print(f"Experiment 026 complete: {OUT}")
    else:
        print("Experiment 026 incomplete inside the global budget; checkpoints and partial tables were preserved.")
    print(f"job elapsed={(time.monotonic() - job_started)/3600.0:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
