#!/usr/bin/env python3
"""Launch, monitor, resume, and aggregate the paired nine-worker CLM-0.3 matrix."""

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

from minicells.growth_checkpoint import GROWTH_CHECKPOINT_FORMAT  # noqa: E402
from minicells.growth_reporting import (  # noqa: E402
    FORMAL_TARGET_TOKENS,
    aggregate_formal_results,
)
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402


ARMS = ("fixed4", "pressure_growth", "random_growth")
WORKER_BATCH_SIZE = 8
WORKER_SEQUENCE_LENGTH = 125
WORKER_EVAL_BATCHES = 4


def jobs() -> list[tuple[int, str]]:
    return [(replicate, arm) for replicate in range(3) for arm in ARMS]


def _worker_dir(args: argparse.Namespace, replicate: int, arm: str) -> Path:
    return args.output_root / f"r{replicate}-{arm}"


def _worker_complete(directory: Path, target_tokens: int) -> bool:
    path = directory / "events.jsonl"
    if not path.exists():
        return False
    complete = False
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    event.get("type") == "worker_complete"
                    and int(event.get("consumed_tokens", -1)) >= target_tokens
                    and int(event.get("target_tokens", -1)) == target_tokens
                    and event.get("mode") != "preflight_only"
                ):
                    complete = True
    except OSError:
        return False
    return complete


def _checkpoint_payload(path: Path) -> dict[str, object] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("format") != GROWTH_CHECKPOINT_FORMAT:
        return None
    return payload


def _latest_resume_checkpoint(directory: Path, target_tokens: int) -> Path | None:
    """Return the most advanced valid committed checkpoint for a worker.

    At the same token/step boundary an after-birth checkpoint outranks a
    before-birth checkpoint through ``growth_event_index``.  This preserves the
    exact birth state while still allowing a failed birth parity check to fall
    back to the pre-birth checkpoint.
    """

    candidates: list[tuple[tuple[int, int, int], Path]] = []
    seen: set[Path] = set()
    for pattern in ("checkpoint-*.pt", "before-birth-*.pt", "after-birth-*.pt", "final.pt"):
        for path in directory.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            payload = _checkpoint_payload(path)
            if payload is None:
                continue
            consumed = int(payload.get("consumed_tokens", -1))
            step = int(payload.get("training_step", -1))
            growth_index = int(payload.get("growth_event_index", -1))
            if consumed < 0 or consumed > target_tokens or step < 0 or growth_index < 0:
                continue
            candidates.append(((consumed, step, growth_index), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def command(
    args: argparse.Namespace,
    replicate: int,
    arm: str,
    *,
    resume_input: Path | None = None,
) -> list[str]:
    result = [
        sys.executable,
        str(Path(__file__).with_name("run_clm_progressive_growth_001_worker.py")),
        "--release-dir", str(args.release_dir),
        "--source-005-dir", str(args.source_005_dir),
        "--output-dir", str(_worker_dir(args, replicate, arm)),
        "--arm", arm,
        "--replicate", str(replicate),
        "--target-tokens", str(args.target_tokens),
        "--batch-size", str(WORKER_BATCH_SIZE),
        "--sequence-length", str(WORKER_SEQUENCE_LENGTH),
        "--eval-batches", str(WORKER_EVAL_BATCHES),
    ]
    if resume_input is not None:
        result.extend(("--resume-input", str(resume_input)))
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


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _prepare_shared_corpus(args: argparse.Namespace) -> None:
    """Materialize the deterministic corpus once before concurrent workers start."""

    source_dir = _resolve_repo_path(args.source_005_dir)
    source_manifest_path = source_dir / "corpus-manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Experiment 005 corpus manifest missing: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    train_tokens = max(
        args.target_tokens + WORKER_SEQUENCE_LENGTH + 2,
        int(source_manifest["train_stream_tokens"]),
    )
    validation_tokens = max(
        2048,
        WORKER_EVAL_BATCHES * WORKER_BATCH_SIZE * (WORKER_SEQUENCE_LENGTH + 1),
        int(source_manifest["validation_stream_tokens"]),
    )
    print(
        "Preparing shared corpus cache before GPU workers: "
        f"train={train_tokens:,}, validation={validation_tokens:,}",
        flush=True,
    )
    train, validation, _tokenizer, manifest = prepare_scaling_corpus(
        REPO_ROOT,
        source_005_dir=source_dir,
        train_stream_tokens=train_tokens,
        validation_stream_tokens=validation_tokens,
    )
    print(
        "Shared corpus cache ready: "
        f"train_sha={manifest.get('train_token_sha256')} "
        f"validation_sha={manifest.get('validation_token_sha256')}",
        flush=True,
    )
    del train, validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CLM-0.3 three-arm, three-replicate matrix")
    parser.add_argument("--output-root", type=Path, default=Path("results/clm-0.3-progressive-growth"))
    parser.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    parser.add_argument(
        "--source-005-dir",
        type=Path,
        default=Path("artifacts/experiments/005-consumer-language-bridge"),
    )
    parser.add_argument("--target-tokens", type=int, default=FORMAL_TARGET_TOKENS)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--restart-existing",
        action="store_true",
        help="ignore existing checkpoints/completed workers and start each arm from zero",
    )
    parser.add_argument("--execute", action="store_true", help="start formal workers; default is plan-only")
    args = parser.parse_args()

    planned: list[tuple[tuple[int, str], list[str]]] = []
    completed_count = 0
    for replicate, arm in jobs():
        directory = _worker_dir(args, replicate, arm)
        if args.execute and not args.restart_existing and _worker_complete(directory, args.target_tokens):
            completed_count += 1
            print(f"SKIP COMPLETE r{replicate} {arm}: {directory}", flush=True)
            continue
        resume = None
        if args.execute and not args.restart_existing:
            resume = _latest_resume_checkpoint(directory, args.target_tokens)
        cmd = command(args, replicate, arm, resume_input=resume)
        planned.append(((replicate, arm), cmd))
        if resume is None:
            print("PLAN", " ".join(cmd), flush=True)
        else:
            print(f"RESUME r{replicate} {arm} <- {resume}", flush=True)
            print("PLAN", " ".join(cmd), flush=True)

    if not args.execute:
        print("PREFLIGHT ONLY: pass --execute to launch the formal matrix.", flush=True)
        return 0

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("formal CLM-0.3 execution requires at least one CUDA GPU")

    # Corpus materialization mutates a shared cache. Do it exactly once in the
    # parent process so the two Kaggle GPU workers never race on cache files.
    _prepare_shared_corpus(args)

    capacity = min(args.max_workers or gpu_count, gpu_count)
    pending = list(planned)
    active: list[tuple[tuple[int, str], int, subprocess.Popen[bytes]]] = []

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
            progress = _worker_dir(args, replicate, arm) / "progress.json"
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
