#!/usr/bin/env python3
"""Launch and aggregate the formal CLM-0.3c counterfactual-mitosis replicates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from minicells.growth_counterfactual_reporting import aggregate_counterfactual_results  # noqa: E402
from minicells.growth_experiment_utils import git_provenance  # noqa: E402
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402


DECISION_TOKENS = 1_500_000
PROBE_TOKENS = 100_000
CONFIRM_TOKENS = 500_000
BATCH_SIZE = 8
SEQUENCE_LENGTH = 125
EVAL_BATCHES = 32
CALIBRATION_BATCHES = 16
BOOTSTRAP_SAMPLES = 2_000


def _worker_dir(root: Path, replicate: int) -> Path:
    return root / f"r{replicate}-counterfactual"


def _complete(directory: Path, code_commit: str) -> bool:
    path = directory / "replicate-result.json"
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("code_commit") == code_commit and int(value.get("births_checked", 0)) == 13


def _existing_training_commit(directory: Path) -> str | None:
    path = directory / "events.jsonl"
    if not path.exists():
        return None
    observed: str | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "worker_started" and row.get("mode") != "preflight_only":
                commit = row.get("code_commit")
                if commit:
                    observed = str(commit)
    return observed


def _command(args: argparse.Namespace, replicate: int) -> list[str]:
    result = [
        sys.executable,
        str(Path(__file__).with_name("run_clm_counterfactual_mitosis_003_worker.py")),
        "--release-dir", str(args.release_dir),
        "--source-005-dir", str(args.source_005_dir),
        "--output-dir", str(_worker_dir(args.output_root, replicate)),
        "--replicate", str(replicate),
        "--decision-tokens", str(args.decision_tokens),
        "--probe-tokens", str(args.probe_tokens),
        "--confirm-tokens", str(args.confirm_tokens),
        "--batch-size", str(BATCH_SIZE),
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--eval-batches", str(EVAL_BATCHES),
        "--calibration-batches", str(CALIBRATION_BATCHES),
        "--bootstrap-samples", str(BOOTSTRAP_SAMPLES),
    ]
    if args.execute:
        result.append("--execute")
    return result


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _prepare_corpus(args: argparse.Namespace) -> None:
    source = _resolve(args.source_005_dir)
    manifest_path = source / "corpus-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment 005 corpus manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_tokens = max(
        args.decision_tokens + args.confirm_tokens + SEQUENCE_LENGTH + 2,
        int(manifest["train_stream_tokens"]),
    )
    validation_tokens = max(
        100_000,
        EVAL_BATCHES * BATCH_SIZE * (SEQUENCE_LENGTH + 1),
        int(manifest["validation_stream_tokens"]),
    )
    print(
        f"Preparing CLM-0.3c shared corpus: train={train_tokens:,} validation={validation_tokens:,}",
        flush=True,
    )
    train, validation, _tokenizer, produced = prepare_scaling_corpus(
        REPO_ROOT,
        source_005_dir=source,
        train_stream_tokens=train_tokens,
        validation_stream_tokens=validation_tokens,
    )
    print(
        "Shared corpus ready: "
        f"train_sha={produced.get('train_token_sha256')} "
        f"validation_sha={produced.get('validation_token_sha256')}",
        flush=True,
    )
    del train, validation


def _dashboard(directory: Path, replicate: int, gpu: int) -> str:
    path = directory / "progress.json"
    if not path.exists():
        return f"r{replicate} | GPU{gpu} | starting"
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        consumed = int(row.get("consumed_tokens", 0))
        target = int(row.get("target_tokens", 0))
        phase = str(row.get("phase", "--"))
        throughput = float(row.get("tokens_per_second", 0.0) or 0.0)
        return (
            f"r{replicate} | GPU{gpu} | {consumed/1e6:.2f}/{target/1e6:.2f}M "
            f"| {phase:<24} | {throughput/1000:.1f}K tok/s"
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return f"r{replicate} | GPU{gpu} | progress updating"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run formal CLM-0.3c counterfactual mitosis")
    parser.add_argument("--output-root", type=Path, default=Path("results/clm-0.3c-counterfactual-mitosis"))
    parser.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    parser.add_argument("--source-005-dir", type=Path, default=Path("artifacts/experiments/005-consumer-language-bridge"))
    parser.add_argument("--decision-tokens", type=int, default=DECISION_TOKENS)
    parser.add_argument("--probe-tokens", type=int, default=PROBE_TOKENS)
    parser.add_argument("--confirm-tokens", type=int, default=CONFIRM_TOKENS)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--restart-existing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    provenance = git_provenance(REPO_ROOT)
    if args.execute and provenance["tracked_tree_dirty"]:
        raise RuntimeError("formal CLM-0.3c matrix requires a clean tracked Git tree")
    code_commit = str(provenance["code_commit"])

    if not args.execute:
        for replicate in range(3):
            print("PLAN", " ".join(_command(args, replicate)), flush=True)
        print(
            "Each replicate: 1.5M trunk + 12×100K candidate probes + 100K no-growth probe "
            "+ 500K selected confirm + 500K no-growth confirm.",
            flush=True,
        )
        print("PREFLIGHT ONLY: pass --execute to launch CLM-0.3c.", flush=True)
        return 0

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("formal CLM-0.3c execution requires CUDA")
    _prepare_corpus(args)
    capacity = min(args.max_workers or gpu_count, gpu_count)

    pending: list[tuple[int, list[str]]] = []
    for replicate in range(3):
        directory = _worker_dir(args.output_root, replicate)
        existing_commit = _existing_training_commit(directory)
        if args.restart_existing and directory.exists():
            print(f"RESTART r{replicate}: removing prior evidence", flush=True)
            shutil.rmtree(directory)
            existing_commit = None
        elif existing_commit is not None and existing_commit != code_commit:
            raise RuntimeError(
                f"r{replicate} contains evidence from {existing_commit[:12]}, current code is "
                f"{code_commit[:12]}; rerun with --restart-existing rather than mixing commits"
            )
        if _complete(directory, code_commit):
            print(f"SKIP r{replicate}: completed on {code_commit[:12]}", flush=True)
            continue
        pending.append((replicate, _command(args, replicate)))

    active: list[tuple[int, int, subprocess.Popen[bytes]]] = []
    completed = 3 - len(pending)
    while pending or active:
        while pending and len(active) < capacity:
            replicate, command = pending.pop(0)
            used = {gpu for _, gpu, _ in active}
            gpu = next(index for index in range(capacity) if index not in used)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            current = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(RESEARCH_ROOT) if not current else str(RESEARCH_ROOT) + os.pathsep + current
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
            active.append((replicate, gpu, process))

        if active:
            lines = [_dashboard(_worker_dir(args.output_root, rep), rep, gpu) for rep, gpu, _ in active]
            print(
                "DASHBOARD | " + " || ".join(lines) + f" | completed {completed}/3 | pending {len(pending)}",
                flush=True,
            )
        time.sleep(2)

        remaining: list[tuple[int, int, subprocess.Popen[bytes]]] = []
        failure: tuple[subprocess.Popen[bytes], int] | None = None
        for replicate, gpu, process in active:
            code = process.poll()
            if code is None:
                remaining.append((replicate, gpu, process))
            elif code != 0:
                failure = (process, code)
                break
            else:
                completed += 1
        if failure is not None:
            for _, _, process in active:
                if process.poll() is None:
                    process.terminate()
            process, code = failure
            raise subprocess.CalledProcessError(code, process.args)
        active = remaining

    result = aggregate_counterfactual_results(args.output_root, formal_gpu_experiment_run=True)
    print(json.dumps(result["decision"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
