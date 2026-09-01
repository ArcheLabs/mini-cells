#!/usr/bin/env python3
"""One-cell resumable Kaggle orchestration for Core Validation 007 confirmation.

The mature repository publisher owns GitHub authentication. Each amended
confirmation seed runs in a fresh Python/CUDA child process, is atomically
checkpointed, reported, committed and pushed before the next seed begins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"
RESULTS = ROOT / "results" / "core-validation-007-functional-boundary-discovery"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-007-functional-boundary-discovery"
SEED_RUNNER = ROOT / "scripts" / "research" / "run_core_validation_007_confirmation_seed.py"
REPORTER = ROOT / "scripts" / "research" / "report_core_validation_007_confirmation.py"
PUBLISHER = ROOT / "scripts" / "research" / "publish_core_validation_007.py"
DEFAULT_BRANCH = "codex/core-validation-007-functional-boundary-discovery"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=check, text=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _hydrate_partial_results() -> None:
    """Restore canonical seed/failure checkpoints in a fresh Kaggle session."""
    source = ARTIFACTS / "confirmation"
    if not source.is_dir():
        return
    for subdir in ("seeds", "failures", "logs"):
        src = source / subdir
        if not src.is_dir():
            continue
        dest = RESULTS / "confirmation" / subdir
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if not item.is_file():
                continue
            target = dest / item.name
            if not target.exists():
                shutil.copy2(item, target)
                print(f"[core-007] hydrated {_display_path(target)} from canonical artifacts")


def _publish(branch: str, secret_name: str, *, push: bool) -> None:
    command = [
        sys.executable,
        str(PUBLISHER),
        "--phase",
        "confirmation",
        "--allow-partial",
        "--commit-results",
        "--branch",
        branch,
        "--secret-name",
        secret_name,
    ]
    if push:
        command.append("--push-results")
    _run(command)


def _preflight(branch: str, secret_name: str, *, push: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Core 007 amended confirmation requires CUDA")
    if push:
        _run(
            [
                sys.executable,
                str(PUBLISHER),
                "--preflight-only",
                "--branch",
                branch,
                "--secret-name",
                secret_name,
            ]
        )
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "PREFLIGHT_OK",
                "branch": branch,
                "winner": amendment["winner"],
                "confirmation_seeds": amendment["confirmation_seeds"],
                "gpu": torch.cuda.get_device_name(0),
                "github_push_checked": push,
                "github_secret_name": secret_name if push else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_seed(seed: int, *, no_cache: bool) -> tuple[int, Path]:
    logs = RESULTS / "confirmation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"seed-{seed}.log"
    command = [
        sys.executable,
        str(SEED_RUNNER),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if no_cache:
        command.append("--no-cache")
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== launch {time.time()} command={command!r} ===\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        log.write(f"=== returncode={returncode} elapsed={time.time() - started:.3f}s ===\n")
    return returncode, log_path


def _record_host_failure(seed: int, returncode: int, log_path: Path) -> None:
    """Record SIGKILL/OOM-style failures where Python never caught an exception."""
    checkpoint = RESULTS / "confirmation" / "seeds" / f"seed-{seed}.json"
    failure = RESULTS / "confirmation" / "failures" / f"seed-{seed}.json"
    if checkpoint.is_file() or failure.is_file():
        return
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    _atomic_json(
        failure,
        {
            "format": "minicells.core-validation.functional-boundary-confirmation-failure.v1",
            "experiment_id": "core-validation-007",
            "phase": "confirmation",
            "complete": False,
            "seed": seed,
            "winner": amendment["winner"],
            "base_protocol_sha256": amendment["base_discovery_protocol_sha256"],
            "confirmation_protocol_sha256": _sha256(AMENDMENT),
            "failure_kind": "child_process_terminated_without_python_failure_record",
            "returncode": returncode,
            "log_path": _display_path(log_path),
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--secret-name", default="GITHUB_TOKEN")
    p.add_argument("--push-results", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    seeds = [int(x) for x in amendment["confirmation_seeds"]]

    # This check happens before a new amended confirmation seed is opened.
    _preflight(args.branch, args.secret_name, push=args.push_results)
    _hydrate_partial_results()

    print(
        f"[core-007] amended confirmation seeds={seeds}; "
        "one fresh Python/CUDA process per seed",
        flush=True,
    )
    for seed in seeds:
        checkpoint = RESULTS / "confirmation" / "seeds" / f"seed-{seed}.json"
        if checkpoint.is_file():
            print(f"[core-007] seed={seed} checkpoint found; seed runner will validate and skip")
        returncode, log_path = _run_seed(seed, no_cache=args.no_cache)
        if returncode != 0:
            _record_host_failure(seed, returncode, log_path)

        # Aggregate whatever survived, even if the child process was SIGKILLed/OOMed.
        _run([sys.executable, str(REPORTER)])
        _publish(args.branch, args.secret_name, push=args.push_results)

        if returncode != 0:
            print(
                f"[core-007] seed={seed} failed with returncode={returncode}. "
                "Completed checkpoints/logs/failure record were published. Re-run this same one-cell command; "
                "matching completed seeds will be skipped.",
                file=sys.stderr,
            )
            return returncode if returncode > 0 else 1

    decision = json.loads(
        (RESULTS / "confirmation" / "decision.json").read_text(encoding="utf-8")
    )
    if decision.get("scientific_decision") is not True:
        raise RuntimeError("all amended seed processes returned success but final decision is incomplete")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
