#!/usr/bin/env python3
"""Run amended Core 007 confirmation one isolated seed process at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"
RESULTS = ROOT / "results" / "core-validation-007-functional-boundary-discovery"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-007-functional-boundary-discovery"
SEED_RUNNER = ROOT / "scripts" / "research" / "run_core_validation_007_confirmation_seed.py"
REPORTER = ROOT / "scripts" / "research" / "report_core_validation_007_confirmation.py"
PUBLISHER = ROOT / "scripts" / "research" / "publish_core_validation_007.py"
PREFLIGHT = ROOT / "scripts" / "research" / "preflight_core_validation_007.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _hydrate_partial_results() -> None:
    """Restore canonical partial seed/failure artifacts into an empty new session."""
    source = ARTIFACTS / "confirmation"
    if not source.is_dir():
        return
    for subdir in ("seeds", "failures"):
        src = source / subdir
        if not src.is_dir():
            continue
        dest = RESULTS / "confirmation" / subdir
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.glob("*.json"):
            target = dest / item.name
            if not target.exists():
                shutil.copy2(item, target)
                print(f"[core-007] hydrated {target.relative_to(ROOT)} from canonical artifacts")


def _publish(branch: str, *, push: bool) -> None:
    command = [
        sys.executable,
        str(PUBLISHER),
        "--phase",
        "confirmation",
        "--commit-results",
        "--branch",
        branch,
    ]
    if push:
        command.append("--push-results")
    _run_checked(command)


def _run_seed(seed: int, *, no_cache: bool) -> int:
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
    if returncode != 0:
        failure = RESULTS / "confirmation" / "failures" / f"seed-{seed}.json"
        if not failure.is_file():
            _atomic_json(
                failure,
                {
                    "format": "minicells.core-validation.functional-boundary-confirmation-failure.v1",
                    "experiment_id": "core-validation-007",
                    "phase": "confirmation",
                    "complete": False,
                    "seed": seed,
                    "failure_kind": "child_process_terminated_without_python_failure_record",
                    "returncode": returncode,
                    "confirmation_protocol_sha256": _sha256(AMENDMENT),
                    "log_path": str(log_path.relative_to(ROOT)),
                    "elapsed_seconds": time.time() - started,
                },
            )
    return returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", required=True)
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    seeds = [int(x) for x in amendment["confirmation_seeds"]]

    if not args.skip_preflight:
        preflight = [sys.executable, str(PREFLIGHT), "--branch", args.branch]
        if not args.push_results:
            preflight.append("--skip-push-check")
        _run_checked(preflight)

    _hydrate_partial_results()
    for seed in seeds:
        checkpoint = RESULTS / "confirmation" / "seeds" / f"seed-{seed}.json"
        if checkpoint.is_file():
            print(f"[core-007] orchestrator sees seed={seed} checkpoint; runner will validate/skip")
        returncode = _run_seed(seed, no_cache=args.no_cache)
        _run_checked([sys.executable, str(REPORTER)])
        _publish(args.branch, push=args.push_results)
        if returncode != 0:
            print(
                f"[core-007] seed={seed} failed with returncode={returncode}; "
                "completed checkpoints and failure artifacts were published. "
                "Rerun this same orchestrator after fixing the underlying issue; completed seeds will be hydrated/skipped.",
                file=sys.stderr,
            )
            return returncode if returncode > 0 else 1

    decision = json.loads(
        (RESULTS / "confirmation" / "decision.json").read_text(encoding="utf-8")
    )
    if decision.get("scientific_decision") is not True:
        raise RuntimeError("all seed processes returned success but final scientific decision was not emitted")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
