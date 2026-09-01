#!/usr/bin/env python3
"""Run one frozen Core Validation 009B-2 phase with per-seed checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-009b2-persistent-effect-geometry"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-009b2-persistent-effect-geometry"
VALIDATION = ROOT / "research" / "validations" / "core-009b2-persistent-effect-geometry"
PROTOCOL = VALIDATION / "protocol.json"
BASIS_LOCK = VALIDATION / "basis-lock.json"


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _protocol_sha() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _format(phase: str) -> str:
    return f"minicells.core-validation.persistent-effect-geometry-{phase}-seed.v1"


def _valid(path: Path, phase: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("format") == _format(phase) and payload.get("phase") == phase and int(payload.get("seed", -1)) == seed and payload.get("protocol_sha256") == _protocol_sha() and payload.get("scientific_decision") is False


def _hydrate(phase: str, seed: int) -> bool:
    src = ARTIFACTS / phase / "seeds" / f"seed-{seed}.json"
    dst = RESULTS / phase / "seeds" / f"seed-{seed}.json"
    if not _valid(src, phase, seed):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[core009b2] hydrated published {phase} seed {seed}", flush=True)
    return True


def _report(phase: str) -> dict:
    _run([sys.executable, "scripts/research/report_core_validation_009b2.py", "--phase", phase])
    return json.loads((RESULTS / phase / "decision.json").read_text(encoding="utf-8"))


def _publish(phase: str, branch: str, secret_name: str) -> None:
    _run([sys.executable, "scripts/research/publish_core_validation_009b2.py", "--phase", phase, "--push-results", "--branch", branch, "--secret-name", secret_name])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--branch", default="codex/core-validation-009b2-persistent-effect-geometry")
    p.add_argument("--secret-name", default="GITHUB_TOKEN")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    protocol = _protocol()
    seeds = [int(x) for x in protocol[args.phase]["seeds"]]
    if args.phase == "confirmation" and not BASIS_LOCK.is_file():
        raise RuntimeError("confirmation requires committed basis-lock.json; run and publish discovery first, then refresh the checkout")

    if args.push_results:
        _run([sys.executable, "scripts/research/publish_core_validation_009b2.py", "--phase", args.phase, "--preflight-only", "--branch", args.branch, "--secret-name", args.secret_name])

    for seed in seeds:
        dst = RESULTS / args.phase / "seeds" / f"seed-{seed}.json"
        complete = False
        if not args.force:
            complete = _valid(dst, args.phase, seed) or _hydrate(args.phase, seed)
        if complete:
            print(f"[core009b2] {args.phase} seed={seed} complete; skipping", flush=True)
        else:
            _run([sys.executable, "scripts/research/run_core_validation_009b2_seed.py", "--phase", args.phase, "--seed", str(seed), "--device", args.device])
        decision = _report(args.phase)
        if args.push_results:
            _publish(args.phase, args.branch, args.secret_name)

    decision = json.loads((RESULTS / args.phase / "decision.json").read_text(encoding="utf-8"))
    if args.phase == "discovery":
        if decision.get("confirmation_allowed"):
            print("[core009b2] discovery complete. Basis lock was published; run confirmation only from a refreshed checkout containing basis-lock.json.", flush=True)
        else:
            print("[core009b2] no <=32-dimensional compact effect subspace is viable; stop rule forbids confirmation.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
