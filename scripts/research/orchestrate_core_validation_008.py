#!/usr/bin/env python3
"""Run Core Validation 008 formal seeds in resumable fresh processes."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-008-certified-functional-atoms"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-008-certified-functional-atoms"
PROTOCOL = ROOT / "research" / "validations" / "core-008-certified-functional-atoms" / "protocol.json"


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, check=check, text=True)


def _valid_seed(path: Path, seed: int, expected_manifest: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("format") == "minicells.core-validation.certified-functional-atoms.seed.v1"
        and int(payload.get("seed", -1)) == seed
        and str(payload.get("data_manifest_sha256", "")) == expected_manifest
    )


def _hydrate_seed(seed: int, expected_manifest: str) -> bool:
    src = ARTIFACTS / "seeds" / f"seed-{seed}.json"
    dst = RESULTS / "seeds" / f"seed-{seed}.json"
    if not _valid_seed(src, seed, expected_manifest):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[core008] hydrated published seed {seed}", flush=True)
    return True


def _report() -> None:
    _run([sys.executable, "scripts/research/report_core_validation_008.py"])


def _publish(branch: str, secret_name: str, *, push: bool) -> None:
    args = [
        sys.executable,
        "scripts/research/publish_core_validation_008.py",
        "--branch",
        branch,
        "--secret-name",
        secret_name,
    ]
    if push:
        args.append("--push-results")
    _run(args)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default="codex/core-validation-008-certified-functional-atoms")
    p.add_argument("--secret-name", default="GITHUB_TOKEN")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--signature-batch-size", type=int, default=8)
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seeds = list(map(int, protocol["replication"]["formal_seeds"]))
    expected_manifest = str(protocol["data"]["expected_manifest_sha256"])

    if args.push_results:
        _run([
            sys.executable,
            "scripts/research/publish_core_validation_008.py",
            "--preflight-only",
            "--branch",
            args.branch,
            "--secret-name",
            args.secret_name,
        ])

    RESULTS.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        dst = RESULTS / "seeds" / f"seed-{seed}.json"
        if not args.force and (_valid_seed(dst, seed, expected_manifest) or _hydrate_seed(seed, expected_manifest)):
            print(f"[core008] seed={seed} already complete; skipping", flush=True)
            continue
        try:
            _run([
                sys.executable,
                "scripts/research/run_core_validation_008_seed.py",
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--signature-batch-size",
                str(args.signature_batch_size),
            ])
        except subprocess.CalledProcessError:
            print(f"[core008] seed={seed} failed; recording resumable partial state", flush=True)
            _report()
            if args.push_results:
                _publish(args.branch, args.secret_name, push=True)
            raise

    _report()
    if args.push_results:
        _publish(args.branch, args.secret_name, push=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
