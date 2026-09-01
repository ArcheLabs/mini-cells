#!/usr/bin/env python3
"""One-cell Kaggle orchestration for resumable Core Validation 007 confirmation.

Each amended confirmation seed runs in a fresh Python/CUDA child process. After
every seed attempt, the current partial/final report is published through the
repository's historical GITHUB_TOKEN + GIT_ASKPASS path.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"
DEFAULT_BRANCH = "codex/core-validation-007-functional-boundary-discovery"
RUNNER = ROOT / "scripts" / "research" / "run_core_validation_007.py"
REPORT = ROOT / "scripts" / "research" / "report_core_validation_007.py"
PUBLISH = ROOT / "scripts" / "research" / "publish_core_validation_007.py"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=check, text=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--secret-name", default="GITHUB_TOKEN")
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()


def _publish(branch: str, secret_name: str, *, push: bool) -> None:
    command = [
        sys.executable,
        str(PUBLISH),
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


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Core 007 amended confirmation requires CUDA")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    seeds = [int(x) for x in amendment["confirmation_seeds"]]

    if not args.no_push:
        # Historical publisher auth is validated before any new scientific seed is opened.
        _run(
            [
                sys.executable,
                str(PUBLISH),
                "--preflight-only",
                "--branch",
                args.branch,
                "--secret-name",
                args.secret_name,
            ]
        )

    print(
        f"[core-007-kaggle] amended confirmation seeds={seeds} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    for seed in seeds:
        print(f"[core-007-kaggle] starting/resuming seed {seed} in a fresh process", flush=True)
        result = _run(
            [
                sys.executable,
                str(RUNNER),
                "--phase",
                "confirmation",
                "--device",
                args.device,
                "--seed",
                str(seed),
            ],
            check=False,
        )
        # The runner writes an aggregate raw.json even after a caught seed failure.
        report_result = _run(
            [sys.executable, str(REPORT), "--phase", "confirmation"],
            check=False,
        )
        if report_result.returncode == 0:
            _publish(args.branch, args.secret_name, push=not args.no_push)
        else:
            print(
                "[core-007-kaggle] report generation failed; seed checkpoints remain in results/",
                file=sys.stderr,
            )
        if result.returncode != 0:
            print(
                f"[core-007-kaggle] seed {seed} failed; completed checkpoints were preserved and published. "
                "Re-run this same command after fixing the recorded failure; completed seeds will be skipped.",
                file=sys.stderr,
            )
            return result.returncode

    print("[core-007-kaggle] all amended confirmation seeds complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
