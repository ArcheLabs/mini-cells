#!/usr/bin/env python3
"""Run CLM-0.4-mini M1 development-seed calibration or plan-only validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import torch

from minicells.clm04mini.calibration import run_calibration, write_plan_only


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "clm-0.4-mini-language-validation"
DEFAULT_PROTOCOL = VALIDATION / "protocol.json"
DEFAULT_ASSETS = VALIDATION / "calibration-assets.json"
DEFAULT_PLAN = VALIDATION / "calibration-plan.json"
DEFAULT_TEMPLATE = VALIDATION / "protocol-lock.template.json"
DEFAULT_OUT = ROOT / "results" / "clm-0.4-mini-m1-calibration"


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _tracked_tree_dirty() -> bool:
    return bool(
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--lock-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=90401)
    parser.add_argument("--confirm-development-seed", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.plan_only:
        result = write_plan_only(
            protocol_path=args.protocol,
            committed_plan_path=args.plan,
            expected_assets_path=args.expected_assets,
            out_dir=args.out,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.confirm_development_seed != 90401 or args.seed != 90401:
        raise RuntimeError(
            "calibration can open development seed 90401 only; pass "
            "--confirm-development-seed 90401 explicitly"
        )
    if args.data_dir is None:
        raise RuntimeError("--data-dir is required for development calibration")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    # Seed before constructing the formal model. The base checkpoint is then trained once
    # and reused unchanged for every candidate configuration.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    result = run_calibration(
        protocol_path=args.protocol,
        expected_assets_path=args.expected_assets,
        committed_plan_path=args.plan,
        protocol_lock_template_path=args.lock_template,
        data_dir=args.data_dir,
        out_dir=args.out,
        seed=args.seed,
        device=device,
        code_commit=_git(["rev-parse", "HEAD"]),
        code_tree=_git(["rev-parse", "HEAD^{tree}"]),
        tracked_tree_dirty=_tracked_tree_dirty(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
