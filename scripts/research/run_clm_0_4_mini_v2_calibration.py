#!/usr/bin/env python3
"""Run CLM-0.4-mini M1-v2 development calibration or plan-only validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from minicells.clm04mini.calibration import verify_committed_plan
from minicells.clm04mini.protocol import load_protocol
from minicells.clm04mini.v2 import run_v2_calibration


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = (
    ROOT
    / "research"
    / "validations"
    / "clm-0.4-mini-m1-v2-language-validation"
)
DEFAULT_PROTOCOL = VALIDATION / "protocol.json"
DEFAULT_ASSET_LOCK = VALIDATION / "asset-lock.json"
DEFAULT_PLAN = VALIDATION / "calibration-plan.json"
DEFAULT_TEMPLATE = VALIDATION / "protocol-lock.template.json"
DEFAULT_OUT = ROOT / "results" / "clm-0.4-mini-m1-v2-calibration"


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _tracked_tree_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT, check=False
        ).returncode
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--asset-lock", type=Path, default=DEFAULT_ASSET_LOCK)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--lock-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--devices")
    parser.add_argument("--seed", type=int, default=90402)
    parser.add_argument("--confirm-development-seed", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if args.plan_only:
        plan = verify_committed_plan(protocol, args.plan)
        args.out.mkdir(parents=True, exist_ok=True)
        decision = {
            "format": "minicells.clm-0.4-mini.m1-v2-calibration.v1",
            "status": "V2_CALIBRATION_PLAN_ONLY",
            "scientific_decision": False,
            "development_seed_observed": False,
            "formal_seeds_observed": False,
            "candidate_count": plan["candidate_count"],
            "plan_sha256": plan["plan_sha256"],
            "asset_lock_status": json.loads(
                args.asset_lock.read_text(encoding="utf-8")
            )["lock_status"],
        }
        (args.out / "decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out / "summary.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    if args.seed != 90402 or args.confirm_development_seed != 90402:
        raise RuntimeError(
            "M1-v2 may open development seed 90402 only; pass "
            "--seed 90402 --confirm-development-seed 90402 explicitly"
        )
    if args.data_dir is None:
        raise RuntimeError("--data-dir is required for M1-v2 calibration")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_v2_calibration(
        protocol_path=args.protocol,
        asset_lock_path=args.asset_lock,
        committed_plan_path=args.plan,
        protocol_lock_template_path=args.lock_template,
        data_dir=args.data_dir,
        out_dir=args.out,
        seed=args.seed,
        device=device,
        devices=args.devices,
        code_commit=_git(["rev-parse", "HEAD"]),
        code_tree=_git(["rev-parse", "HEAD^{tree}"]),
        tracked_tree_dirty=_tracked_tree_dirty(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
