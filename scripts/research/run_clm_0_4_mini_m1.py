#!/usr/bin/env python3
"""Run CLM-0.4-mini M1 infrastructure smoke only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from minicells.clm04mini.m1 import run_m1_infrastructure_smoke


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "research" / "validations" / "clm-0.4-mini-language-validation" / "protocol.json"
)
DEFAULT_OUT = ROOT / "results" / "clm-0.4-mini-m1-infrastructure"


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_tree_dirty() -> bool | None:
    try:
        a = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        b = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
        return bool(a or b)
    except OSError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=90400)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke:
        raise RuntimeError(
            "M1 infrastructure runner requires --smoke. Development seed 90401 calibration "
            "is intentionally not executed by this stage."
        )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = run_m1_infrastructure_smoke(
        protocol_path=args.protocol,
        out_dir=args.out,
        device=device,
        seed=args.seed,
    )
    protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest = {
        "format": "minicells.clm-0.4-mini.m1-infrastructure-run-manifest.v1",
        "mode": "infrastructure-smoke",
        "seed": int(args.seed),
        "protocol_sha256": protocol_sha256,
        "code_commit": _git(["rev-parse", "HEAD"]),
        "code_tree": _git(["rev-parse", "HEAD^{tree}"]),
        "tracked_tree_dirty": _tracked_tree_dirty(),
        "device": str(device),
        "scientific_decision": False,
        "status": payload["status"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
