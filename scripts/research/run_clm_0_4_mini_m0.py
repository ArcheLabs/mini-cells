#!/usr/bin/env python3
"""Run CLM-0.4-mini M0 execution smoke. Never emits a scientific decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from minicells.clm04mini import M0_SEED, run_m0


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "research" / "validations" / "clm-0.4-mini-language-validation" / "protocol.json"
)
DEFAULT_OUT = ROOT / "results" / "clm-0.4-mini-m0"


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dirty() -> bool | None:
    try:
        a = subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT, check=False, stderr=subprocess.DEVNULL
        ).returncode
        b = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ROOT,
            check=False,
            stderr=subprocess.DEVNULL,
        ).returncode
        return bool(a or b)
    except OSError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=M0_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("experiment_id") != "clm-0.4-mini-language-validation":
        raise RuntimeError("unexpected CLM-0.4-mini protocol")

    summary = run_m0(args.out, device=device, seed=args.seed)
    provenance = {
        "code_commit": _git(["rev-parse", "HEAD"]),
        "code_tree": _git(["rev-parse", "HEAD^{tree}"]),
        "tracked_tree_dirty": _dirty(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    summary["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    summary["provenance"] = provenance
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    decision_path = args.out / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["protocol_sha256"] = summary["protocol_sha256"]
    decision["provenance"] = provenance
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote M0 smoke artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
