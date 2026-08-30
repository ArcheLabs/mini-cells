#!/usr/bin/env python3
"""Run/resume CLM-0.4 Preview with longitudinal public telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from minicells.clm04mini.preview import PREVIEW_SEED, run_preview
from minicells.clm04mini.protocol import CandidateOptimizerConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "research" / "validations" / "clm-0.4-mini-m1-v2-language-validation" / "protocol.json"
DEFAULT_DATA = ROOT / "results" / "clm-0.4-preview-data"
DEFAULT_OUT = ROOT / "results" / "clm-0.4-preview"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--devices")
    parser.add_argument("--seed", type=int, default=PREVIEW_SEED)
    parser.add_argument("--max-transactions", type=int, default=192)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--capability-every", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--direct-lr", type=float, default=0.003)
    parser.add_argument("--direct-steps", type=int, default=32)
    parser.add_argument("--growth-lr", type=float, default=0.003)
    parser.add_argument("--growth-steps", type=int, default=64)
    args = parser.parse_args()

    if args.max_transactions < 0 or args.max_transactions > 192:
        raise RuntimeError("--max-transactions must be in [0, 192]")
    if args.checkpoint_every <= 0 or args.capability_every <= 0:
        raise RuntimeError("checkpoint/capability cadence must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    direct = CandidateOptimizerConfig("AdamW", 32, float(args.direct_lr), int(args.direct_steps), 0.0)
    growth = CandidateOptimizerConfig("AdamW", 32, float(args.growth_lr), int(args.growth_steps), 0.0)
    result = run_preview(
        protocol_path=args.protocol,
        data_dir=args.data_dir,
        out_dir=args.out,
        seed=int(args.seed),
        device=device,
        devices=args.devices,
        max_transactions=int(args.max_transactions),
        checkpoint_every=int(args.checkpoint_every),
        capability_every=int(args.capability_every),
        resume=not args.no_resume,
        direct_optimizer=direct,
        growth_optimizer=growth,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
