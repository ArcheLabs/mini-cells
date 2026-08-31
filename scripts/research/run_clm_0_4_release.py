#!/usr/bin/env python3
"""Run CLM-0.4 Release 1M smoke or guarded 30M release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from minicells.clm04mini.release import (
    RELEASE_PROFILES,
    release_pipeline_identity,
    release_pipeline_sha256,
    release_source_fingerprint,
    run_release,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "research" / "validations" / "clm-0.4-mini-m1-v2-language-validation" / "protocol.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(RELEASE_PROFILES), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--devices")
    parser.add_argument("--smoke-readiness", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    plan = {
        "format": "minicells.clm-0.4-release-plan.v1",
        "profile": args.profile,
        "target_tokens": RELEASE_PROFILES[args.profile],
        "pipeline_sha256": release_pipeline_sha256(),
        "pipeline_identity": release_pipeline_identity(),
        "source_fingerprint": release_source_fingerprint(ROOT),
        "requires_smoke_readiness": args.profile == "release-30m",
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.data_dir is None or args.out is None:
        raise RuntimeError("--data-dir and --out are required unless --plan-only is used")
    if args.profile == "release-30m" and args.smoke_readiness is None:
        raise RuntimeError("release-30m requires --smoke-readiness from the completed 1M run")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_release(
        profile=args.profile,
        protocol_path=args.protocol,
        data_dir=args.data_dir,
        out_dir=args.out,
        repo_root=ROOT,
        device=device,
        devices=args.devices,
        smoke_readiness_path=args.smoke_readiness,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    if "readiness" in result:
        print(json.dumps(result["readiness"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
