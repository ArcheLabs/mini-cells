#!/usr/bin/env python3
"""Run Core Validation 005 — Replay-Free Subspace-Certified Mitosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from minicells.subspace_mitosis_005 import (
    CoreValidation005Config,
    run_primary_seed,
    smoke_config,
    summarize_experiment,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "research" / "validations" / "core-005-subspace-certified-mitosis" / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-005-subspace-certified-mitosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = CoreValidation005Config.from_protocol(args.protocol)
    formal_seeds = [int(x) for x in protocol["replication"]["formal_seeds"]]
    if args.smoke:
        config = smoke_config(config)
        seeds = args.seeds or formal_seeds[:1]
    else:
        seeds = args.seeds or formal_seeds
        if seeds != formal_seeds:
            raise RuntimeError(f"formal Core Validation 005 must run exactly {formal_seeds}")

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[core-005] seed={seed} device=cpu", flush=True)
        run = run_primary_seed(config, seed=seed)
        runs.append(run)
        gate = run["gate_summary"]
        growth = run["variants"]["certificate_growth"]["summary"]
        wrong = run["variants"]["wrong_certificate"]["summary"]
        print(
            "[core-005] "
            f"seed={seed} pass={gate['pass']} "
            f"gain_ratio={gate['committed_gain_ratio_vs_unsafe']:.6f} "
            f"damage_ratio={gate['regression_damage_ratio_vs_unsafe']:.6f} "
            f"false_safe={growth['false_safe_count']} "
            f"mismatch={growth['decision_mismatch_count']} "
            f"rescue={growth['growth_rescue_rate']:.4f} "
            f"reuse={growth['child_reuse_acceptance_rate']:.4f} "
            f"wrong_false_safe={wrong['false_safe_count']}",
            flush=True,
        )

    if args.smoke:
        decision = {
            "status": str(protocol["gates"]["smoke_status"]),
            "pass": None,
            "scientific_decision": False,
            "passed_seeds": None,
            "total_seeds": len(runs),
            "reason": "Smoke mode validates execution only and cannot emit a scientific decision.",
        }
    else:
        decision = summarize_experiment(
            runs,
            positive_status=str(protocol["gates"]["positive_status"]),
            negative_status=str(protocol["gates"]["negative_status"]),
        )

    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "mode": "smoke" if args.smoke else "formal",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "parent_experiment": protocol["parent_experiment"],
        "provenance": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
        },
        "runs": runs,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    raw = args.out / "raw.json"
    raw.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
