#!/usr/bin/env python3
"""Run MiniCells Core Validation 002B — Sparse Functional Write Assemblies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from minicells.write_addressability_002b import CoreValidation002BConfig
from minicells.write_addressability_002b_experiment import run_primary_seed, summarize_experiment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research" / "validations" / "core-002b-sparse-write-assembly" / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-002b-sparse-write-assembly"


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_tree_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT, check=False, stderr=subprocess.DEVNULL
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False, stderr=subprocess.DEVNULL
        ).returncode
        return bool(unstaged or staged)
    except OSError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _smoke_config(config: CoreValidation002BConfig) -> CoreValidation002BConfig:
    base = replace(
        config.base,
        observation_dim=16,
        num_features=32,
        active_features=2,
        output_dim=8,
        latent_dim=48,
        latent_topk=4,
        pretrain_steps=3,
        pretrain_examples=96,
        pretrain_batch_size=24,
        validation_examples=48,
        edit_count=3,
        edit_examples=6,
        affected_examples=24,
        invariant_examples=24,
        retention_examples_per_edit=4,
        oracle_probe_examples=96,
        dense_edit_steps=2,
        moe_edit_steps=2,
    )
    return replace(config, base=base, address_widths=(1, 2, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = CoreValidation002BConfig.from_protocol(args.protocol)
    if args.smoke:
        config = _smoke_config(config)
    device = _device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 002B requires CUDA")

    protocol_seeds = [int(value) for value in protocol["replication"]["seeds"]]
    seeds = args.seeds or (protocol_seeds[:1] if args.smoke else protocol_seeds)
    if not args.smoke and seeds != protocol_seeds:
        raise RuntimeError(
            "formal Core Validation 002B must run exactly the frozen protocol seeds "
            f"{protocol_seeds}; use --smoke for execution-only custom-seed checks"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[core-002b] seed={seed} device={device}", flush=True)
        run = run_primary_seed(config, seed=seed, device=device)
        runs.append(run)
        gates = run["gates"]
        print(
            "[core-002b] "
            f"best_r={gates['best_width']} "
            f"U={gates['best_median_update_error']:.6g} "
            f"L={gates['best_median_write_leakage']:.6g} "
            f"relative_U={gates['relative_update_error_vs_width1']:.6g} "
            f"pass={gates['pass']}",
            flush=True,
        )

    if args.smoke:
        decision = {
            "status": "SMOKE_ONLY",
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
        "mode": "smoke" if args.smoke else "formal",
        "protocol_sha256": _sha256(args.protocol),
        "parent_experiment": protocol["parent_experiment"],
        "provenance": {
            "code_commit": _git(["rev-parse", "HEAD"]),
            "code_tree": _git(["rev-parse", "HEAD^{tree}"]),
            "tracked_tree_dirty": _tracked_tree_dirty(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "runs": runs,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    raw_path = args.out / "raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
