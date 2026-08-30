#!/usr/bin/env python3
"""Run Core Validation 002C — Oracle Sparse-Assembly Representation Tomography."""

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

from minicells.write_addressability_002c import (
    CoreValidation002CConfig,
    run_primary_seed,
    summarize_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research" / "core-validation-002c-protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-002c-oracle-tomography"


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


def _smoke_config(config: CoreValidation002CConfig) -> CoreValidation002CConfig:
    base = replace(
        config.base,
        observation_dim=12,
        num_features=24,
        active_features=3,
        output_dim=5,
        latent_dim=32,
        latent_topk=4,
        pretrain_steps=3,
        pretrain_examples=96,
        pretrain_batch_size=24,
        validation_examples=48,
        oracle_probe_examples=96,
    )
    return replace(
        config,
        base=base,
        widths=(1, 2, 4),
        train_probe_examples=128,
        test_probe_examples=96,
        probe_batch_size=32,
        maximum_sparse_base_normalized_mse=100.0,
        maximum_sparse_affected_fit_error=100.0,
        maximum_sparse_leakage=100.0,
        maximum_relative_fit_error_vs_width1=100.0,
        minimum_joint_feature_success_fraction=0.0,
        minimum_feature_improvement_fraction=0.0,
    )


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
    config = CoreValidation002CConfig.from_protocol(args.protocol)
    if args.smoke:
        config = _smoke_config(config)
    device = _device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 002C requires CUDA")

    protocol_seeds = [int(value) for value in protocol["replication"]["seeds"]]
    if args.smoke:
        seeds = args.seeds or protocol_seeds[:1]
    else:
        seeds = args.seeds or protocol_seeds
        if seeds != protocol_seeds:
            raise RuntimeError(
                "formal Core Validation 002C must run exactly the frozen protocol seeds "
                f"in order: {protocol_seeds}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[core-002c] seed={seed} device={device}", flush=True)
        run = run_primary_seed(config, seed=seed, device=device)
        runs.append(run)
        gate = run["gates"]
        print(
            "[core-002c] "
            f"base={run['pretraining']['base_normalized_mse']:.6g} "
            f"pass={gate['pass']} regime={gate['representation_regime']} "
            f"best_r={gate['best_sparse_width']} "
            f"best_U={gate['best_sparse_median_affected_fit_error']:.6g} "
            f"best_L={gate['best_sparse_median_off_support_leakage']:.6g}",
            flush=True,
        )

    if args.smoke:
        decision = {
            "status": "SMOKE_ONLY",
            "pass": None,
            "scientific_decision": False,
            "passed_seeds": None,
            "total_seeds": len(runs),
        }
    else:
        gates = protocol["gates"]
        decision = summarize_experiment(
            runs,
            positive_status=str(gates["positive_status"]),
            negative_status=str(gates["negative_status"]),
            invalid_status=str(gates["invalid_status"]),
        )

    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "mode": "smoke" if args.smoke else "formal",
        "parent_experiments": protocol["parent_experiments"],
        "protocol_sha256": _sha256(args.protocol),
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
