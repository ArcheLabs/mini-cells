#!/usr/bin/env python3
"""Run MiniCells Core Validation 002 — Write Addressability under Superposition."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from minicells.write_addressability import WriteAddressabilityConfig
from minicells.write_addressability_experiment import (
    oracle_exact_zero_check,
    run_primary_seed,
    summarize_experiment,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "research" / "validations" / "core-002-write-addressability" / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-002-write-addressability"


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
            ["git", "diff", "--quiet"],
            cwd=ROOT,
            check=False,
            stderr=subprocess.DEVNULL,
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False, stderr=subprocess.DEVNULL
        ).returncode
        return bool(unstaged or staged)
    except OSError:
        return None


def _smoke_config(config: WriteAddressabilityConfig) -> WriteAddressabilityConfig:
    return replace(
        config,
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
        global_edit_steps=2,
        dense_edit_steps=2,
        moe_edit_steps=2,
        maximum_base_normalized_mse=100.0,
        maximum_candidate_update_error=100.0,
        maximum_baseline_update_error=100.0,
        maximum_leakage_ratio=1e9,
        minimum_mechanistic_correlation=-1.0,
        minimum_permutation_degradation=0.0,
    )


def _load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--sweep", action="store_true", help="run descriptive recovery-load sweep")
    return parser.parse_args()


def _run_sweep(
    base: WriteAddressabilityConfig,
    protocol: dict[str, Any],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    sweep = protocol["diagnostic_sweep"]
    seed = int(sweep["seed"])
    rows: list[dict[str, Any]] = []
    for index, condition in enumerate(sweep["conditions"]):
        config = replace(
            base,
            num_features=int(condition["num_features"]),
            active_features=int(condition["active_features"]),
            latent_dim=int(condition["latent_dim"]),
            latent_topk=int(condition["latent_topk"]),
            pretrain_steps=int(sweep["pretrain_steps"]),
            edit_count=int(sweep["edit_count"]),
        )
        print(
            f"[core-002-sweep] condition={index} F={config.num_features} "
            f"k={config.active_features} alpha={config.superposition_load:.3f} "
            f"rho={config.recovery_load:.3f}",
            flush=True,
        )
        run = run_primary_seed(config, seed=seed + index * 1009, device=device)
        rows.append(
            {
                "condition": index,
                "seed": seed + index * 1009,
                "num_features": config.num_features,
                "active_features": config.active_features,
                "latent_dim": config.latent_dim,
                "latent_topk": config.latent_topk,
                "superposition_load": config.superposition_load,
                "recovery_load": config.recovery_load,
                "summary": run["summary"],
                "gates": run["gates"],
                "pretraining": run["pretraining"],
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    protocol = _load_protocol(args.protocol)
    config = WriteAddressabilityConfig.from_protocol(args.protocol)
    if args.smoke:
        config = _smoke_config(config)
    device = _device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 002 requires CUDA")

    protocol_seeds = [int(value) for value in protocol["replication"]["seeds"]]
    seeds = args.seeds or (protocol_seeds[:1] if args.smoke else protocol_seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[core-002] seed={seed} device={device}", flush=True)
        run = run_primary_seed(config, seed=seed, device=device)
        runs.append(run)
        candidate = run["summary"]["inferred_address"]
        baseline = run["summary"]["global_write"]
        print(
            "[core-002] "
            f"pass={run['gates']['pass']} "
            f"U={candidate['median_update_error']:.6g} "
            f"L={candidate['median_write_leakage']:.6g} "
            f"global_L={baseline['median_write_leakage']:.6g} "
            f"corr={candidate['mechanistic_pearson']}",
            flush=True,
        )

    decision = summarize_experiment(
        runs,
        positive_status=str(protocol["gates"]["positive_status"]),
        negative_status=str(protocol["gates"]["negative_status"]),
    )
    sweep_rows = None
    if args.sweep and not args.smoke:
        sweep_rows = _run_sweep(config, protocol, device=device)

    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "mode": "smoke" if args.smoke else "formal",
        "provenance": {
            "code_commit": _git(["rev-parse", "HEAD"]),
            "code_tree": _git(["rev-parse", "HEAD^{tree}"]),
            "tracked_tree_dirty": _tracked_tree_dirty(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "config": asdict(config),
        "oracle_exact_zero_check": oracle_exact_zero_check(seed=seeds[0]),
        "runs": runs,
        "diagnostic_sweep": sweep_rows,
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
