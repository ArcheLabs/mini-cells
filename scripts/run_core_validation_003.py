#!/usr/bin/env python3
"""Run MiniCells Core Validation 003 — Dependency-Scoped Transactional Learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from minicells.dependency_scoped_experiment import run_primary_seed, summarize_experiment
from minicells.dependency_scoped_transactional import (
    CoreValidation003Config,
    smoke_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research" / "core-validation-003-protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-003-dependency-scoped-transactional-learning"


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
    config = CoreValidation003Config.from_protocol(args.protocol)
    if args.smoke:
        config = smoke_config(config)

    device = _device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 003 requires CUDA")

    protocol_seeds = [int(x) for x in protocol["replication"]["seeds"]]
    seeds = args.seeds or (protocol_seeds[:1] if args.smoke else protocol_seeds)
    if not args.smoke and seeds != protocol_seeds:
        raise RuntimeError(
            "formal Core Validation 003 must run exactly the frozen protocol seeds "
            f"{protocol_seeds}; use --smoke for custom execution checks"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[core-003] seed={seed} device={device}", flush=True)
        run = run_primary_seed(config, seed=seed, device=device)
        runs.append(run)
        for granularity_run in run["granularities"]:
            gate = granularity_run["gate_summary"]
            tx = gate["local_tx_summary"]
            print(
                "[core-003] "
                f"seed={seed} g={gate['granularity']} "
                f"base={gate['base_normalized_mse']:.5f} "
                f"coverage={tx['mean_dependency_coverage']:.4f} "
                f"FSR={tx['false_safe_rate']:.4f} "
                f"accept={tx['acceptance_rate']:.4f} "
                f"damage_ratio={gate['regression_damage_ratio_vs_local_always']:.4f} "
                f"gain_ratio={gate['committed_gain_ratio_vs_local_always']:.4f} "
                f"pass={gate['pass']}",
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
            granularities=config.granularities,
            positive_status=str(protocol["gates"]["positive_status"]),
            negative_status=str(protocol["gates"]["negative_status"]),
        )

    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "mode": "smoke" if args.smoke else "formal",
        "protocol_sha256": _sha256(args.protocol),
        "research_transition": protocol["research_transition"],
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
