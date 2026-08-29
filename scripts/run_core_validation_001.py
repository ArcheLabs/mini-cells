#!/usr/bin/env python3
"""Run MiniCells Core Validation 001."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from minicells.knowledge_subsumption import (
    KnowledgeSubsumptionConfig,
    effective_heldout_fraction,
    summarize_experiment,
    train_oracle_reference,
    train_sequential_run,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research" / "core-validation-001-protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-001-knowledge-subsumption"


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_tree_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        return bool(unstaged or staged)
    except OSError:
        return None


def _load_protocol(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _smoke_config(config: KnowledgeSubsumptionConfig) -> KnowledgeSubsumptionConfig:
    return replace(
        config,
        modulus=7,
        curriculum_fractions=(0.20, 0.20),
        phase_steps=(3, 3),
        eval_interval_steps=1,
        embedding_dim=8,
        num_cells=4,
        neurons_per_cell=2,
        batch_size=16,
        probe_examples_per_partition=8,
        path_cells=2,
        key_frequency_pairs=1,
        early_minimum_seen_accuracy=0.0,
        early_maximum_unseen_accuracy=1.0,
        late_minimum_old_accuracy=0.0,
        late_minimum_current_accuracy=0.0,
        late_minimum_heldout_accuracy=0.0,
        restricted_minimum_old_accuracy=0.0,
        restricted_minimum_heldout_accuracy=0.0,
        early_excluded_minimum_seen_accuracy=0.0,
        late_excluded_maximum_old_accuracy=1.0,
        late_excluded_maximum_heldout_accuracy=1.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = _load_protocol(args.protocol)
    config = KnowledgeSubsumptionConfig.from_protocol(args.protocol)
    if args.smoke:
        config = _smoke_config(config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 001 requires CUDA")

    protocol_seeds = [int(value) for value in protocol["replication"]["seeds"]]
    seeds = args.seeds or (protocol_seeds[:1] if args.smoke else protocol_seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out / "checkpoints"
    started = time.time()
    runs: list[dict[str, object]] = []
    tasks = ("modular_addition", "balanced_random_labels")
    for seed in seeds:
        for task in tasks:
            print(f"[core-001] task={task} seed={seed} device={device}", flush=True)
            run = train_sequential_run(
                config,
                seed=seed,
                task=task,
                device=device,
                save_dir=None if args.smoke else checkpoint_dir,
            )
            runs.append(run)
            gates = run["gates"]
            late = run["late"]
            mech = run["mechanistic"]
            print(
                "[core-001] "
                f"pass={gates['pass']} "
                f"heldout={late['heldout']['accuracy']:.4f} "
                f"fourier_gain={mech['fourier_concentration_gain']:.4f} "
                f"path_reuse_gain={mech['path_reuse_gain']:.4f}",
                flush=True,
            )

    oracle = None
    oracle_enabled = bool(protocol["replication"].get("oracle_reference", False))
    if oracle_enabled and not args.skip_oracle and not args.smoke:
        oracle_seed = int(protocol["replication"]["oracle_reference_seed"])
        print(f"[core-001] oracle seed={oracle_seed}", flush=True)
        oracle = train_oracle_reference(config, seed=oracle_seed, device=device)

    decision = summarize_experiment(
        runs,
        positive_status=str(protocol["gates"]["positive_status"]),
        negative_status=str(protocol["gates"]["negative_status"]),
    )
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
        "heldout_fraction": effective_heldout_fraction(config),
        "runs": runs,
        "oracle_reference": oracle,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    raw_path = args.out / "raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
