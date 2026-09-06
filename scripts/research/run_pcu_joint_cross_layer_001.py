#!/usr/bin/env python3
"""Run PCU-JOINT-CROSS-LAYER-001 with two isolated GPU worker processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from minicells.pcu_kill_001.governance import write_json
from minicells.pcu_kill_001.joint_cross_layer import (
    DEFAULT_OUTPUT,
    DEPTH3_ROOT,
    ENGINEERING_SEED,
    EXPERIMENT_ID,
    PRIMARY_STEPS,
    READOUT_K,
    READOUT_LAYER,
    SECONDARY_STEPS,
    TRANSPORT_K,
    TRANSPORT_LAYER,
    aggregate_joint,
    load_published_depth3,
    run_joint_arm,
)


DEFAULT_WORKER_ROOT = Path("/kaggle/working/pcu-joint-cross-layer-001-workers")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root(), text=True).strip()


def assert_clean_source() -> dict[str, str | bool]:
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError(f"joint runner requires clean source tree before execution: {status[:500]}")
    return {
        "source_ref": git("branch", "--show-current"),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_dirty": False,
        "status_porcelain": "",
    }


def worker_mode(args: argparse.Namespace) -> int:
    run_joint_arm(
        steps=int(args.steps),
        output=Path(args.out),
        device=str(args.device),
        depth3_root=Path(args.depth3),
        seed=int(args.seed),
    )
    return 0


def launch_worker(*, steps: int, device: str, output: Path, depth3: Path, seed: int) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--steps",
        str(int(steps)),
        "--device",
        str(device),
        "--out",
        str(output),
        "--depth3",
        str(depth3),
        "--seed",
        str(int(seed)),
    ]
    print("+", " ".join(command), flush=True)
    return subprocess.Popen(command, cwd=repo_root())


def orchestrate(args: argparse.Namespace) -> int:
    if int(args.seed) != ENGINEERING_SEED:
        raise ValueError("joint runner is engineering-seed only")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("PCU-JOINT-CROSS-LAYER-001 requires two visible CUDA GPUs")
    source = assert_clean_source()
    depth3 = load_published_depth3(Path(args.depth3))

    output_root = Path(args.out)
    existing = sorted(path.name for path in output_root.glob("*.json")) if output_root.exists() else []
    if existing:
        raise RuntimeError(f"joint output already contains evidence; inspect before rerun: {existing}")

    worker_root = Path(args.worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    primary_file = worker_root / "JOINT_128.json"
    secondary_file = worker_root / "JOINT_256.json"
    for path in (primary_file, secondary_file):
        if path.exists():
            raise RuntimeError(f"stale external worker evidence exists: {path}")

    print(json.dumps({
        "experiment": EXPERIMENT_ID,
        "gpu0": torch.cuda.get_device_name(0),
        "gpu1": torch.cuda.get_device_name(1),
        "dual_gpu_plan": {
            "cuda:0": f"joint_{PRIMARY_STEPS}_primary_per_parameter_update_matched",
            "cuda:1": f"joint_{SECONDARY_STEPS}_secondary_extra_joint_updates",
        },
        "source": source,
        "published_depth3_source": {
            "commit": depth3.source_commit,
            "tree": depth3.source_tree,
        },
    }, indent=2), flush=True)

    primary = launch_worker(
        steps=PRIMARY_STEPS,
        device="cuda:0",
        output=primary_file,
        depth3=Path(args.depth3),
        seed=int(args.seed),
    )
    secondary = launch_worker(
        steps=SECONDARY_STEPS,
        device="cuda:1",
        output=secondary_file,
        depth3=Path(args.depth3),
        seed=int(args.seed),
    )
    codes = {"joint128": primary.wait(), "joint256": secondary.wait()}
    if any(code != 0 for code in codes.values()):
        raise RuntimeError(f"joint worker failure: {codes}")

    result = aggregate_joint(
        primary_file=primary_file,
        secondary_file=secondary_file,
        output_root=output_root,
    )
    shutil.copy2(primary_file, output_root / "JOINT_128.json")
    shutil.copy2(secondary_file, output_root / "JOINT_256.json")

    design = {
        "schema": "minicells.pcu-joint-cross-layer-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_variable": "joint_vs_sequential_optimization_of_exact_same_L15_L23_cells",
        "fixed": {
            "association_layer": 7,
            "association_state": "exact_published_L7_K64_hybrid_replay_then_frozen",
            "transport_layer": TRANSPORT_LAYER,
            "transport_k": TRANSPORT_K,
            "readout_layer": READOUT_LAYER,
            "readout_k": READOUT_K,
            "cell_identity": "exact_published_depth3_cells_no_reallocation",
            "objective": "answer-token-causal-cross-entropy",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "batch_size": 8,
        },
        "arms": {
            "sequential_control": "published_L15_128_freeze_then_L23_128",
            "joint128_primary": "L15_and_L23_jointly_trainable_for_128_steps_per_parameter_update_matched",
            "joint256_secondary": "L15_and_L23_jointly_trainable_for_256_steps_extra_joint_updates_diagnostic",
        },
        "primary_scientific_decision_arm": "joint128_primary",
        "dual_gpu_execution": {
            "required": True,
            "cuda:0": "joint128_primary",
            "cuda:1": "joint256_secondary",
            "process_isolation": True,
        },
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output_root / "DESIGN.json", design)
    write_json(output_root / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-joint-cross-layer-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output_root.name,
        "source": source,
        "dual_gpu_execution_required": True,
        "worker_devices": {"joint128": "cuda:0", "joint256": "cuda:1"},
        "formal_execution_not_started": True,
    })
    print(json.dumps({
        "status": result["status"],
        "output": str(output_root),
        "summary": result["summary"],
    }, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--depth3", type=Path, default=DEPTH3_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    args = parser.parse_args()
    if args.worker:
        if args.steps is None:
            parser.error("--worker requires --steps")
        return worker_mode(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
