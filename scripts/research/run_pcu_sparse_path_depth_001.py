#!/usr/bin/env python3
"""Run PCU-SPARSE-PATH-DEPTH-001 with process-isolated dual GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from minicells.pcu_kill_001.governance import git_provenance, write_json
from minicells.pcu_kill_001.sparse_path_depth import (
    DEFAULT_OUTPUT,
    DEPTHS,
    ENGINEERING_SEED,
    EXPERIMENT_ID,
    READOUT_K,
    READOUT_LAYER,
    READOUT_STEPS,
    TOTAL_ADDED_K,
    TOTAL_ADDED_STEPS,
    TRANSPORT_K_TOTAL,
    TRANSPORT_STEPS_TOTAL,
    aggregate_depth_sweep,
    run_topology,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_ROOT = Path("/kaggle/working/pcu-sparse-path-depth-001-workers")
FORMAL_REGISTRY = REPO_ROOT / "research/formal_seed_registry.json"
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
FORMAL_SEEDS = (26090511, 26090512, 26090513)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def assert_formal_untouched() -> None:
    payload = json.loads(FORMAL_REGISTRY.read_text(encoding="utf-8"))
    states = {int(row["seed"]): str(row["state"]) for row in payload.get("seeds", [])}
    expected = {seed: "RESERVED_UNTOUCHED" for seed in FORMAL_SEEDS}
    if states != expected:
        raise RuntimeError(f"formal seed registry changed: {states}")
    if _git("hash-object", str(FORMAL_REGISTRY.relative_to(REPO_ROOT))) != FORMAL_REGISTRY_SHA:
        raise RuntimeError("formal seed registry blob changed")


def _worker_cmd(*, depth: int, device: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--depth",
        str(int(depth)),
        "--device",
        str(device),
        "--worker-out",
        str(output),
    ]


def _run_wave(assignments: list[tuple[int, str, Path]]) -> None:
    processes: list[tuple[int, str, subprocess.Popen[str]]] = []
    for depth, device, output in assignments:
        print(f"[pcu-path-depth] launch depth{depth} on {device}", flush=True)
        process = subprocess.Popen(_worker_cmd(depth=depth, device=device, output=output), cwd=REPO_ROOT, text=True)
        processes.append((depth, device, process))
    failures: list[tuple[int, str, int]] = []
    for depth, device, process in processes:
        code = process.wait()
        if code != 0:
            failures.append((depth, device, code))
        else:
            print(f"[pcu-path-depth] depth{depth} complete on {device}", flush=True)
    if failures:
        raise RuntimeError(f"dual-GPU worker failure(s): {failures}")


def run_dual_gpu(*, output_root: Path, worker_root: Path) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("PCU-SPARSE-PATH-DEPTH-001 requires two CUDA GPUs")
    assert_formal_untouched()
    source = git_provenance(REPO_ROOT)
    if source.get("source_dirty") is not False:
        raise RuntimeError("sparse path depth sweep requires a clean source tree")

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.glob("*.json")):
        raise RuntimeError(f"output already exists; inspect before rerun: {output_root}")
    worker_root = Path(worker_root)
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_files = {depth: worker_root / f"DEPTH_{depth}.json" for depth in DEPTHS}

    print(json.dumps({
        "experiment": EXPERIMENT_ID,
        "gpu0": torch.cuda.get_device_name(0),
        "gpu1": torch.cuda.get_device_name(1),
        "wave1": {"depth3": "cuda:0", "depth4": "cuda:1"},
        "wave2": {"depth5": "cuda:0"},
        "total_added_k_each": TOTAL_ADDED_K,
        "total_added_steps_each": TOTAL_ADDED_STEPS,
    }, indent=2), flush=True)

    _run_wave([
        (3, "cuda:0", worker_files[3]),
        (4, "cuda:1", worker_files[4]),
    ])
    _run_wave([
        (5, "cuda:0", worker_files[5]),
    ])

    assert_formal_untouched()
    result = aggregate_depth_sweep(worker_files=worker_files, output_root=output_root)
    topologies = {key: value["layers"] for key, value in result["depths"].items()}
    write_json(output_root / "DESIGN.json", {
        "schema": "minicells.pcu-sparse-path-depth-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_question": "does_distributing_fixed_transport_readout_budget_across_more_nested_layers_improve_native_generation",
        "fixed": {
            "association_state": "exact_published_L7_K64_hybrid_replay_then_frozen",
            "readout_layer": READOUT_LAYER,
            "readout_k": READOUT_K,
            "readout_steps": READOUT_STEPS,
            "transport_k_total": TRANSPORT_K_TOTAL,
            "transport_steps_total": TRANSPORT_STEPS_TOTAL,
            "total_added_k_each": TOTAL_ADDED_K,
            "total_added_steps_each": TOTAL_ADDED_STEPS,
            "objective_for_added_layers": "answer-token-causal-cross-entropy",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "batch_size": 8,
            "allocation": "first64_A_train_answer_CE_gradient_under_preceding_frozen_path",
        },
        "changed": {
            "path_depths": list(DEPTHS),
            "nested_topologies": topologies,
            "transport_budget_split_evenly_across_transport_layers": True,
        },
        "execution": {
            "requires_two_gpus": True,
            "process_isolation": True,
            "wave1": {"depth3": "cuda:0", "depth4": "cuda:1"},
            "wave2": {"depth5": "cuda:0"},
            "worker_evidence_generated_outside_repo_then_aggregated": True,
        },
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    })
    write_json(output_root / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-sparse-path-depth-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output_root.name,
        "source": source,
        "dual_gpu_execution": True,
        "gpu_count": 2,
        "formal_execution_not_started": True,
    })
    assert_formal_untouched()
    print(json.dumps({
        "status": result["status"],
        "best_depth": result["best_depth"],
        "best_direct_accuracy": result["best_direct_accuracy"],
        "depths": result["depths"],
        "formal_seeds": "RESERVED_UNTOUCHED",
    }, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--device")
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    args = parser.parse_args()

    if int(args.seed) != ENGINEERING_SEED:
        raise ValueError("engineering seed is frozen at 26090501")
    if args.worker:
        if args.depth not in DEPTHS or args.device is None or args.worker_out is None:
            raise ValueError("worker mode requires --depth {3,4,5}, --device, and --worker-out")
        assert_formal_untouched()
        result = run_topology(depth=args.depth, output=args.worker_out, device=args.device, seed=args.seed)
        assert_formal_untouched()
        print(json.dumps({
            "worker": True,
            "depth": args.depth,
            "device": args.device,
            "direct_accuracy": result["metrics"]["direct_accuracy"],
            "ranking_eval_accuracy": result["metrics"]["ranking_eval_accuracy"],
        }, indent=2), flush=True)
        return 0

    run_dual_gpu(output_root=args.out, worker_root=args.worker_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
