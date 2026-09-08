#!/usr/bin/env python3
"""Run protocol-v3 PCU Hybrid Reattachment with two isolated GPU workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicells.pcu_kill_001.governance import write_json  # noqa: E402
from minicells.pcu_kill_001.hybrid_reattachment_v3 import (  # noqa: E402
    ALPHA_SWEEP,
    DEFAULT_OUTPUT,
    EQUIVALENCE_MAX_ABS_LOGIT_DIFF,
    EXPERIMENT_ID,
    FIRST_ENGINEERING_SOURCE_COMMIT,
    MAX_CONTROL_ANSWER_NLL_INCREASE,
    MIN_CAUSAL_RANKING_GAIN,
    PROTOCOL_VERSION,
    RESTORATION_MAX_ABS_LOGIT_DIFF,
    aggregate_dual_gpu,
    run_amplitude_sweep_arm,
    run_primary_arm,
)
from minicells.pcu_kill_001.objective_alignment import ASSOCIATION_FLOOR  # noqa: E402


DEFAULT_WORKER_ROOT = Path("/kaggle/working/pcu-hybrid-reattachment-001-v3-workers")


def repo_root() -> Path:
    return ROOT


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root(), text=True).strip()


def assert_clean_source() -> dict[str, str | bool]:
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError(f"v3 orchestrator requires clean source before execution: {status[:500]}")
    return {
        "source_ref": git("branch", "--show-current"),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_dirty": False,
        "status_porcelain": "",
        "generated_artifact_status_porcelain": "",
    }


def worker_mode(args: argparse.Namespace) -> int:
    if args.arm == "primary":
        result = run_primary_arm(output=Path(args.out), device=str(args.device))
    elif args.arm == "sweep":
        result = run_amplitude_sweep_arm(output=Path(args.out), device=str(args.device))
    else:
        raise ValueError(f"unknown worker arm: {args.arm}")
    print(json.dumps({
        "arm": args.arm,
        "status": result["status"],
        "valid_run": result["valid_run"],
        "output": str(args.out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def launch_worker(*, arm: str, device: str, output: Path) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--arm",
        arm,
        "--device",
        device,
        "--out",
        str(output),
    ]
    print("+", " ".join(command), flush=True)
    return subprocess.Popen(command, cwd=repo_root())


def orchestrate(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("PCU-HYBRID-REATTACHMENT-001 v3 requires two visible CUDA GPUs")
    source = assert_clean_source()
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"v3 output already contains evidence; inspect before rerun: {output}")

    worker_root = Path(args.worker_root)
    primary_root = worker_root / "primary"
    sweep_root = worker_root / "sweep"
    for root in (primary_root, sweep_root):
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"stale external worker evidence exists: {root}")
        root.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "experiment": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "gpu0": torch.cuda.get_device_name(0),
        "gpu1": torch.cuda.get_device_name(1),
        "dual_gpu_plan": {
            "cuda:0": "primary causal ON/OFF + matched-graph equivalence",
            "cuda:1": "alpha sweep without additional training",
        },
        "alpha_grid": list(ALPHA_SWEEP),
        "source": source,
    }, indent=2), flush=True)

    primary = launch_worker(arm="primary", device="cuda:0", output=primary_root)
    sweep = launch_worker(arm="sweep", device="cuda:1", output=sweep_root)
    codes = {"primary": primary.wait(), "sweep": sweep.wait()}
    if any(code != 0 for code in codes.values()):
        raise RuntimeError(f"dual-GPU worker failure: {codes}")

    result = aggregate_dual_gpu(
        primary_root=primary_root,
        sweep_root=sweep_root,
        output=output,
        source=source,
    )
    design = {
        "schema": "minicells.pcu-hybrid-reattachment-001.design.v3",
        "experiment": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "phase": "engineering_diagnostic",
        "causal_question": "can_frozen_Granite_consume_an_already_learned_PCU_mutation",
        "protocol_amendment": {
            "triggered_by_first_engineering_source_commit": FIRST_ENGINEERING_SOURCE_COMMIT,
            "reason": "v2_zero_state_gate_compared_native_fused_vs_cellularized_reduction_graphs",
            "thresholds_changed": False,
            "native_G0_comparison": "retained_as_non_gating_numerical_diagnostic",
            "strict_zero_state_gate": "PARENT_ZERO_DELTA_vs_CELL_OFF_same_cellular_graph",
        },
        "primary_arm": {
            "gpu": "cuda:0",
            "source_mutation": "PCU-OBJECTIVE-ALIGNMENT-001/ranking-only/L7/K64",
            "states": ["BASE", "PARENT_ZERO_DELTA", "CELL_ON", "CELL_OFF", "CELL_RESTORED"],
            "new_bridge": False,
            "new_router": False,
            "cell_alone_takeover_required": False,
        },
        "amplitude_sweep_arm": {
            "gpu": "cuda:1",
            "alpha_grid": list(ALPHA_SWEEP),
            "additional_training_after_exact_replay": False,
            "selection_rule": "highest_A_ranking_then_lowest_B_harm_then_lowest_alpha_among_joint_passes",
        },
        "thresholds": {
            "same_graph_equivalence_max_abs_logit_diff": EQUIVALENCE_MAX_ABS_LOGIT_DIFF,
            "restoration_max_abs_logit_diff": RESTORATION_MAX_ABS_LOGIT_DIFF,
            "association_floor": ASSOCIATION_FLOOR,
            "minimum_causal_ranking_gain": MIN_CAUSAL_RANKING_GAIN,
            "maximum_B_control_answer_nll_increase": MAX_CONTROL_ANSWER_NLL_INCREASE,
        },
        "dual_gpu_execution": {
            "required": True,
            "process_isolation": True,
            "cuda:0": "primary_causal_reattachment",
            "cuda:1": "amplitude_sweep",
        },
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-hybrid-reattachment-001.run-identity.v3",
        "experiment": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "phase": "engineering_diagnostic",
        "run_id": output.name,
        "source": source,
        "dual_gpu_execution_required": True,
        "worker_devices": {"primary": "cuda:0", "sweep": "cuda:1"},
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    })
    print(json.dumps({
        "experiment": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": result["status"],
        "valid_run": result["valid_run"],
        "primary_causal_effect": result["primary_causal_effect"],
        "selected_locality_compatible_point": result["selected_locality_compatible_point"],
        "visualizations": result.get("visualizations", []),
        "output": str(output),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=("primary", "sweep"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    args = parser.parse_args()
    if args.worker:
        if not args.arm:
            parser.error("--worker requires --arm")
        return worker_mode(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
