#!/usr/bin/env python3
"""Recover the exact L7/K64 CE paired control for PCU-OBJECTIVE-ALIGNMENT-001.

This is used only when the completed locality-width sweep lived in an ephemeral
Kaggle session and was not published. It reruns exactly one condition: L7/K64
with the registered CE objective. K16/K32 are not rerun.

The recovery writes two things:
1. PAIRED_CE_K64.json inside the final objective output directory. This is the
   real paired-control evidence that will be published with the final result.
2. A local-only compatibility envelope at the locality baseline path containing
   RUN_IDENTITY/DESIGN/DECISION/WIDTH_064. This lets the unchanged final
   objective implementation consume the recovered K64 condition. The envelope
   is explicitly marked recovery_mode=paired_ce_k64_only and is never published
   as a replacement for the historical locality-width sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minicells.pcu_kill_001.governance import write_json
from minicells.pcu_kill_001.locality_width import (
    ENGINEERING_SEED,
    LAYER_BASELINE_ROOT,
    DEFAULT_OUTPUT as LOCALITY_DEFAULT_OUTPUT,
    _load_layer_baseline,
    _run_one_width,
)
from minicells.pcu_kill_001.locality_width_isolated import (
    SCIENTIFIC_SOURCE_COMMIT,
    SCIENTIFIC_SOURCE_TREE,
)

EXPECTED_DIRECT_ACCURACY = 0.265625
DEFAULT_FINAL_OUT = Path(
    "artifacts/research/pcu-objective-alignment-001/engineering/26090501-l7-k64-ranking"
)
DEFAULT_CONTROL_OUT = DEFAULT_FINAL_OUT / "PAIRED_CE_K64.json"


def scientific_source() -> dict[str, object]:
    return {
        "source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "source_tree": SCIENTIFIC_SOURCE_TREE,
        "source_dirty": False,
        "source_ref": "codex/pcu-composability-kill-001",
        "status_porcelain": "",
        "execution_repair_separated": True,
    }


def _write_local_compatibility_envelope(
    root: Path,
    *,
    baseline: dict,
    width_result: dict,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    source = scientific_source()
    write_json(root / "WIDTH_064.json", width_result)
    write_json(root / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-locality-width-001.run-identity.v1",
        "experiment": "PCU-LOCALITY-WIDTH-001",
        "phase": "engineering_diagnostic_recovery_envelope",
        "seed": ENGINEERING_SEED,
        "run_id": root.name,
        "source": source,
        "baseline_source": baseline.get("baseline_source", {}),
        "recovery_mode": "paired_ce_k64_only",
        "historical_sweep_not_reconstructed": True,
        "formal_execution_not_started": True,
    })
    write_json(root / "DESIGN.json", {
        "schema": "minicells.pcu-locality-width-001.design.v1",
        "experiment": "PCU-LOCALITY-WIDTH-001",
        "phase": "engineering_diagnostic_recovery_envelope",
        "seed": ENGINEERING_SEED,
        "causal_variable": "selected_cell_width_k_only",
        "fixed": {
            "task": "A_only_U_to_V",
            "target_layer": 7,
            "target_path": baseline["target_path"],
            "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
            "loss": "answer-token-causal-cross-entropy",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "max_optimizer_steps": 128,
            "max_training_tokens": 500_000,
            "batch_size": 8,
            "routing": "inherited_parent_router",
            "evaluation": "A_eval_greedy_exact",
            "capability_floor": 0.80,
        },
        "baseline": {
            "artifact": str(LAYER_BASELINE_ROOT),
            "selected_k": 8,
            "direct_accuracy": baseline["direct_accuracy"],
            "selected": list(baseline["selected"]),
            "effective_count": baseline["effective_count"],
            "gradient_mass_at_k": baseline["topk_mass"].get("8"),
        },
        "recovery_mode": "paired_ce_k64_only",
        "historical_sweep_not_reconstructed": True,
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    })
    # The unchanged objective loader requires the historical non-rescue status.
    # We retain it only as a compatibility field while explicitly declaring that
    # this envelope contains no reconstructed K16/K32 sweep evidence.
    write_json(root / "DECISION.json", {
        "schema": "minicells.pcu-locality-width-001.decision.v1",
        "experiment": "PCU-LOCALITY-WIDTH-001",
        "phase": "engineering_diagnostic_recovery_envelope",
        "status": "LOCALITY_WIDTH_IMPROVES_BUT_DOES_NOT_RESCUE",
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "recovery_mode": "paired_ce_k64_only",
        "historical_sweep_not_reconstructed": True,
        "paired_control_direct_accuracy": float(width_result["direct_accuracy"]),
        "scientific_source": source,
        "interpretation": "local compatibility envelope only; final objective comparison uses the fresh paired K64 CE control",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", type=Path, default=LAYER_BASELINE_ROOT)
    parser.add_argument("--locality-root", type=Path, default=LOCALITY_DEFAULT_OUTPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONTROL_OUT)
    args = parser.parse_args()
    if args.seed != ENGINEERING_SEED:
        raise ValueError("paired K64 control is engineering-seed only")
    parsed = torch.device(args.device)
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("paired K64 control requires explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {args.device}")
    torch.cuda.set_device(parsed.index)

    baseline = _load_layer_baseline(args.baseline)
    result = _run_one_width(
        width=64,
        device=args.device,
        baseline=baseline,
        source=scientific_source(),
    )
    accuracy = float(result["direct_accuracy"])
    if abs(accuracy - EXPECTED_DIRECT_ACCURACY) > 1e-12:
        raise RuntimeError(
            f"PAIRED_CE_K64_REPRODUCTION_MISMATCH: expected {EXPECTED_DIRECT_ACCURACY}, got {accuracy}"
        )
    selected = list(result["allocation"]["selected"])
    if len(selected) != 64:
        raise RuntimeError("paired K64 control did not select exactly 64 Cells")
    if result.get("allocation", {}).get("baseline_prefix_match") is not True:
        raise RuntimeError("paired K64 control allocation does not preserve published L7/K8 prefix")

    payload = {
        "schema": "minicells.pcu-objective-alignment-001.paired-ce-k64.v1",
        "experiment": "PCU-OBJECTIVE-ALIGNMENT-001",
        "phase": "engineering_diagnostic_control",
        "seed": ENGINEERING_SEED,
        "control_objective": "answer-token-causal-cross-entropy",
        "target_layer": 7,
        "selected_k": 64,
        "selected_cells": selected,
        "direct_accuracy": accuracy,
        "training": result["training"],
        "allocation": result["allocation"],
        "identity": result["identity"],
        "foundation": baseline["foundation"],
        "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
        "target_path": baseline["target_path"],
        "scientific_source": scientific_source(),
        "formal_execution_not_started": True,
        "scientific_evidence": False,
        "recovery_mode": "paired_ce_k64_only",
        "historical_sweep_not_reconstructed": True,
        "recovery_reason": "ephemeral locality-width artifacts unavailable; exact K64 CE condition rerun as paired control",
    }
    write_json(args.out, payload)
    _write_local_compatibility_envelope(
        args.locality_root,
        baseline=baseline,
        width_result=result,
    )
    print(json.dumps({
        "recovered": True,
        "direct_accuracy": accuracy,
        "selected_k": len(selected),
        "paired_control": str(args.out),
        "local_compatibility_envelope": str(args.locality_root),
        "historical_sweep_reconstructed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
