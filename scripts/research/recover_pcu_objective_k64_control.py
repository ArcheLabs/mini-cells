#!/usr/bin/env python3
"""Recover the exact L7/K64 CE paired control for PCU-OBJECTIVE-ALIGNMENT-001.

This is used only when the completed locality-width sweep lived in an ephemeral
Kaggle session and was not published. It reruns exactly one condition: L7/K64
with the registered CE objective. K16/K32 are not rerun.
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
    _load_layer_baseline,
    _run_one_width,
)
from minicells.pcu_kill_001.locality_width_isolated import (
    SCIENTIFIC_SOURCE_COMMIT,
    SCIENTIFIC_SOURCE_TREE,
)

EXPECTED_DIRECT_ACCURACY = 0.265625
DEFAULT_OUT = Path(
    "artifacts/research/pcu-objective-alignment-001/engineering/26090501-l7-k64-ranking"
) / "PAIRED_CE_K64.json"


def scientific_source() -> dict[str, object]:
    return {
        "source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "source_tree": SCIENTIFIC_SOURCE_TREE,
        "source_dirty": False,
        "source_ref": "codex/pcu-composability-kill-001",
        "status_porcelain": "",
        "execution_repair_separated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=ENGINEERING_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", type=Path, default=LAYER_BASELINE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.seed != ENGINEERING_SEED:
        raise ValueError("paired K64 control is engineering-seed only")
    parsed = torch.device(args.device)
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("paired K64 control requires explicit CUDA device")
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
        "recovery_reason": "ephemeral locality-width artifacts unavailable; exact K64 CE condition rerun as paired control",
    }
    write_json(args.out, payload)
    print(json.dumps({
        "recovered": True,
        "direct_accuracy": accuracy,
        "selected_k": len(selected),
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
