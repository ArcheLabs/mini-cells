from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from minicells.functional_cellization import freeze_foundation_
from minicells.hybrid_clm import HybridCellOverlay, load_cell_artifact

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[3]
SEQUENCE_ROOT = ROOT / "scripts" / "research" / "jam_knowledge_mutation_001"
CONVERSION_ROOT = ROOT / "scripts" / "research" / "clm_conversion_kill_test_001"
LOCAL_ROOT = Path(__file__).resolve().parent
for path in (SEQUENCE_ROOT, CONVERSION_ROOT, LOCAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sequence as seq  # noqa: E402
from dataset import CANDIDATE_CODES, demo_facts, evaluation_rows, update_rows  # noqa: E402
from semantic_choice import candidate_choice_metrics  # noqa: E402

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
MODEL_REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
PROTOCOL: dict[str, Any] = {
    "sequence_task": {"prompt_template": PROMPT_TEMPLATE, "max_sequence_tokens": 96},
    "evaluation": {"batch_size": 8},
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_choice(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    device: str,
    overlay: HybridCellOverlay,
) -> dict[str, Any]:
    return candidate_choice_metrics(
        model,
        tokenizer,
        rows,
        CANDIDATE_CODES,
        protocol=PROTOCOL,
        device=device,
        overlay=overlay,
        sequence_module=seq,
    )


def run(*, result_dir: Path, device: str) -> dict[str, Any]:
    result = _load_json(result_dir / "result.json")
    manifest = _load_json(result_dir / "manifest.json")
    if manifest["foundation_model_id"] != MODEL_ID or manifest["foundation_revision"] != MODEL_REVISION:
        raise RuntimeError("manifest foundation identity mismatch")

    import transformers

    transformers.logging.set_verbosity_error()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
    ).to(device)
    freeze_foundation_(model)

    overlay = HybridCellOverlay(
        hidden_size=int(model.config.hidden_size),
        read_layer_index=18,
        write_layer_indices=(20, 22),
        max_cells=max(64, len(manifest["cells"]) + 8),
        rank=16,
        gate_threshold=0.8,
        gate_temperature=1.0,
        seed=0,
    ).to(device=device, dtype=torch.float32)

    loaded: list[str] = []
    child_artifact = None
    for entry in manifest["cells"]:
        cell_id = str(entry["cell_id"])
        artifact = load_cell_artifact(result_dir / "cells" / f"{cell_id}.pt")
        if artifact.digest() != str(entry["digest"]):
            raise RuntimeError(f"artifact digest mismatch for {cell_id}")
        overlay.apply_artifact_(artifact, commit=True)
        loaded.append(cell_id)
        if artifact.parent_id is not None:
            child_artifact = artifact

    fact_count = int(result["committed_facts"])
    facts = demo_facts(fact_count)
    retained_rows = [row for fact in facts for row in evaluation_rows(fact)]
    retention = _candidate_choice(model, tokenizer, retained_rows, device, overlay)

    child_old_accuracy = None
    child_new_accuracy = None
    if child_artifact is not None:
        parent_index = int(child_artifact.parent_id.split("-")[-1])
        parent = next(fact for fact in facts if fact.index == parent_index)
        new_value = CANDIDATE_CODES[(CANDIDATE_CODES.index(parent.value) + 3) % len(CANDIDATE_CODES)]
        new_rows = update_rows(parent, new_value, "v2")["evaluation"]
        child_old_accuracy = float(
            _candidate_choice(model, tokenizer, evaluation_rows(parent), device, overlay)[
                "strict_choice_accuracy"
            ]
        )
        child_new_accuracy = float(
            _candidate_choice(model, tokenizer, new_rows, device, overlay)["strict_choice_accuracy"]
        )

    report = {
        "experiment": "GRANITE_HYBRID_CLM_V0_1_RELOAD",
        "foundation": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "trainable": False},
        "loaded_cells": loaded,
        "loaded_cell_count": len(loaded),
        "retention_choice_accuracy": float(retention["strict_choice_accuracy"]),
        "contextual_child_old_choice_accuracy": child_old_accuracy,
        "contextual_child_new_choice_accuracy": child_new_accuracy,
    }
    report["status"] = (
        "GRANITE_HYBRID_CLM_V01_RELOAD_VERIFIED"
        if report["retention_choice_accuracy"] >= 0.98
        and (child_artifact is None or (child_old_accuracy == 1.0 and child_new_accuracy == 1.0))
        else "GRANITE_HYBRID_CLM_V01_RELOAD_FAILED"
    )
    _write_json(result_dir / "reload_verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a fresh reload of Granite Hybrid CLM v0.1")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results" / "granite-hybrid-clm-v0.1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(result_dir=args.result_dir, device=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "GRANITE_HYBRID_CLM_V01_RELOAD_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
