"""Engineering-only locality-width diagnostic for PCU local Cell mutation.

PCU-LOCALITY-WIDTH-001 holds the successful layer-placement probe conditions
fixed at decoder layer 7 and changes only the number of selected task-gradient
Cells. The published L7/K=8 result is reused. K=16 and K=32 are run in parallel;
K=64 is run only when neither primary width reaches the inherited 80% direct
capability floor.

This is a diagnostic continuation of PCU-KILL-001. It does not alter the frozen
scientific protocol and never consumes formal seeds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .layer_placement import (
    CALIBRATION_BATCH_SIZE,
    CALIBRATION_ROWS,
    DIRECT_CAPABILITY_FLOOR,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    MAX_TRAINING_TOKENS,
    BATCH_SIZE,
    _assert_only_selected_deltas_trainable,
    _train_full_model_branch,
    _validate_foundation_manifest,
    full_model_task_conditioned_allocation,
)
from .model import load_granite, target_module
from .synthetic import audit_dataset, generate_world
from .task import build_task_sequences, validate_answer_only_labels
from .training import Allocation, BranchTrainingConfig


EXPERIMENT_ID = "PCU-LOCALITY-WIDTH-001"
ENGINEERING_SEED = 26090501
TARGET_LAYER = 7
BASELINE_K = 8
PRIMARY_WIDTHS = (16, 32)
FALLBACK_WIDTH = 64
LAYER_BASELINE_ROOT = Path(
    "artifacts/research/pcu-layer-placement-001/engineering/26090501-layer-only"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-locality-width-001/engineering/26090501-l7-width"
)


def _gradient_mass_at(allocation: Allocation, k: int) -> float:
    ordered = sorted(allocation.scores, key=lambda key: (-allocation.scores[key], key))
    total = sum(float(value) for value in allocation.scores.values())
    if total <= 0.0:
        return 0.0
    return sum(float(allocation.scores[key]) for key in ordered[: int(k)]) / total


def should_run_fallback(primary_results: Mapping[int, Mapping[str, Any]]) -> bool:
    """K=64 is needed only when neither K=16 nor K=32 reaches the frozen floor."""
    required = set(PRIMARY_WIDTHS)
    if set(int(key) for key in primary_results) != required:
        raise ValueError("fallback decision requires complete K=16 and K=32 results")
    return max(float(primary_results[k]["direct_accuracy"]) for k in PRIMARY_WIDTHS) < DIRECT_CAPABILITY_FLOOR


def _load_layer_baseline(root: Path) -> dict[str, Any]:
    decision = json.loads((root / "DECISION.json").read_text(encoding="utf-8"))
    design = json.loads((root / "DESIGN.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    layer = json.loads((root / f"LAYER_{TARGET_LAYER:02d}.json").read_text(encoding="utf-8"))

    if decision.get("status") != "LAYER_PLACEMENT_DID_NOT_RESCUE":
        raise RuntimeError("locality-width baseline is not the published layer-placement negative")
    if decision.get("valid_run") is not True or decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("layer-placement baseline is not a valid pre-formal engineering run")
    if identity.get("source", {}).get("source_dirty") is not False:
        raise RuntimeError("layer-placement baseline source is dirty")
    if int(layer.get("layer", -1)) != TARGET_LAYER:
        raise RuntimeError("published layer-placement baseline does not contain L7")
    if int(layer.get("allocation", {}).get("selected_k", -1)) != BASELINE_K:
        raise RuntimeError("published L7 baseline is not K=8")
    if float(layer.get("training", {}).get("learning_rate", -1.0)) != LEARNING_RATE:
        raise RuntimeError("published L7 baseline learning rate changed")
    if int(layer.get("training", {}).get("training_steps", -1)) != MAX_OPTIMIZER_STEPS:
        raise RuntimeError("published L7 baseline step budget changed")
    if int(layer.get("training", {}).get("batch_size", -1)) != BATCH_SIZE:
        raise RuntimeError("published L7 baseline batch size changed")
    fixed = design.get("fixed", {})
    expected_fixed = {
        "loss": "answer-token-causal-cross-entropy",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "selected_k": BASELINE_K,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "max_training_tokens": MAX_TRAINING_TOKENS,
        "batch_size": BATCH_SIZE,
        "routing": "inherited_parent_router",
        "evaluation": "A_eval_greedy_exact",
        "capability_floor": DIRECT_CAPABILITY_FLOOR,
    }
    for key, value in expected_fixed.items():
        if fixed.get(key) != value:
            raise RuntimeError(f"layer-placement baseline fixed condition changed: {key}")

    selected = tuple(str(value) for value in layer["allocation"]["selected"])
    if len(selected) != BASELINE_K:
        raise RuntimeError("published L7 baseline selected Cell count is not 8")
    foundation = layer.get("foundation")
    if not isinstance(foundation, Mapping):
        raise RuntimeError("published L7 baseline lacks foundation identity")
    return {
        "layer": TARGET_LAYER,
        "target_path": str(layer["target_path"]),
        "foundation": dict(foundation),
        "dataset_manifest_sha256": str(layer["dataset_manifest_sha256"]),
        "selected": selected,
        "direct_accuracy": float(layer["direct_accuracy"]),
        "effective_count": float(layer["allocation"]["effective_count"]),
        "topk_mass": dict(layer["allocation"].get("topk_mass", {})),
        "training": dict(layer["training"]),
        "baseline_source": identity.get("source", {}),
        "artifact_source": str(root),
    }


def _result_identity(
    *,
    width: int,
    baseline: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "seed": ENGINEERING_SEED,
        "source_commit": source.get("source_commit"),
        "source_tree": source.get("source_tree"),
        "foundation_tensor_sha256": baseline["foundation"].get("foundation_tensor_sha256"),
        "model_revision": baseline["foundation"].get("model_revision"),
        "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
        "target_layer": TARGET_LAYER,
        "target_path": baseline["target_path"],
        "selected_k": int(width),
        "learning_rate": LEARNING_RATE,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "max_training_tokens": MAX_TRAINING_TOKENS,
        "batch_size": BATCH_SIZE,
        "calibration_rows": CALIBRATION_ROWS,
        "calibration_batch_size": CALIBRATION_BATCH_SIZE,
    }


def _load_resumable_width_result(
    path: Path,
    *,
    width: int,
    baseline: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _result_identity(width=width, baseline=baseline, source=source)
    if payload.get("schema") != "minicells.pcu-locality-width-001.width-result.v1":
        raise RuntimeError(f"stale locality-width result schema: {path}")
    if payload.get("identity") != expected:
        raise RuntimeError(f"stale locality-width result identity: {path}")
    if int(payload.get("training", {}).get("training_steps", -1)) != MAX_OPTIMIZER_STEPS:
        raise RuntimeError(f"incomplete locality-width result: {path}")
    return payload


def _run_one_width(
    *,
    width: int,
    device: str,
    baseline: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if int(width) <= BASELINE_K or int(width) > 128:
        raise ValueError(f"unsupported locality width: {width}")
    print(f"[pcu-locality-width] loading L{TARGET_LAYER}/K{width} on {device}", flush=True)
    tokenizer, model, manifest = load_granite(
        str(baseline["foundation"]["model_repo"]),
        revision=str(baseline["foundation"]["model_revision"]),
        device=device,
    )
    try:
        _validate_foundation_manifest(manifest, baseline["foundation"])
        model.requires_grad_(False)
        block = target_module(model, str(baseline["target_path"]))
        projection = extract_expert_projections(block.experts, 0)
        cellular_experts = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
        model.requires_grad_(False)
        cellular_experts.requires_grad_(False)

        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed:
            raise RuntimeError(f"locality-width dataset audit failed: {audit.errors}")
        if world.manifest_sha256() != baseline["dataset_manifest_sha256"]:
            raise RuntimeError("locality-width dataset identity differs from published L7 baseline")
        train = build_task_sequences(tokenizer, world.splits["A_train"], "A_train", max_length=128)
        validate_answer_only_labels(train)

        print(f"[pcu-locality-width] allocating L{TARGET_LAYER}/K{width}", flush=True)
        allocation = full_model_task_conditioned_allocation(
            model,
            block,
            cellular_experts,
            train,
            layer=TARGET_LAYER,
            calibration_rows=CALIBRATION_ROWS,
            calibration_batch_size=CALIBRATION_BATCH_SIZE,
            device=device,
        )
        selected = tuple(str(value) for value in allocation.selected[: int(width)])
        if selected[:BASELINE_K] != tuple(baseline["selected"]):
            raise RuntimeError(
                "LOCALITY_ALLOCATION_DRIFT: recomputed L7 top-8 differs from published K=8 baseline"
            )
        torch.manual_seed(ENGINEERING_SEED)
        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        print(f"[pcu-locality-width] training L{TARGET_LAYER}/K{width}", flush=True)
        runtime, training = _train_full_model_branch(
            model,
            block,
            cellular_experts,
            train,
            selected,
            layer=TARGET_LAYER,
            device=device,
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime)
        print(f"[pcu-locality-width] evaluating L{TARGET_LAYER}/K{width}", flush=True)
        evaluation = evaluate_samples(
            model,
            tokenizer,
            world.splits["A_eval"],
            split="A_eval",
            device=device,
            max_new_tokens=16,
            batch_size=16,
        )
        return {
            "schema": "minicells.pcu-locality-width-001.width-result.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "identity": _result_identity(width=width, baseline=baseline, source=source),
            "device": device,
            "allocation": {
                "method": "task-conditioned-gradient-l2-per-parameter",
                "calibration_split": "A_train",
                "calibration_sample_rule": "first_64_samples",
                "calibration_execution": f"weighted_microbatch_{CALIBRATION_BATCH_SIZE}",
                "selected_k": int(width),
                "selected": list(selected),
                "baseline_prefix_match": True,
                "gradient_mass_at_k": _gradient_mass_at(allocation, int(width)),
                "effective_count": float(allocation.effective_count),
            },
            "training": training,
            "evaluation": evaluation.to_dict(),
            "direct_accuracy": float(evaluation.exact),
            "capability_floor": DIRECT_CAPABILITY_FLOOR,
            "passes": bool(evaluation.exact >= DIRECT_CAPABILITY_FLOOR),
            "scientific_evidence": False,
            "formal_execution_not_started": True,
        }
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _assert_nested_widths(results: Mapping[int, Mapping[str, Any]], baseline: Mapping[str, Any]) -> None:
    ordered_widths = sorted(int(value) for value in results)
    previous = tuple(baseline["selected"])
    previous_k = BASELINE_K
    for width in ordered_widths:
        selected = tuple(str(value) for value in results[width]["allocation"]["selected"])
        if selected[:previous_k] != previous:
            raise RuntimeError(
                f"LOCALITY_ALLOCATION_DRIFT: K={width} is not a strict prefix extension of K={previous_k}"
            )
        previous = selected
        previous_k = width


def _classify(results: Mapping[int, Mapping[str, Any]], baseline: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    rows = {
        BASELINE_K: {
            "selected_k": BASELINE_K,
            "direct_accuracy": float(baseline["direct_accuracy"]),
            "gradient_mass_at_k": float(baseline["topk_mass"].get(str(BASELINE_K), 0.0)),
            "effective_count": float(baseline["effective_count"]),
            "source": "published_L7_K8_baseline",
        }
    }
    for width in sorted(results):
        rows[int(width)] = {
            "selected_k": int(width),
            "direct_accuracy": float(results[width]["direct_accuracy"]),
            "gradient_mass_at_k": float(results[width]["allocation"]["gradient_mass_at_k"]),
            "effective_count": float(results[width]["allocation"]["effective_count"]),
            "final_loss": float(results[width]["training"]["final_loss"]),
            "source": "new_width_probe",
        }
    best_k, best = max(rows.items(), key=lambda item: float(item[1]["direct_accuracy"]))
    rescued = best_k != BASELINE_K and float(best["direct_accuracy"]) >= DIRECT_CAPABILITY_FLOOR
    improved = best_k != BASELINE_K and float(best["direct_accuracy"]) > float(baseline["direct_accuracy"])
    if rescued:
        status = "LOCALITY_WIDTH_RESCUES_LOCAL_CELL_MUTATION"
    elif improved:
        status = "LOCALITY_WIDTH_IMPROVES_BUT_DOES_NOT_RESCUE"
    else:
        status = "LOCALITY_WIDTH_DID_NOT_IMPROVE"
    return status, {
        "rows": {str(key): value for key, value in rows.items()},
        "best": {"selected_k": int(best_k), **best},
        "rescued": rescued,
        "improved": improved,
    }


def run_locality_width_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    baseline_root: Path = LAYER_BASELINE_ROOT,
    devices: tuple[str, str] = ("cuda:0", "cuda:1"),
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    """Run K=16/32 at L7 in parallel, then K=64 only if the primary widths fail."""
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-LOCALITY-WIDTH-001 is engineering-seed only")
    if len(devices) != 2 or devices[0] == devices[1]:
        raise ValueError("two distinct CUDA devices are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("PCU-LOCALITY-WIDTH-001 requires two CUDA devices")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load_layer_baseline(Path(baseline_root))
    source = git_provenance(Path(__file__).resolve().parents[3])
    if source.get("source_dirty") is not False:
        raise RuntimeError("locality-width diagnostic requires a clean source tree")

    design = {
        "schema": "minicells.pcu-locality-width-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_variable": "selected_cell_width_k_only",
        "fixed": {
            "task": "A_only_U_to_V",
            "target_layer": TARGET_LAYER,
            "target_path": baseline["target_path"],
            "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
            "loss": "answer-token-causal-cross-entropy",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "batch_size": BATCH_SIZE,
            "allocation": "task-conditioned-gradient-l2-per-parameter:first_64_A_train",
            "allocation_execution": f"weighted_microbatch_{CALIBRATION_BATCH_SIZE}",
            "routing": "inherited_parent_router",
            "evaluation": "A_eval_greedy_exact",
            "capability_floor": DIRECT_CAPABILITY_FLOOR,
        },
        "widths": {
            "baseline_reused": BASELINE_K,
            "primary_parallel": list(PRIMARY_WIDTHS),
            "fallback": FALLBACK_WIDTH,
            "fallback_rule": "run only if max(K16,K32) direct accuracy < 0.80",
        },
        "baseline": {
            "artifact": baseline["artifact_source"],
            "selected_k": BASELINE_K,
            "direct_accuracy": baseline["direct_accuracy"],
            "selected": list(baseline["selected"]),
            "effective_count": baseline["effective_count"],
            "gradient_mass_at_k": baseline["topk_mass"].get(str(BASELINE_K)),
        },
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-locality-width-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output.name,
        "source": source,
        "baseline_source": baseline["baseline_source"],
        "formal_execution_not_started": True,
    })

    results: dict[int, dict[str, Any]] = {}
    pending: dict[Any, int] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for width, device in zip(PRIMARY_WIDTHS, devices):
            path = output / f"WIDTH_{width:03d}.json"
            resumed = _load_resumable_width_result(
                path,
                width=width,
                baseline=baseline,
                source=source,
            )
            if resumed is not None:
                print(f"[pcu-locality-width] resume K={width} from {path}", flush=True)
                results[width] = resumed
                continue
            future = pool.submit(
                _run_one_width,
                width=width,
                device=device,
                baseline=baseline,
                source=source,
            )
            pending[future] = width
        for future in as_completed(pending):
            width = pending[future]
            results[width] = future.result()
            write_json(output / f"WIDTH_{width:03d}.json", results[width])

    primary = {width: results[width] for width in PRIMARY_WIDTHS}
    fallback_required = should_run_fallback(primary)
    if fallback_required:
        width = FALLBACK_WIDTH
        path = output / f"WIDTH_{width:03d}.json"
        resumed = _load_resumable_width_result(
            path,
            width=width,
            baseline=baseline,
            source=source,
        )
        if resumed is None:
            results[width] = _run_one_width(
                width=width,
                device=devices[0],
                baseline=baseline,
                source=source,
            )
            write_json(path, results[width])
        else:
            print(f"[pcu-locality-width] resume K={width} from {path}", flush=True)
            results[width] = resumed

    _assert_nested_widths(results, baseline)
    status, comparison = _classify(results, baseline)
    decision = {
        "schema": "minicells.pcu-locality-width-001.decision.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "fallback_k64_required": fallback_required,
        "comparison": comparison["rows"],
        "best": comparison["best"],
        "rescued": comparison["rescued"],
        "improved": comparison["improved"],
        "capability_floor": DIRECT_CAPABILITY_FLOOR,
        "interpretation": (
            "increasing only L7 mutation width reached the inherited direct-capability floor"
            if comparison["rescued"]
            else "increasing only L7 mutation width did not reach the inherited direct-capability floor"
        ),
        "source": source,
    }
    write_json(output / "DECISION.json", decision)
    return decision


__all__ = [
    "EXPERIMENT_ID",
    "ENGINEERING_SEED",
    "TARGET_LAYER",
    "BASELINE_K",
    "PRIMARY_WIDTHS",
    "FALLBACK_WIDTH",
    "should_run_fallback",
    "run_locality_width_diagnostic",
]
