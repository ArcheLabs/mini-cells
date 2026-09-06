"""Depth sweep for sparse cross-layer Cell paths.

PCU-SPARSE-PATH-DEPTH-001 tests whether distributing a fixed late-path mutation
budget across 3, 4, or 5 decoder layers improves native autoregressive readout.

Every topology starts from the exact published L7/K64 hybrid association state,
which is replayed and frozen. Added layers are nested and chosen deterministically
from the model's actual MoE layers. On the canonical 24-layer Granite target they
resolve to:

    depth3: L7 -> L15 -> L23
    depth4: L7 -> L11 -> L15 -> L23
    depth5: L7 -> L11 -> L15 -> L19 -> L23

Crucially, depth is not allowed to buy more Cell capacity or more optimization:

    total added Cell budget = K32 for every topology
    final L23 readout       = K16, 128 optimizer steps
    all transport layers    = K16 total, 128 optimizer steps total

The transport K/step budgets are split as evenly as possible across transport
layers. Each added layer is allocated under the already-frozen preceding path
state using the existing first-64 A_train answer-token-CE gradient geometry,
trained with answer-token CE only, then frozen before the next stage.

Engineering-only evidence. Formal PCU-KILL-001 seeds are never consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .cross_layer_readout import (
    HYBRID_BASELINE_ROOT,
    READOUT_BASELINE_ROOT,
    PublishedBaselines,
    _freeze_l7_runtime,
    _gradient_mass_at_k,
    _load_published_baselines,
)
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .hybrid_objective import OBJECTIVE_BASELINE_ROOT, _load_baselines, _train_hybrid_branch
from .layer_placement import (
    BATCH_SIZE,
    CALIBRATION_BATCH_SIZE,
    CALIBRATION_ROWS,
    DIRECT_CAPABILITY_FLOOR,
    LEARNING_RATE,
    MAX_TRAINING_TOKENS,
    _assert_only_selected_deltas_trainable,
    _train_full_model_branch,
    _validate_foundation_manifest,
    discover_moe_layers,
    full_model_task_conditioned_allocation,
)
from .locality_width import ENGINEERING_SEED
from .model import load_granite, target_module
from .objective_alignment import ASSOCIATION_FLOOR, evaluate_candidate_ranking
from .readout_localization import gold_prefix_token_readout
from .synthetic import audit_dataset, generate_world
from .task import build_task_sequences, validate_answer_only_labels
from .training import BranchTrainingConfig


EXPERIMENT_ID = "PCU-SPARSE-PATH-DEPTH-001"
ASSOCIATION_LAYER = 7
ASSOCIATION_K = 64
READOUT_LAYER = 23
READOUT_K = 16
READOUT_STEPS = 128
TRANSPORT_K_TOTAL = 16
TRANSPORT_STEPS_TOTAL = 128
TOTAL_ADDED_K = READOUT_K + TRANSPORT_K_TOTAL
TOTAL_ADDED_STEPS = READOUT_STEPS + TRANSPORT_STEPS_TOTAL
DEPTHS = (3, 4, 5)
READOUT_OBJECTIVE = "answer-token-causal-cross-entropy"
PREVIOUS_CROSS_LAYER_ROOT = Path(
    "artifacts/research/pcu-cross-layer-readout-001/engineering/26090501-l7k64-plus-l23k16"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-sparse-path-depth-001/engineering/26090501-depth3-4-5"
)
EXPECTED_PREVIOUS_STATUS = "CROSS_LAYER_READOUT_IMPROVES_BUT_DOES_NOT_RESCUE"
EXPECTED_PREVIOUS_DIRECT = 0.15625
EXPECTED_PREVIOUS_RANKING = 0.828125


@dataclass(frozen=True)
class TopologySpec:
    depth: int
    layers: tuple[int, ...]
    transport_layers: tuple[int, ...]
    transport_k: tuple[int, ...]
    transport_steps: tuple[int, ...]
    readout_k: int = READOUT_K
    readout_steps: int = READOUT_STEPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": int(self.depth),
            "layers": list(self.layers),
            "transport_layers": list(self.transport_layers),
            "transport_k": list(self.transport_k),
            "transport_steps": list(self.transport_steps),
            "readout_layer": READOUT_LAYER,
            "readout_k": int(self.readout_k),
            "readout_steps": int(self.readout_steps),
            "total_added_k": int(sum(self.transport_k) + self.readout_k),
            "total_added_steps": int(sum(self.transport_steps) + self.readout_steps),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _balanced_integer_split(total: int, parts: int) -> tuple[int, ...]:
    if total <= 0 or parts <= 0 or total < parts:
        raise ValueError("invalid balanced integer split")
    base, remainder = divmod(int(total), int(parts))
    return tuple(base + (1 if index < remainder else 0) for index in range(parts))


def _nearest_available(values: Sequence[int], target: float, forbidden: set[int]) -> int:
    candidates = [int(value) for value in values if int(value) not in forbidden]
    if not candidates:
        raise RuntimeError("no available MoE layer remains for sparse path topology")
    return min(candidates, key=lambda value: (abs(float(value) - float(target)), value))


def choose_nested_topologies(available_layers: Sequence[int]) -> dict[int, TopologySpec]:
    """Choose nested 3/4/5-layer paths from actual MoE layers."""
    available = tuple(sorted({int(value) for value in available_layers}))
    if ASSOCIATION_LAYER not in available or READOUT_LAYER not in available:
        raise RuntimeError("sparse path requires MoE endpoints L7 and L23")
    between = [value for value in available if ASSOCIATION_LAYER < value < READOUT_LAYER]
    if len(between) < 3:
        raise RuntimeError("sparse path depth-5 requires at least three interior MoE layers")

    midpoint = _nearest_available(between, (ASSOCIATION_LAYER + READOUT_LAYER) / 2.0, set())
    lower = _nearest_available(between, (ASSOCIATION_LAYER + midpoint) / 2.0, {midpoint})
    upper = _nearest_available(between, (midpoint + READOUT_LAYER) / 2.0, {midpoint, lower})

    layer_sets = {
        3: (ASSOCIATION_LAYER, midpoint, READOUT_LAYER),
        4: (ASSOCIATION_LAYER, lower, midpoint, READOUT_LAYER),
        5: (ASSOCIATION_LAYER, lower, midpoint, upper, READOUT_LAYER),
    }
    if not set(layer_sets[3]).issubset(layer_sets[4]) or not set(layer_sets[4]).issubset(layer_sets[5]):
        raise RuntimeError("sparse path topology selection lost nesting")

    specs: dict[int, TopologySpec] = {}
    for depth, layers in layer_sets.items():
        transport = tuple(layer for layer in layers if layer not in (ASSOCIATION_LAYER, READOUT_LAYER))
        k_split = _balanced_integer_split(TRANSPORT_K_TOTAL, len(transport))
        step_split = _balanced_integer_split(TRANSPORT_STEPS_TOTAL, len(transport))
        spec = TopologySpec(depth, tuple(layers), transport, k_split, step_split)
        if sum(spec.transport_k) + spec.readout_k != TOTAL_ADDED_K:
            raise RuntimeError("topology Cell budget drift")
        if sum(spec.transport_steps) + spec.readout_steps != TOTAL_ADDED_STEPS:
            raise RuntimeError("topology optimizer-step budget drift")
        specs[depth] = spec
    return specs


def _load_previous_cross_layer(root: Path = PREVIOUS_CROSS_LAYER_ROOT) -> dict[str, Any]:
    root = Path(root)
    decision_path = root / "DECISION.json"
    if not decision_path.is_file():
        raise RuntimeError("sparse path depth sweep requires published cross-layer baseline")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != EXPECTED_PREVIOUS_STATUS:
        raise RuntimeError("previous cross-layer decision changed")
    if abs(float(decision.get("cross_layer_direct_accuracy", -1)) - EXPECTED_PREVIOUS_DIRECT) > 1e-12:
        raise RuntimeError("previous cross-layer direct accuracy changed")
    if abs(float(decision.get("cross_layer_ranking_accuracy", -1)) - EXPECTED_PREVIOUS_RANKING) > 1e-12:
        raise RuntimeError("previous cross-layer ranking accuracy changed")
    if decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("previous cross-layer evidence crossed formal boundary")
    return decision


def _evaluate_final(
    model: nn.Module,
    tokenizer: Any,
    eval_samples: Sequence[Any],
    candidate_universe: Sequence[str],
    *,
    device: str,
) -> dict[str, Any]:
    ranking = evaluate_candidate_ranking(model, tokenizer, eval_samples, candidate_universe, device=device)
    direct = evaluate_samples(
        model,
        tokenizer,
        eval_samples,
        split="A_eval",
        device=device,
        max_new_tokens=16,
        batch_size=16,
    )
    sequences = build_task_sequences(tokenizer, eval_samples, "A_eval", max_length=128)
    validate_answer_only_labels(sequences)
    gold = gold_prefix_token_readout(model, tokenizer, sequences, device=device, batch_size=8)
    return {
        "ranking_eval_accuracy": float(ranking.accuracy),
        "direct_accuracy": float(direct.exact),
        "first_token_top1_accuracy": float(gold["first_token_top1_accuracy"]),
        "later_token_top1_accuracy": float(gold["later_token_top1_accuracy"]),
        "all_token_top1_accuracy": float(gold["all_token_top1_accuracy"]),
        "sequence_all_tokens_top1_accuracy": float(gold["sequence_all_tokens_top1_accuracy"]),
        "ranking": ranking.to_dict(),
        "direct_evaluation": direct.to_dict(),
        "gold_prefix": gold,
    }


def _train_one_added_layer(
    model: nn.Module,
    tokenizer: Any,
    train_sequences: Any,
    *,
    layer: int,
    k: int,
    steps: int,
    device: str,
) -> dict[str, Any]:
    block = target_module(model, f"model.layers.{int(layer)}.block_sparse_moe")
    projection = extract_expert_projections(block.experts, 0)
    cellular = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
    model.requires_grad_(False)
    cellular.requires_grad_(False)

    allocation = full_model_task_conditioned_allocation(
        model,
        block,
        cellular,
        train_sequences,
        layer=int(layer),
        calibration_rows=CALIBRATION_ROWS,
        calibration_batch_size=CALIBRATION_BATCH_SIZE,
        device=device,
    )
    selected = tuple(allocation.selected[: int(k)])
    if len(selected) != int(k):
        raise RuntimeError(f"L{layer} allocation did not produce exact K{k}")
    mass = _gradient_mass_at_k(allocation, int(k))

    config = BranchTrainingConfig(
        optimizer="AdamW",
        learning_rate=LEARNING_RATE,
        max_optimizer_steps=int(steps),
        max_training_tokens=MAX_TRAINING_TOKENS,
        batch_size=BATCH_SIZE,
        seed=ENGINEERING_SEED,
    )
    runtime, training = _train_full_model_branch(
        model,
        block,
        cellular,
        train_sequences,
        selected,
        layer=int(layer),
        device=device,
        config=config,
    )
    _assert_only_selected_deltas_trainable(model, runtime)
    if tuple(training["selected_cells"]) != selected:
        raise RuntimeError(f"L{layer} selected Cell drift")
    runtime.requires_grad_(False)
    model.requires_grad_(False)
    unexpected = [name for name, value in model.named_parameters() if value.requires_grad]
    if unexpected:
        raise RuntimeError(f"L{layer} freeze failed: {unexpected[:8]}")
    return {
        "layer": int(layer),
        "selected_k": int(k),
        "optimizer_steps": int(steps),
        "selected_cells": list(selected),
        "gradient_mass_at_k": float(mass),
        "effective_count": float(allocation.effective_count),
        "allocation_method": "first64_A_train_answer_CE_gradient_under_preceding_frozen_path",
        "training": training,
    }


def run_topology(
    *,
    depth: int,
    output: Path,
    device: str,
    objective_root: Path = OBJECTIVE_BASELINE_ROOT,
    hybrid_root: Path = HYBRID_BASELINE_ROOT,
    readout_root: Path = READOUT_BASELINE_ROOT,
    previous_cross_layer_root: Path = PREVIOUS_CROSS_LAYER_ROOT,
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-SPARSE-PATH-DEPTH-001 is engineering-seed only")
    if int(depth) not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("sparse path topology requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)

    _load_previous_cross_layer(Path(previous_cross_layer_root))
    published: PublishedBaselines = _load_published_baselines(Path(hybrid_root), Path(readout_root))
    baseline = _load_baselines(Path(objective_root))
    if baseline.selected_cells != published.selected_l7:
        raise RuntimeError("sparse path L7 Cell identity drifted")
    if baseline.dataset_manifest_sha256 != published.dataset_manifest_sha256:
        raise RuntimeError("sparse path dataset identity drifted")

    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("sparse path topology requires clean source tree")

    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=device,
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        available = tuple(item.layer for item in discover_moe_layers(model))
        specs = choose_nested_topologies(available)
        spec = specs[int(depth)]

        model.requires_grad_(False)
        block7 = target_module(model, f"model.layers.{ASSOCIATION_LAYER}.block_sparse_moe")
        projection7 = extract_expert_projections(block7.experts, 0)
        cellular7 = patch_moe_block(block7, CellPartition(projection7.intermediate_size, 4))
        model.requires_grad_(False)
        cellular7.requires_grad_(False)

        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed or world.manifest_sha256() != published.dataset_manifest_sha256:
            raise RuntimeError("sparse path dataset audit/identity failure")
        train_samples = list(world.splits["A_train"])
        eval_samples = list(world.splits["A_eval"])
        candidate_universe = tuple(item.v for item in world.triples)
        train_sequences = build_task_sequences(tokenizer, train_samples, "A_train", max_length=128)
        validate_answer_only_labels(train_sequences)

        l7_config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=128,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        print(f"[pcu-path-depth{depth}] replaying exact L7/K64 hybrid on {device}", flush=True)
        runtime7, l7_training = _train_hybrid_branch(
            model,
            block7,
            cellular7,
            tokenizer,
            train_samples,
            published.selected_l7,
            device=device,
            config=l7_config,
        )
        _assert_only_selected_deltas_trainable(model, runtime7)
        l7_ranking = evaluate_candidate_ranking(model, tokenizer, eval_samples, candidate_universe, device=device)
        l7_direct = evaluate_samples(
            model, tokenizer, eval_samples, split="A_eval", device=device, max_new_tokens=16, batch_size=16
        )
        if abs(float(l7_ranking.accuracy) - published.l7_ranking_accuracy) > 1e-12:
            raise RuntimeError("SPARSE_PATH_L7_REPRODUCTION_MISMATCH ranking")
        if abs(float(l7_direct.exact) - published.l7_direct_accuracy) > 1e-12:
            raise RuntimeError("SPARSE_PATH_L7_REPRODUCTION_MISMATCH direct")
        _freeze_l7_runtime(model, runtime7)

        stages: list[dict[str, Any]] = []
        for layer, k, steps in zip(spec.transport_layers, spec.transport_k, spec.transport_steps):
            print(f"[pcu-path-depth{depth}] transport L{layer}/K{k} steps={steps}", flush=True)
            stages.append(
                _train_one_added_layer(
                    model,
                    tokenizer,
                    train_sequences,
                    layer=layer,
                    k=k,
                    steps=steps,
                    device=device,
                )
            )

        print(f"[pcu-path-depth{depth}] readout L{READOUT_LAYER}/K{READOUT_K} steps={READOUT_STEPS}", flush=True)
        stages.append(
            _train_one_added_layer(
                model,
                tokenizer,
                train_sequences,
                layer=READOUT_LAYER,
                k=READOUT_K,
                steps=READOUT_STEPS,
                device=device,
            )
        )

        metrics = _evaluate_final(model, tokenizer, eval_samples, candidate_universe, device=device)
        result = {
            "schema": "minicells.pcu-sparse-path-depth-001.topology-result.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "seed": ENGINEERING_SEED,
            "depth": int(depth),
            "status": "COMPLETE",
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": source,
            "foundation": dict(manifest),
            "dataset_manifest_sha256": world.manifest_sha256(),
            "available_moe_layers": list(available),
            "topology": spec.to_dict(),
            "all_topologies": {str(key): value.to_dict() for key, value in specs.items()},
            "l7_reproduction": {
                "ranking_eval_accuracy": float(l7_ranking.accuracy),
                "direct_accuracy": float(l7_direct.exact),
                "exact": True,
                "training": l7_training,
            },
            "stages": stages,
            "metrics": metrics,
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
        return result
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass


def classify_depth_sweep(results: Mapping[int, Mapping[str, Any]]) -> str:
    if set(int(key) for key in results) != set(DEPTHS):
        raise ValueError("depth sweep classification requires depth3/4/5")
    metrics = {int(depth): payload["metrics"] for depth, payload in results.items()}
    rescued = [
        depth
        for depth in DEPTHS
        if float(metrics[depth]["direct_accuracy"]) >= DIRECT_CAPABILITY_FLOOR
        and float(metrics[depth]["ranking_eval_accuracy"]) >= ASSOCIATION_FLOOR
    ]
    if rescued:
        return f"SPARSE_PATH_DEPTH_{min(rescued)}_RESCUES_NATIVE_GENERATION"
    best_depth = max(DEPTHS, key=lambda value: float(metrics[value]["direct_accuracy"]))
    best_direct = float(metrics[best_depth]["direct_accuracy"])
    if best_direct > EXPECTED_PREVIOUS_DIRECT:
        return "DEEPER_SPARSE_PATH_IMPROVES_BUT_DOES_NOT_RESCUE"
    return "DEEPER_SPARSE_PATH_DID_NOT_IMPROVE"


def aggregate_depth_sweep(
    *,
    worker_files: Mapping[int, Path],
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    _load_previous_cross_layer(PREVIOUS_CROSS_LAYER_ROOT)
    results: dict[int, dict[str, Any]] = {}
    for depth in DEPTHS:
        path = Path(worker_files[int(depth)])
        if not path.is_file():
            raise RuntimeError(f"missing depth{depth} worker result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") != EXPERIMENT_ID or int(payload.get("depth", -1)) != depth:
            raise RuntimeError(f"depth{depth} worker identity mismatch")
        if payload.get("valid_run") is not True or payload.get("formal_execution_not_started") is not True:
            raise RuntimeError(f"depth{depth} worker invalid/formal")
        spec = payload.get("topology", {})
        if int(spec.get("total_added_k", -1)) != TOTAL_ADDED_K:
            raise RuntimeError(f"depth{depth} Cell budget drift")
        if int(spec.get("total_added_steps", -1)) != TOTAL_ADDED_STEPS:
            raise RuntimeError(f"depth{depth} step budget drift")
        if payload.get("l7_reproduction", {}).get("exact") is not True:
            raise RuntimeError(f"depth{depth} failed L7 reproduction")
        results[depth] = payload

    topo3 = set(results[3]["topology"]["layers"])
    topo4 = set(results[4]["topology"]["layers"])
    topo5 = set(results[5]["topology"]["layers"])
    if not topo3.issubset(topo4) or not topo4.issubset(topo5):
        raise RuntimeError("published depth topologies are not nested")
    if not (
        results[3]["available_moe_layers"]
        == results[4]["available_moe_layers"]
        == results[5]["available_moe_layers"]
    ):
        raise RuntimeError("worker MoE layer discovery differs across GPUs")

    status = classify_depth_sweep(results)
    summary = {
        str(depth): {
            "layers": results[depth]["topology"]["layers"],
            "transport_k": results[depth]["topology"]["transport_k"],
            "transport_steps": results[depth]["topology"]["transport_steps"],
            "direct_accuracy": results[depth]["metrics"]["direct_accuracy"],
            "ranking_eval_accuracy": results[depth]["metrics"]["ranking_eval_accuracy"],
            "first_token_top1_accuracy": results[depth]["metrics"]["first_token_top1_accuracy"],
            "later_token_top1_accuracy": results[depth]["metrics"]["later_token_top1_accuracy"],
            "sequence_all_tokens_top1_accuracy": results[depth]["metrics"]["sequence_all_tokens_top1_accuracy"],
        }
        for depth in DEPTHS
    }
    best_depth = max(DEPTHS, key=lambda value: float(summary[str(value)]["direct_accuracy"]))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for depth in DEPTHS:
        write_json(output_root / f"DEPTH_{depth}.json", results[depth])
    result = {
        "schema": "minicells.pcu-sparse-path-depth-001.result.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "seed": ENGINEERING_SEED,
        "constant_budget": {
            "transport_k_total": TRANSPORT_K_TOTAL,
            "readout_k": READOUT_K,
            "total_added_k": TOTAL_ADDED_K,
            "transport_steps_total": TRANSPORT_STEPS_TOTAL,
            "readout_steps": READOUT_STEPS,
            "total_added_steps": TOTAL_ADDED_STEPS,
        },
        "previous_depth2_baseline": {
            "layers": [ASSOCIATION_LAYER, READOUT_LAYER],
            "direct_accuracy": EXPECTED_PREVIOUS_DIRECT,
            "ranking_eval_accuracy": EXPECTED_PREVIOUS_RANKING,
            "note": "previous experiment used only K16 added at L23 and is a lower-capacity anchor, not equal-budget control",
        },
        "depths": summary,
        "best_depth": int(best_depth),
        "best_direct_accuracy": float(summary[str(best_depth)]["direct_accuracy"]),
        "worker_files_external": {str(depth): str(worker_files[depth]) for depth in DEPTHS},
    }
    write_json(output_root / "RESULT.json", result)
    write_json(output_root / "DECISION.json", {
        "schema": "minicells.pcu-sparse-path-depth-001.decision.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "seed": ENGINEERING_SEED,
        "depths": summary,
        "best_depth": int(best_depth),
        "best_direct_accuracy": float(summary[str(best_depth)]["direct_accuracy"]),
        "association_floor": ASSOCIATION_FLOOR,
        "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
        "total_added_k_each": TOTAL_ADDED_K,
        "total_added_steps_each": TOTAL_ADDED_STEPS,
        "nested_topologies": True,
        "dual_gpu_execution_required": True,
    })
    return result


__all__ = [
    "EXPERIMENT_ID",
    "DEPTHS",
    "ASSOCIATION_LAYER",
    "READOUT_LAYER",
    "TRANSPORT_K_TOTAL",
    "READOUT_K",
    "TOTAL_ADDED_K",
    "TRANSPORT_STEPS_TOTAL",
    "READOUT_STEPS",
    "TOTAL_ADDED_STEPS",
    "DEFAULT_OUTPUT",
    "TopologySpec",
    "choose_nested_topologies",
    "run_topology",
    "aggregate_depth_sweep",
    "classify_depth_sweep",
]
