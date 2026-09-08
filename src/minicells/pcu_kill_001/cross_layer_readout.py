"""Minimal sparse cross-layer readout test for PCU local-Cell mutation.

PCU-CROSS-LAYER-READOUT-001 asks whether the already demonstrated L7
association state becomes a usable autoregressive capability when one small late
readout footprint is added.

The published L7/K64 hybrid mutation is replayed exactly and then frozen. K=16
L23 Cells are allocated with the existing first-64 A_train answer-token-CE
gradient geometry under that frozen-L7 state, and only those L23 deltas are
trained with the original answer-token CE for 128 steps.

A matched-footprint L23-only control reloads the untouched foundation and trains
the exact same selected L23 Cell IDs with the exact same objective/budget. Thus
the experiment compares:

    published L7-only hybrid
    foundation + same L23 readout footprint
    frozen L7 hybrid + L23 readout footprint

The L23-only arm is a matched-footprint control, not an independently optimized
upper bound for all possible L23-only allocations. Engineering-only evidence;
formal PCU-KILL-001 seeds are never consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .hybrid_objective import (
    OBJECTIVE_BASELINE_ROOT,
    _load_baselines,
    _train_hybrid_branch,
)
from .layer_placement import (
    BATCH_SIZE,
    CALIBRATION_BATCH_SIZE,
    CALIBRATION_ROWS,
    DIRECT_CAPABILITY_FLOOR,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    MAX_TRAINING_TOKENS,
    _assert_only_selected_deltas_trainable,
    _train_full_model_branch,
    _validate_foundation_manifest,
    full_model_task_conditioned_allocation,
)
from .locality_width import ENGINEERING_SEED
from .model import load_granite, target_module
from .objective_alignment import ASSOCIATION_FLOOR, evaluate_candidate_ranking
from .readout_localization import gold_prefix_token_readout
from .synthetic import audit_dataset, generate_world
from .task import build_task_sequences, validate_answer_only_labels
from .training import Allocation, BranchTrainingConfig


EXPERIMENT_ID = "PCU-CROSS-LAYER-READOUT-001"
ASSOCIATION_LAYER = 7
ASSOCIATION_K = 64
READOUT_LAYER = 23
READOUT_K = 16
READOUT_OBJECTIVE = "answer-token-causal-cross-entropy"
SYNERGY_FLOOR = 0.30
HYBRID_BASELINE_ROOT = Path(
    "artifacts/research/pcu-hybrid-objective-001/engineering/26090501-l7-k64-rank-plus-ce025"
)
READOUT_BASELINE_ROOT = Path(
    "artifacts/research/pcu-readout-localization-001/engineering/26090501-l7-k64-hybrid-readout"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-cross-layer-readout-001/engineering/26090501-l7k64-plus-l23k16"
)
HYBRID_SCIENTIFIC_SOURCE_COMMIT = "0241475a387a9114415cf7ed143670dd5c7e1b3b"
HYBRID_CORE_BLOB_SHA = "851c77cdd283def0698ebe721ea8bf216f5ed556"
EXPECTED_L7_RANKING = 0.8359375
EXPECTED_L7_DIRECT = 0.03125
EXPECTED_L7_LATER_TOKEN_TOP1 = 0.535031847133758


@dataclass(frozen=True)
class PublishedBaselines:
    selected_l7: tuple[str, ...]
    dataset_manifest_sha256: str
    l7_ranking_accuracy: float
    l7_direct_accuracy: float
    l7_later_token_top1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "l7_ranking_accuracy": float(self.l7_ranking_accuracy),
            "l7_direct_accuracy": float(self.l7_direct_accuracy),
            "l7_later_token_top1": float(self.l7_later_token_top1),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=_repo_root(), text=True, stderr=subprocess.DEVNULL
    ).strip()


def _assert_hybrid_core_unchanged() -> None:
    blob = _git("rev-parse", "HEAD:src/minicells/pcu_kill_001/hybrid_objective.py")
    if blob != HYBRID_CORE_BLOB_SHA:
        raise RuntimeError(f"HYBRID_SCIENTIFIC_CORE_DRIFT: {blob}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", HYBRID_SCIENTIFIC_SOURCE_COMMIT, "HEAD"],
        cwd=_repo_root(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("hybrid scientific source is not an ancestor of HEAD")


def _load_published_baselines(hybrid_root: Path, readout_root: Path) -> PublishedBaselines:
    hybrid_root = Path(hybrid_root)
    readout_root = Path(readout_root)
    for root, names in (
        (hybrid_root, ("RESULT.json", "DECISION.json")),
        (readout_root, ("RESULT.json", "DECISION.json")),
    ):
        for name in names:
            if not (root / name).is_file():
                raise RuntimeError(f"cross-layer readout requires published baseline: missing {root / name}")

    hybrid_result = json.loads((hybrid_root / "RESULT.json").read_text(encoding="utf-8"))
    hybrid_decision = json.loads((hybrid_root / "DECISION.json").read_text(encoding="utf-8"))
    readout_result = json.loads((readout_root / "RESULT.json").read_text(encoding="utf-8"))
    readout_decision = json.loads((readout_root / "DECISION.json").read_text(encoding="utf-8"))

    if hybrid_decision.get("status") != "HYBRID_OBJECTIVE_PRESERVES_ASSOCIATION_GENERATION_UNRESOLVED":
        raise RuntimeError("cross-layer test requires published hybrid readout failure")
    if readout_decision.get("status") != "SINGLE_LAYER_GOLD_PREFIX_READOUT_INADEQUATE":
        raise RuntimeError("cross-layer test requires published single-layer readout inadequacy")
    if hybrid_decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("hybrid baseline crossed formal boundary")
    if readout_decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("readout baseline crossed formal boundary")
    if readout_decision.get("hybrid_reproduction_exact") is not True:
        raise RuntimeError("readout baseline did not exactly reproduce hybrid model")

    ranking = float(hybrid_decision["ranking_eval_accuracy"])
    direct = float(hybrid_decision["direct_accuracy"])
    later = float(readout_decision["later_token_top1_accuracy"])
    for actual, expected, label in (
        (ranking, EXPECTED_L7_RANKING, "L7 ranking"),
        (direct, EXPECTED_L7_DIRECT, "L7 direct"),
        (later, EXPECTED_L7_LATER_TOKEN_TOP1, "L7 later-token top1"),
    ):
        if abs(actual - expected) > 1e-12:
            raise RuntimeError(f"published {label} changed: expected {expected}, got {actual}")

    selected = tuple(str(value) for value in hybrid_result.get("selected_cells", ()))
    if len(selected) != ASSOCIATION_K or int(hybrid_result.get("selected_k", -1)) != ASSOCIATION_K:
        raise RuntimeError("published L7 hybrid is not exact K64")
    if tuple(str(value) for value in readout_result.get("selected_cells", ())) != selected:
        raise RuntimeError("readout diagnostic L7 Cell identity differs from hybrid baseline")
    dataset_sha = str(hybrid_result["dataset_manifest_sha256"])
    if str(readout_result.get("dataset_manifest_sha256")) != dataset_sha:
        raise RuntimeError("hybrid/readout baseline dataset identity differs")
    return PublishedBaselines(
        selected_l7=selected,
        dataset_manifest_sha256=dataset_sha,
        l7_ranking_accuracy=ranking,
        l7_direct_accuracy=direct,
        l7_later_token_top1=later,
    )


def _freeze_l7_runtime(model: nn.Module, runtime: nn.Module) -> None:
    runtime.requires_grad_(False)
    model.requires_grad_(False)
    unexpected = [name for name, value in model.named_parameters() if value.requires_grad]
    if unexpected:
        raise RuntimeError(f"L7 freeze failed: {unexpected[:8]}")


def _gradient_mass_at_k(allocation: Allocation, k: int) -> float:
    ordered = sorted(allocation.scores, key=lambda key: (-allocation.scores[key], key))
    total = sum(float(value) for value in allocation.scores.values())
    if total <= 0.0:
        return 0.0
    return sum(float(allocation.scores[key]) for key in ordered[: int(k)]) / total


def _evaluate_arm(
    model: nn.Module,
    tokenizer: Any,
    eval_samples: Sequence[Any],
    candidate_universe: Sequence[str],
    *,
    device: str,
) -> dict[str, Any]:
    ranking = evaluate_candidate_ranking(
        model, tokenizer, eval_samples, candidate_universe, device=device
    )
    direct = evaluate_samples(
        model,
        tokenizer,
        eval_samples,
        split="A_eval",
        device=device,
        max_new_tokens=16,
        batch_size=16,
    )
    eval_sequences = build_task_sequences(tokenizer, eval_samples, "A_eval", max_length=128)
    validate_answer_only_labels(eval_sequences)
    gold = gold_prefix_token_readout(model, tokenizer, eval_sequences, device=device, batch_size=8)
    return {
        "ranking_eval_accuracy": float(ranking.accuracy),
        "direct_accuracy": float(direct.exact),
        "later_token_top1_accuracy": float(gold["later_token_top1_accuracy"]),
        "first_token_top1_accuracy": float(gold["first_token_top1_accuracy"]),
        "sequence_all_tokens_top1_accuracy": float(gold["sequence_all_tokens_top1_accuracy"]),
        "ranking": ranking.to_dict(),
        "direct_evaluation": direct.to_dict(),
        "gold_prefix": gold,
    }


def _classify(*, l7_direct: float, l23_only_direct: float, cross_direct: float, cross_ranking: float) -> str:
    cross_direct_pass = cross_direct >= DIRECT_CAPABILITY_FLOOR
    cross_ranking_pass = cross_ranking >= ASSOCIATION_FLOOR
    l23_only_pass = l23_only_direct >= DIRECT_CAPABILITY_FLOOR
    synergy = cross_direct - max(l7_direct, l23_only_direct)
    if cross_direct_pass and cross_ranking_pass and not l23_only_pass:
        if synergy >= SYNERGY_FLOOR:
            return "SPARSE_CROSS_LAYER_READOUT_RESCUE_SUPPORTED"
        return "CROSS_LAYER_READOUT_RESCUE_WITH_WEAK_SYNERGY"
    if l23_only_pass:
        return "L23_ONLY_READOUT_SUFFICIENT_CROSS_LAYER_NOT_REQUIRED"
    if cross_direct_pass and not cross_ranking_pass:
        return "CROSS_LAYER_GENERATION_RESCUE_ASSOCIATION_REGRESSED"
    if cross_direct > max(l7_direct, l23_only_direct):
        return "CROSS_LAYER_READOUT_IMPROVES_BUT_DOES_NOT_RESCUE"
    return "MINIMAL_L23_READOUT_DID_NOT_HELP"


def _train_l23_only_control(
    *,
    baseline: Any,
    published: PublishedBaselines,
    selected_l23: Sequence[str],
    device: str,
) -> dict[str, Any]:
    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=device,
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        model.requires_grad_(False)
        block23 = target_module(model, f"model.layers.{READOUT_LAYER}.block_sparse_moe")
        projection23 = extract_expert_projections(block23.experts, 0)
        cellular23 = patch_moe_block(block23, CellPartition(projection23.intermediate_size, 4))
        model.requires_grad_(False)
        cellular23.requires_grad_(False)

        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed or world.manifest_sha256() != published.dataset_manifest_sha256:
            raise RuntimeError("L23-only control dataset identity mismatch")
        train_samples = list(world.splits["A_train"])
        eval_samples = list(world.splits["A_eval"])
        train_sequences = build_task_sequences(tokenizer, train_samples, "A_train", max_length=128)
        validate_answer_only_labels(train_sequences)
        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        runtime23, training = _train_full_model_branch(
            model,
            block23,
            cellular23,
            train_sequences,
            selected_l23,
            layer=READOUT_LAYER,
            device=device,
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime23)
        if tuple(training["selected_cells"]) != tuple(selected_l23):
            raise RuntimeError("L23_ONLY_CONTROL_ALLOCATION_DRIFT")
        metrics = _evaluate_arm(
            model,
            tokenizer,
            eval_samples,
            tuple(item.v for item in world.triples),
            device=device,
        )
        return {
            "schema": "minicells.pcu-cross-layer-readout-001.arm.v1",
            "arm": "foundation_plus_l23_only_matched_footprint",
            "control_scope": "matched_footprint_not_independently_optimized_L23_upper_bound",
            "selected_l23": list(selected_l23),
            "training": training,
            "metrics": metrics,
        }
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass


def run_cross_layer_readout_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    objective_root: Path = OBJECTIVE_BASELINE_ROOT,
    hybrid_root: Path = HYBRID_BASELINE_ROOT,
    readout_root: Path = READOUT_BASELINE_ROOT,
    device: str = "cuda:0",
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-CROSS-LAYER-READOUT-001 is engineering-seed only")
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("cross-layer readout requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)
    _assert_hybrid_core_unchanged()

    published = _load_published_baselines(Path(hybrid_root), Path(readout_root))
    baseline = _load_baselines(Path(objective_root))
    if baseline.selected_cells != published.selected_l7:
        raise RuntimeError("objective/hybrid L7 Cell identity drifted")
    if baseline.dataset_manifest_sha256 != published.dataset_manifest_sha256:
        raise RuntimeError("objective/hybrid dataset identity drifted")

    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("cross-layer readout requires a clean source tree")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    design = {
        "schema": "minicells.pcu-cross-layer-readout-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_question": "does_a_minimal_late_readout_footprint_rescue_frozen_L7_association",
        "association_state": {
            "layer": ASSOCIATION_LAYER,
            "selected_k": ASSOCIATION_K,
            "selected_cells": list(published.selected_l7),
            "training": "exact_replay_of_published_rank_plus_ce025_then_frozen",
        },
        "readout_state": {
            "layer": READOUT_LAYER,
            "selected_k": READOUT_K,
            "allocation": "first64_A_train_answer_CE_gradient_under_frozen_L7_state",
            "objective": READOUT_OBJECTIVE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "effective_batch_size": BATCH_SIZE,
            "calibration_rows": CALIBRATION_ROWS,
            "calibration_batch_size": CALIBRATION_BATCH_SIZE,
        },
        "required_arms": [
            "published_L7_only",
            "foundation_plus_same_L23_only_matched_footprint",
            "frozen_L7_plus_L23",
        ],
        "control_limit": "L23_only_uses_cross_layer_selected_cells_and_is_not_an_independently_optimized_capacity_upper_bound",
        "success_gate": {
            "cross_layer_ranking_min": ASSOCIATION_FLOOR,
            "cross_layer_direct_min": DIRECT_CAPABILITY_FLOOR,
            "l23_only_direct_must_be_below": DIRECT_CAPABILITY_FLOOR,
            "strong_support_direct_synergy_over_best_control_min": SYNERGY_FLOOR,
        },
        "published_L7_baseline": published.to_dict(),
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-cross-layer-readout-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output.name,
        "source": source,
        "hybrid_scientific_source_commit": HYBRID_SCIENTIFIC_SOURCE_COMMIT,
        "hybrid_scientific_core_blob_sha": HYBRID_CORE_BLOB_SHA,
        "formal_execution_not_started": True,
    })

    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=device,
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        model.requires_grad_(False)
        block7 = target_module(model, f"model.layers.{ASSOCIATION_LAYER}.block_sparse_moe")
        projection7 = extract_expert_projections(block7.experts, 0)
        cellular7 = patch_moe_block(block7, CellPartition(projection7.intermediate_size, 4))
        model.requires_grad_(False)
        cellular7.requires_grad_(False)

        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed or world.manifest_sha256() != published.dataset_manifest_sha256:
            raise RuntimeError("cross-layer dataset identity mismatch")
        train_samples = list(world.splits["A_train"])
        eval_samples = list(world.splits["A_eval"])
        candidate_universe = tuple(item.v for item in world.triples)
        train_sequences = build_task_sequences(tokenizer, train_samples, "A_train", max_length=128)
        validate_answer_only_labels(train_sequences)

        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        print("[pcu-cross-layer] replaying L7/K64 hybrid association state", flush=True)
        runtime7, l7_training = _train_hybrid_branch(
            model,
            block7,
            cellular7,
            tokenizer,
            train_samples,
            published.selected_l7,
            device=device,
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime7)
        l7_ranking = evaluate_candidate_ranking(
            model, tokenizer, eval_samples, candidate_universe, device=device
        )
        l7_direct = evaluate_samples(
            model,
            tokenizer,
            eval_samples,
            split="A_eval",
            device=device,
            max_new_tokens=16,
            batch_size=16,
        )
        if abs(float(l7_ranking.accuracy) - published.l7_ranking_accuracy) > 1e-12:
            raise RuntimeError("L7_HYBRID_REPRODUCTION_MISMATCH ranking")
        if abs(float(l7_direct.exact) - published.l7_direct_accuracy) > 1e-12:
            raise RuntimeError("L7_HYBRID_REPRODUCTION_MISMATCH direct")
        _freeze_l7_runtime(model, runtime7)

        block23 = target_module(model, f"model.layers.{READOUT_LAYER}.block_sparse_moe")
        projection23 = extract_expert_projections(block23.experts, 0)
        cellular23 = patch_moe_block(block23, CellPartition(projection23.intermediate_size, 4))
        model.requires_grad_(False)
        cellular23.requires_grad_(False)
        print("[pcu-cross-layer] allocating L23 readout Cells under frozen L7 state", flush=True)
        allocation = full_model_task_conditioned_allocation(
            model,
            block23,
            cellular23,
            train_sequences,
            layer=READOUT_LAYER,
            calibration_rows=CALIBRATION_ROWS,
            calibration_batch_size=CALIBRATION_BATCH_SIZE,
            device=device,
        )
        selected_l23 = tuple(allocation.selected[:READOUT_K])
        if len(selected_l23) != READOUT_K:
            raise RuntimeError("L23 allocation did not produce exact K16")
        gradient_mass_at_k = _gradient_mass_at_k(allocation, READOUT_K)

        print(f"[pcu-cross-layer] training frozen-L7 + L23/K{READOUT_K} readout", flush=True)
        runtime23, cross_training = _train_full_model_branch(
            model,
            block23,
            cellular23,
            train_sequences,
            selected_l23,
            layer=READOUT_LAYER,
            device=device,
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime23)
        if tuple(cross_training["selected_cells"]) != selected_l23:
            raise RuntimeError("CROSS_LAYER_L23_ALLOCATION_DRIFT")
        cross_metrics = _evaluate_arm(
            model, tokenizer, eval_samples, candidate_universe, device=device
        )
        cross_arm = {
            "schema": "minicells.pcu-cross-layer-readout-001.arm.v1",
            "arm": "frozen_l7_plus_l23",
            "selected_l23": list(selected_l23),
            "training": cross_training,
            "metrics": cross_metrics,
        }
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass

    print("[pcu-cross-layer] running exact same L23 footprint without L7 mutation", flush=True)
    l23_only = _train_l23_only_control(
        baseline=baseline,
        published=published,
        selected_l23=selected_l23,
        device=device,
    )
    if tuple(l23_only["selected_l23"]) != selected_l23:
        raise RuntimeError("L23 control did not reuse exact cross-layer Cell set")

    l23_direct = float(l23_only["metrics"]["direct_accuracy"])
    cross_direct = float(cross_arm["metrics"]["direct_accuracy"])
    cross_ranking = float(cross_arm["metrics"]["ranking_eval_accuracy"])
    synergy = cross_direct - max(published.l7_direct_accuracy, l23_direct)
    status = _classify(
        l7_direct=published.l7_direct_accuracy,
        l23_only_direct=l23_direct,
        cross_direct=cross_direct,
        cross_ranking=cross_ranking,
    )
    allocation_payload = {
        "method": "task-conditioned-gradient-l2-per-parameter",
        "state": "frozen_L7_hybrid",
        "calibration_split": "A_train",
        "calibration_sample_rule": "first_64_samples",
        "calibration_execution": f"weighted_microbatch_{CALIBRATION_BATCH_SIZE}",
        "selected_k": READOUT_K,
        "selected": list(selected_l23),
        "topk_mass_registered_by_shared_helper": {str(key): value for key, value in allocation.topk_mass.items()},
        "gradient_mass_at_k": gradient_mass_at_k,
        "effective_count": float(allocation.effective_count),
    }
    result = {
        "schema": "minicells.pcu-cross-layer-readout-001.result.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "source": source,
        "foundation": dict(baseline.foundation),
        "dataset_manifest_sha256": published.dataset_manifest_sha256,
        "published_l7_only": published.to_dict(),
        "l7_reproduction": {
            "ranking_eval_accuracy": float(l7_ranking.accuracy),
            "direct_accuracy": float(l7_direct.exact),
            "training": l7_training,
            "exact": True,
        },
        "l23_allocation": allocation_payload,
        "l23_only_control": l23_only,
        "cross_layer_arm": cross_arm,
        "comparison": {
            "l7_only_direct": published.l7_direct_accuracy,
            "l23_only_direct": l23_direct,
            "cross_layer_direct": cross_direct,
            "cross_layer_ranking": cross_ranking,
            "direct_synergy_over_best_control": synergy,
            "direct_synergy_floor": SYNERGY_FLOOR,
            "association_floor": ASSOCIATION_FLOOR,
            "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
            "l23_only_control_scope": "matched_footprint_not_independently_optimized_capacity_upper_bound",
        },
    }
    write_json(output / "RESULT.json", result)
    decision = {
        "schema": "minicells.pcu-cross-layer-readout-001.decision.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "association_layer": ASSOCIATION_LAYER,
        "association_k": ASSOCIATION_K,
        "readout_layer": READOUT_LAYER,
        "readout_k": READOUT_K,
        "readout_objective": READOUT_OBJECTIVE,
        "l7_reproduction_exact": True,
        "l23_selected_once_and_reused": True,
        "l23_only_control_scope": "matched_footprint_not_independently_optimized_capacity_upper_bound",
        "l7_only_direct_accuracy": published.l7_direct_accuracy,
        "l23_only_direct_accuracy": l23_direct,
        "cross_layer_direct_accuracy": cross_direct,
        "cross_layer_ranking_accuracy": cross_ranking,
        "cross_layer_later_token_top1_accuracy": cross_arm["metrics"]["later_token_top1_accuracy"],
        "l23_only_later_token_top1_accuracy": l23_only["metrics"]["later_token_top1_accuracy"],
        "direct_synergy_over_best_control": synergy,
        "direct_synergy_floor": SYNERGY_FLOOR,
        "association_floor": ASSOCIATION_FLOOR,
        "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
        "selected_l23": list(selected_l23),
        "l23_gradient_mass_at_k": gradient_mass_at_k,
        "l23_effective_count": float(allocation.effective_count),
        "source": source,
    }
    write_json(output / "DECISION.json", decision)
    return result


__all__ = [
    "EXPERIMENT_ID",
    "ASSOCIATION_LAYER",
    "ASSOCIATION_K",
    "READOUT_LAYER",
    "READOUT_K",
    "READOUT_OBJECTIVE",
    "SYNERGY_FLOOR",
    "DEFAULT_OUTPUT",
    "_classify",
    "_gradient_mass_at_k",
    "run_cross_layer_readout_diagnostic",
]
