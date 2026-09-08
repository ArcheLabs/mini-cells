"""Narrow hybrid-objective diagnostic for PCU local-Cell readout.

PCU-HYBRID-OBJECTIVE-001 starts from the completed objective-alignment result:
L7, the exact same K64 Cell set, engineering seed 26090501, A-only synthetic
world, AdamW, LR=1e-3, 128 optimizer steps, effective batch=8, inherited parent
routing, and the same direct/ranking evaluations.

Exactly one scientific knob is added to the ranking objective:

    L_hybrid = L_rank + 0.25 * L_CE

L_rank is the existing 16-way context-oracle-v2 candidate-ranking loss.
L_CE is the original answer-token causal CE used by the paired CE/K64 control,
computed with the original task-sequence encoding rather than approximated from
candidate-ranking scores. No layer, K, allocation, optimizer, dataset, seed,
budget, prompt family, or decoding setting changes.

Engineering-only evidence; formal PCU-KILL-001 seeds are never consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .layer_placement import (
    BATCH_SIZE,
    DIRECT_CAPABILITY_FLOOR,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    MAX_TRAINING_TOKENS,
    _assert_only_selected_deltas_trainable,
    _selected_map,
    _validate_foundation_manifest,
)
from .locality_width import ENGINEERING_SEED, TARGET_LAYER
from .model import load_granite, target_module
from .objective_alignment import (
    ASSOCIATION_FLOOR,
    CANDIDATE_POOL_SIZE,
    RANKING_TEMPERATURE,
    _candidate_scores_tensor,
    _ranking_diagnostic,
    evaluate_candidate_ranking,
)
from .synthetic import _candidate_pool, audit_dataset, generate_world
from .task import answer_token_cross_entropy, build_task_sequences, validate_answer_only_labels
from .training import BranchTrainingConfig, ForkedCellularExperts, selected_delta_parameters


EXPERIMENT_ID = "PCU-HYBRID-OBJECTIVE-001"
TARGET_K = 64
CE_WEIGHT = 0.25
OBJECTIVE_BASELINE_ROOT = Path(
    "artifacts/research/pcu-objective-alignment-001/engineering/26090501-l7-k64-ranking"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-hybrid-objective-001/engineering/26090501-l7-k64-rank-plus-ce025"
)
EXPECTED_RANKING_EVAL = 0.8203125
EXPECTED_RANKING_TRAIN = 0.9921875
EXPECTED_RANKING_DIRECT = 0.0
EXPECTED_CE_DIRECT = 0.265625


@dataclass(frozen=True)
class HybridBaselines:
    foundation: Mapping[str, Any]
    dataset_manifest_sha256: str
    target_path: str
    selected_cells: tuple[str, ...]
    ce_direct_accuracy: float
    ranking_train_accuracy: float
    ranking_eval_accuracy: float
    ranking_direct_accuracy: float
    objective_artifact: str
    objective_source: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ce_direct_accuracy": float(self.ce_direct_accuracy),
            "ranking_train_accuracy": float(self.ranking_train_accuracy),
            "ranking_eval_accuracy": float(self.ranking_eval_accuracy),
            "ranking_direct_accuracy": float(self.ranking_direct_accuracy),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_baselines(root: Path) -> HybridBaselines:
    root = Path(root)
    required = ("RUN_IDENTITY.json", "DESIGN.json", "RESULT.json", "DECISION.json", "PAIRED_CE_K64.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"hybrid objective requires published objective-alignment baseline: missing {missing}")

    identity = json.loads((root / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    design = json.loads((root / "DESIGN.json").read_text(encoding="utf-8"))
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "DECISION.json").read_text(encoding="utf-8"))
    ce = json.loads((root / "PAIRED_CE_K64.json").read_text(encoding="utf-8"))

    if identity.get("experiment") != "PCU-OBJECTIVE-ALIGNMENT-001":
        raise RuntimeError("hybrid baseline has wrong experiment identity")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("objective-alignment baseline crossed formal boundary")
    if decision.get("status") != "ASSOCIATION_LEARNED_GENERATION_UNRESOLVED":
        raise RuntimeError("hybrid objective requires the association-learned/readout-unresolved result")
    if decision.get("valid_run") is not True or decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("objective-alignment baseline is not valid pre-formal evidence")
    if design.get("causal_variable") != "training_objective_only":
        raise RuntimeError("objective-alignment baseline design identity changed")

    ranking_train = float(decision["ranking_train_accuracy"])
    ranking_eval = float(decision["ranking_eval_accuracy"])
    ranking_direct = float(decision["direct_accuracy"])
    ce_direct = float(ce["direct_accuracy"])
    for actual, expected, label in (
        (ranking_train, EXPECTED_RANKING_TRAIN, "ranking_train_accuracy"),
        (ranking_eval, EXPECTED_RANKING_EVAL, "ranking_eval_accuracy"),
        (ranking_direct, EXPECTED_RANKING_DIRECT, "ranking_direct_accuracy"),
        (ce_direct, EXPECTED_CE_DIRECT, "ce_direct_accuracy"),
    ):
        if abs(actual - expected) > 1e-12:
            raise RuntimeError(f"hybrid baseline {label} changed: expected {expected}, got {actual}")

    selected = tuple(str(value) for value in ce.get("selected_cells", ()))
    if len(selected) != TARGET_K or int(ce.get("selected_k", -1)) != TARGET_K:
        raise RuntimeError("paired CE baseline is not exact K64")
    if tuple(str(value) for value in result.get("selected_cells", ())) != selected:
        raise RuntimeError("ranking and CE baselines do not use identical K64 Cells")
    if int(result.get("selected_k", -1)) != TARGET_K:
        raise RuntimeError("ranking baseline K changed")
    if decision.get("selected_cells_exact_baseline_match") is not True:
        raise RuntimeError("objective-alignment decision did not certify allocation identity")

    dataset_sha = str(ce["dataset_manifest_sha256"])
    if str(result.get("dataset_manifest_sha256")) != dataset_sha:
        raise RuntimeError("ranking and CE baseline datasets differ")
    foundation = ce.get("foundation")
    if not isinstance(foundation, Mapping):
        raise RuntimeError("paired CE baseline lacks foundation manifest")
    target_path = str(ce.get("target_path", ""))
    if target_path != f"model.layers.{TARGET_LAYER}.block_sparse_moe":
        raise RuntimeError("hybrid baseline target path changed")

    return HybridBaselines(
        foundation=dict(foundation),
        dataset_manifest_sha256=dataset_sha,
        target_path=target_path,
        selected_cells=selected,
        ce_direct_accuracy=ce_direct,
        ranking_train_accuracy=ranking_train,
        ranking_eval_accuracy=ranking_eval,
        ranking_direct_accuracy=ranking_direct,
        objective_artifact=str(root),
        objective_source=dict(identity.get("source", {})),
    )


def _hybrid_ranking_loss_for_sample(
    model: nn.Module,
    tokenizer: Any,
    sample: Any,
    candidate_universe: Sequence[str],
    *,
    device: str,
) -> tuple[Tensor, dict[str, Any], int]:
    """Compute the unchanged ranking term for one sample.

    The CE regularizer is intentionally not derived from these scores because
    the original CE control uses task-sequence tokenization without the ranking
    completion's injected whitespace. The caller computes original CE once per
    effective 8-row batch.
    """
    candidates = _candidate_pool(
        candidate_universe,
        str(sample.answer),
        str(sample.sample_id),
        size=CANDIDATE_POOL_SIZE,
    )
    if len(candidates) != CANDIDATE_POOL_SIZE:
        raise RuntimeError("hybrid objective requires exactly 16 ranking candidates")
    correct_index = candidates.index(str(sample.answer))
    scores, completion_ids = _candidate_scores_tensor(
        model,
        tokenizer,
        str(sample.prompt),
        candidates,
        device=device,
    )
    target = torch.tensor([correct_index], dtype=torch.long, device=device)
    ranking_loss = F.cross_entropy(
        (scores / float(RANKING_TEMPERATURE)).unsqueeze(0),
        target,
    )
    diagnostic = _ranking_diagnostic(candidates, str(sample.answer), scores)
    diagnostic.update({
        "sample_id": str(sample.sample_id),
        "correct_index": int(correct_index),
        "candidate_count": len(candidates),
    })
    return ranking_loss, diagnostic, sum(len(value) for value in completion_ids)


def _train_hybrid_branch(
    model: nn.Module,
    block: nn.Module,
    parent_experts: nn.Module,
    tokenizer: Any,
    train_samples: Sequence[Any],
    selected_cells: Sequence[str],
    *,
    device: str,
    config: BranchTrainingConfig,
) -> tuple[ForkedCellularExperts, dict[str, Any]]:
    """Train exact K64 with rank + 0.25*original-answer-CE.

    For each effective batch of eight, eight single-sample ranking forwards are
    accumulated with 1/8 weight, followed by one original task-sequence CE
    forward weighted by CE_WEIGHT. One optimizer step is then taken. This keeps
    the same batch traversal, optimizer, step budget, and answer-token budget as
    both registered controls.
    """
    runtime = ForkedCellularExperts(
        parent_experts,
        _selected_map(selected_cells, TARGET_LAYER),
    ).to(device)
    parameters = selected_delta_parameters(runtime)
    if not parameters:
        raise RuntimeError("hybrid branch has no trainable Cell deltas")
    block.experts = runtime
    optimizer = torch.optim.AdamW(parameters, lr=float(config.learning_rate))
    torch.manual_seed(int(config.seed))

    samples = list(train_samples)
    sequences = build_task_sequences(tokenizer, samples, "A_train", max_length=128)
    validate_answer_only_labels(sequences)
    candidate_universe = tuple(str(sample.answer) for sample in samples)
    batches_per_epoch = len(samples) // int(config.batch_size)
    if batches_per_epoch <= 0 or len(samples) % int(config.batch_size) != 0:
        raise RuntimeError("hybrid objective requires full deterministic batches")

    steps = 0
    answer_tokens = 0
    candidate_compute_tokens = 0
    final_rank_loss = float("nan")
    final_ce_loss = float("nan")
    final_hybrid_loss = float("nan")
    final_batch_ranking_accuracy = 0.0

    while steps < int(config.max_optimizer_steps):
        batch_index = steps % batches_per_epoch
        start = batch_index * int(config.batch_size)
        end = start + int(config.batch_size)
        batch = samples[start:end]
        optimizer.zero_grad(set_to_none=True)

        rank_losses: list[float] = []
        rank_exact = 0
        batch_candidate_tokens = 0
        for sample in batch:
            rank_loss, diagnostic, candidate_tokens = _hybrid_ranking_loss_for_sample(
                model,
                tokenizer,
                sample,
                candidate_universe,
                device=device,
            )
            if not torch.isfinite(rank_loss):
                raise RuntimeError("non-finite hybrid ranking loss")
            (rank_loss / float(config.batch_size)).backward()
            rank_losses.append(float(rank_loss.detach()))
            rank_exact += int(bool(diagnostic["exact"]))
            batch_candidate_tokens += int(candidate_tokens)

        input_ids = sequences.input_ids[start:end].to(device)
        attention = sequences.attention_mask[start:end].to(device)
        labels = sequences.labels[start:end].to(device)
        batch_answer_tokens = int(sequences.loss_mask[start:end].sum())
        if answer_tokens + batch_answer_tokens > int(config.max_training_tokens):
            raise RuntimeError("hybrid objective unexpectedly hit inherited answer-token budget")
        output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise RuntimeError("hybrid CE forward produced no logits tensor")
        ce_loss = answer_token_cross_entropy(logits, labels)
        if not torch.isfinite(ce_loss):
            raise RuntimeError("non-finite hybrid CE loss")
        (float(CE_WEIGHT) * ce_loss).backward()
        optimizer.step()

        steps += 1
        answer_tokens += batch_answer_tokens
        candidate_compute_tokens += batch_candidate_tokens
        final_rank_loss = sum(rank_losses) / len(rank_losses)
        final_ce_loss = float(ce_loss.detach())
        final_hybrid_loss = final_rank_loss + float(CE_WEIGHT) * final_ce_loss
        final_batch_ranking_accuracy = rank_exact / len(batch)
        if steps % 32 == 0 or steps == int(config.max_optimizer_steps):
            print(
                f"[pcu-hybrid] step={steps} hybrid={final_hybrid_loss:.6f} "
                f"rank={final_rank_loss:.6f} ce={final_ce_loss:.6f} "
                f"batch_rank_acc={final_batch_ranking_accuracy:.3f} tokens={answer_tokens}",
                flush=True,
            )

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    if steps != int(config.max_optimizer_steps):
        raise RuntimeError(f"hybrid branch stopped at {steps}, expected {config.max_optimizer_steps}")
    return runtime, {
        "optimizer": config.optimizer,
        "learning_rate": float(config.learning_rate),
        "max_optimizer_steps": int(config.max_optimizer_steps),
        "max_training_tokens": int(config.max_training_tokens),
        "batch_size": int(config.batch_size),
        "training_steps": steps,
        "training_tokens": answer_tokens,
        "candidate_compute_completion_tokens": candidate_compute_tokens,
        "objective": "ranking-plus-original-answer-token-ce",
        "ranking_weight": 1.0,
        "ce_weight": float(CE_WEIGHT),
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "ranking_temperature": RANKING_TEMPERATURE,
        "use_cache": False,
        "final_ranking_loss": final_rank_loss,
        "final_ce_loss": final_ce_loss,
        "final_hybrid_loss": final_hybrid_loss,
        "final_batch_ranking_accuracy": final_batch_ranking_accuracy,
        "selected_cells": list(selected_cells),
    }


def _classify(*, ranking_accuracy: float, direct_accuracy: float) -> str:
    ranking_pass = ranking_accuracy >= ASSOCIATION_FLOOR
    direct_pass = direct_accuracy >= DIRECT_CAPABILITY_FLOOR
    if ranking_pass and direct_pass:
        return "HYBRID_OBJECTIVE_RESCUES_ASSOCIATION_AND_GENERATION"
    if ranking_pass:
        return "HYBRID_OBJECTIVE_PRESERVES_ASSOCIATION_GENERATION_UNRESOLVED"
    if direct_pass:
        return "HYBRID_OBJECTIVE_RESCUES_GENERATION_ASSOCIATION_REGRESSED"
    return "HYBRID_OBJECTIVE_DID_NOT_JOINTLY_RESCUE"


def run_hybrid_objective_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    baseline_root: Path = OBJECTIVE_BASELINE_ROOT,
    device: str = "cuda:0",
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-HYBRID-OBJECTIVE-001 is engineering-seed only")
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("hybrid objective requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load_baselines(Path(baseline_root))
    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("hybrid objective diagnostic requires a clean source tree")

    design = {
        "schema": "minicells.pcu-hybrid-objective-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_variable": "ce_readout_regularizer_weight_only",
        "fixed": {
            "task": "A_only_U_to_V",
            "target_layer": TARGET_LAYER,
            "selected_k": TARGET_K,
            "selected_cells": list(baseline.selected_cells),
            "dataset_manifest_sha256": baseline.dataset_manifest_sha256,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "effective_batch_size": BATCH_SIZE,
            "ranking_weight": 1.0,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "ranking_temperature": RANKING_TEMPERATURE,
            "routing": "inherited_parent_router",
            "ranking_evaluation": "A_eval_16way_exact",
            "direct_evaluation": "A_eval_greedy_exact",
            "association_floor": ASSOCIATION_FLOOR,
            "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
        },
        "changed": {
            "from": "ranking_only",
            "to": "ranking_plus_original_answer_token_ce",
            "ce_weight": float(CE_WEIGHT),
            "ce_encoding": "original_task_sequence_encoding",
            "ce_prompt_and_answer": "unchanged A_train prompt and answer",
        },
        "baselines": baseline.to_dict(),
        "baseline_artifact": baseline.objective_artifact,
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-hybrid-objective-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output.name,
        "source": source,
        "baseline_source": dict(baseline.objective_source),
        "formal_execution_not_started": True,
    })

    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=str(device),
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        model.requires_grad_(False)
        block = target_module(model, baseline.target_path)
        projection = extract_expert_projections(block.experts, 0)
        cellular_experts = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
        model.requires_grad_(False)
        cellular_experts.requires_grad_(False)

        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed:
            raise RuntimeError(f"hybrid objective dataset audit failed: {audit.errors}")
        if world.manifest_sha256() != baseline.dataset_manifest_sha256:
            raise RuntimeError("hybrid objective dataset differs from published baselines")
        train_samples = list(world.splits["A_train"])
        eval_samples = list(world.splits["A_eval"])
        candidate_universe = tuple(item.v for item in world.triples)

        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        print(
            f"[pcu-hybrid] training L{TARGET_LAYER}/K{TARGET_K} "
            f"rank + {CE_WEIGHT:g}*CE on {device}",
            flush=True,
        )
        runtime, training = _train_hybrid_branch(
            model,
            block,
            cellular_experts,
            tokenizer,
            train_samples,
            baseline.selected_cells,
            device=str(device),
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime)
        if tuple(training["selected_cells"]) != baseline.selected_cells:
            raise RuntimeError("HYBRID_OBJECTIVE_ALLOCATION_DRIFT")

        print("[pcu-hybrid] evaluating candidate ranking on A_train", flush=True)
        train_ranking = evaluate_candidate_ranking(
            model, tokenizer, train_samples, candidate_universe, device=str(device)
        )
        print("[pcu-hybrid] evaluating candidate ranking on A_eval", flush=True)
        eval_ranking = evaluate_candidate_ranking(
            model, tokenizer, eval_samples, candidate_universe, device=str(device)
        )
        print("[pcu-hybrid] evaluating greedy A_eval", flush=True)
        direct = evaluate_samples(
            model,
            tokenizer,
            eval_samples,
            split="A_eval",
            device=str(device),
            max_new_tokens=16,
            batch_size=16,
        )

        status = _classify(
            ranking_accuracy=float(eval_ranking.accuracy),
            direct_accuracy=float(direct.exact),
        )
        result = {
            "schema": "minicells.pcu-hybrid-objective-001.result.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": source,
            "foundation": dict(manifest),
            "dataset_manifest_sha256": world.manifest_sha256(),
            "selected_cells": list(baseline.selected_cells),
            "selected_k": TARGET_K,
            "ce_weight": float(CE_WEIGHT),
            "baselines": baseline.to_dict(),
            "training": training,
            "ranking": {
                "train": train_ranking.to_dict(),
                "eval": eval_ranking.to_dict(),
                "association_floor": ASSOCIATION_FLOOR,
                "eval_passes": bool(eval_ranking.accuracy >= ASSOCIATION_FLOOR),
            },
            "direct_evaluation": direct.to_dict(),
            "direct_accuracy": float(direct.exact),
            "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
            "direct_passes": bool(direct.exact >= DIRECT_CAPABILITY_FLOOR),
        }
        write_json(output / "RESULT.json", result)
        write_json(output / "DECISION.json", {
            "schema": "minicells.pcu-hybrid-objective-001.decision.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "ce_weight": float(CE_WEIGHT),
            "ranking_train_accuracy": train_ranking.accuracy,
            "ranking_eval_accuracy": eval_ranking.accuracy,
            "direct_accuracy": direct.exact,
            "association_floor": ASSOCIATION_FLOOR,
            "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
            "ranking_passes": bool(eval_ranking.accuracy >= ASSOCIATION_FLOOR),
            "direct_passes": bool(direct.exact >= DIRECT_CAPABILITY_FLOOR),
            "selected_k": TARGET_K,
            "selected_cells_exact_baseline_match": True,
            "baselines": baseline.to_dict(),
            "source": source,
        })
        return result
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass


__all__ = [
    "EXPERIMENT_ID",
    "TARGET_K",
    "CE_WEIGHT",
    "OBJECTIVE_BASELINE_ROOT",
    "DEFAULT_OUTPUT",
    "run_hybrid_objective_diagnostic",
]
