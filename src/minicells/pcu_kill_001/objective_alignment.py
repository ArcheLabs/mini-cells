"""Final engineering diagnostic for PCU local-Cell association learning.

PCU-OBJECTIVE-ALIGNMENT-001 freezes the best permissive locality condition from
PCU-LOCALITY-WIDTH-001 (L7, exact published K=64 Cell set, AdamW, LR=1e-3,
128 optimizer steps, effective batch=8, engineering seed 26090501, A-only
synthetic world) and changes exactly one scientific variable: the training
objective.

The original answer-token causal CE is replaced by a differentiable 16-way
candidate-ranking objective using the exact completion-token mean log-
likelihood semantics already used by context-oracle v2.  The correct V and 15
deterministic V-domain distractors are scored from the unchanged A_train prompt;
no answer is inserted into the prompt.

This is engineering-only evidence.  It does not modify the frozen/formal
PCU-KILL-001 protocol and never consumes formal seeds.
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
from .synthetic import (
    POSITIVE_CONTROL_CANDIDATES,
    _candidate_pool,
    _completion_encoding,
    _rank_candidate,
    audit_dataset,
    generate_world,
)
from .training import BranchTrainingConfig, ForkedCellularExperts, selected_delta_parameters


EXPERIMENT_ID = "PCU-OBJECTIVE-ALIGNMENT-001"
TARGET_K = 64
CANDIDATE_POOL_SIZE = POSITIVE_CONTROL_CANDIDATES
RANKING_TEMPERATURE = 1.0
RANKING_SAMPLE_MICROBATCH = 1
ASSOCIATION_FLOOR = 0.80
LOCALITY_BASELINE_ROOT = Path(
    "artifacts/research/pcu-locality-width-001/engineering/26090501-l7-width"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-objective-alignment-001/engineering/26090501-l7-k64-ranking"
)


@dataclass(frozen=True)
class RankingSummary:
    accuracy: float
    mean_correct_rank: float
    mean_correct_margin: float
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "mean_correct_rank": float(self.mean_correct_rank),
            "mean_correct_margin": float(self.mean_correct_margin),
            "rows": [dict(row) for row in self.rows],
        }


def _load_locality_baseline(root: Path) -> dict[str, Any]:
    root = Path(root)
    for name in ("RUN_IDENTITY.json", "DESIGN.json", "DECISION.json", "WIDTH_064.json"):
        if not (root / name).is_file():
            raise RuntimeError(
                f"objective-alignment requires completed PCU-LOCALITY-WIDTH-001 baseline: missing {name}"
            )
    identity = json.loads((root / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    design = json.loads((root / "DESIGN.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "DECISION.json").read_text(encoding="utf-8"))
    width = json.loads((root / "WIDTH_064.json").read_text(encoding="utf-8"))

    if identity.get("experiment") != "PCU-LOCALITY-WIDTH-001":
        raise RuntimeError("wrong locality-width baseline experiment")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("locality-width baseline crossed formal boundary")
    if decision.get("valid_run") is not True or decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("locality-width baseline is not valid pre-formal evidence")
    if decision.get("status") != "LOCALITY_WIDTH_IMPROVES_BUT_DOES_NOT_RESCUE":
        raise RuntimeError(
            "objective-alignment expects the completed locality-width non-rescue result"
        )
    if int(width.get("identity", {}).get("target_layer", -1)) != TARGET_LAYER:
        raise RuntimeError("locality-width baseline target layer changed")
    if int(width.get("identity", {}).get("selected_k", -1)) != TARGET_K:
        raise RuntimeError("locality-width baseline does not contain K=64")
    if float(width.get("identity", {}).get("learning_rate", -1.0)) != LEARNING_RATE:
        raise RuntimeError("locality-width baseline learning rate changed")
    if int(width.get("identity", {}).get("max_optimizer_steps", -1)) != MAX_OPTIMIZER_STEPS:
        raise RuntimeError("locality-width baseline step budget changed")
    if int(width.get("identity", {}).get("batch_size", -1)) != BATCH_SIZE:
        raise RuntimeError("locality-width baseline batch size changed")
    selected = tuple(str(value) for value in width.get("allocation", {}).get("selected", ()))
    if len(selected) != TARGET_K:
        raise RuntimeError("locality-width baseline selected Cell count is not exactly 64")
    if width.get("allocation", {}).get("baseline_prefix_match") is not True:
        raise RuntimeError("locality-width K64 allocation did not preserve the registered prefix")
    if width.get("formal_execution_not_started") is not True:
        raise RuntimeError("K64 baseline crossed formal boundary")

    # The WIDTH result identity binds the foundation/dataset; the L7 layer
    # baseline carries the full foundation manifest needed for exact reload.
    layer_root = Path(str(design.get("baseline", {}).get("artifact", "")))
    if not layer_root.is_absolute():
        layer_root = Path.cwd() / layer_root
    layer_path = layer_root / "LAYER_07.json"
    if not layer_path.is_file():
        raise RuntimeError(f"objective-alignment cannot resolve L7 foundation baseline: {layer_path}")
    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    foundation = layer.get("foundation")
    if not isinstance(foundation, Mapping):
        raise RuntimeError("L7 baseline lacks foundation identity")
    dataset_sha = str(width["identity"]["dataset_manifest_sha256"])
    if str(layer.get("dataset_manifest_sha256")) != dataset_sha:
        raise RuntimeError("K64 and L7 baseline dataset identities differ")
    if str(layer.get("target_path")) != str(width["identity"]["target_path"]):
        raise RuntimeError("K64 and L7 target paths differ")

    return {
        "foundation": dict(foundation),
        "dataset_manifest_sha256": dataset_sha,
        "target_path": str(width["identity"]["target_path"]),
        "selected": selected,
        "ce_direct_accuracy": float(width["direct_accuracy"]),
        "ce_final_loss": float(width["training"]["final_loss"]),
        "gradient_mass_at_k": float(width["allocation"]["gradient_mass_at_k"]),
        "effective_count": float(width["allocation"]["effective_count"]),
        "artifact_source": str(root),
        "baseline_source": identity.get("source", {}),
        "scientific_source": decision.get("scientific_source", identity.get("source", {})),
    }


def _candidate_scores_tensor(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[str],
    *,
    device: str,
) -> Tensor:
    """Differentiable oracle-v2 candidate scores: mean completion log-likelihood."""
    encoded = [_completion_encoding(tokenizer, prompt, str(candidate)) for candidate in candidates]
    full = [item[0] for item in encoded]
    completion_ids = [item[1] for item in encoded]
    prompt_length = len(full[0]) - len(completion_ids[0])
    if any(len(ids) - len(answer) != prompt_length for ids, answer in encoded):
        raise RuntimeError("ranking candidates do not share one prompt-token boundary")

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    width = max(len(value) for value in full)
    input_ids = torch.full((len(full), width), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((len(full), width), dtype=torch.long, device=device)
    for row, values in enumerate(full):
        input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        attention[row, : len(values)] = 1

    output = model(input_ids=input_ids, attention_mask=attention)
    logits = getattr(output, "logits", output)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if not isinstance(logits, Tensor):
        raise RuntimeError("ranking model output has no logits tensor")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    scores: list[Tensor] = []
    for row, answer_ids in enumerate(completion_ids):
        positions = torch.arange(
            prompt_length - 1,
            prompt_length + len(answer_ids) - 1,
            device=device,
        )
        targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
        values = log_probs[row, positions, targets]
        if not torch.isfinite(values).all():
            raise RuntimeError("non-finite differentiable ranking candidate score")
        scores.append(values.mean())
    return torch.stack(scores)


def _ranking_loss_for_sample(
    model: nn.Module,
    tokenizer: Any,
    sample: Any,
    candidate_universe: Sequence[str],
    *,
    device: str,
) -> tuple[Tensor, dict[str, Any]]:
    candidates = _candidate_pool(
        candidate_universe,
        str(sample.answer),
        str(sample.sample_id),
        size=CANDIDATE_POOL_SIZE,
    )
    if len(candidates) != CANDIDATE_POOL_SIZE:
        raise RuntimeError("objective-alignment requires exactly 16 candidates")
    correct_index = candidates.index(str(sample.answer))
    scores = _candidate_scores_tensor(
        model,
        tokenizer,
        str(sample.prompt),
        candidates,
        device=device,
    )
    scaled = scores / float(RANKING_TEMPERATURE)
    target = torch.tensor([correct_index], dtype=torch.long, device=device)
    loss = F.cross_entropy(scaled.unsqueeze(0), target)
    order = torch.argsort(scores.detach(), descending=True)
    rank = int((order == correct_index).nonzero(as_tuple=False)[0, 0]) + 1
    best_wrong = torch.max(torch.cat((scores[:correct_index], scores[correct_index + 1 :])))
    margin = float((scores[correct_index] - best_wrong).detach())
    return loss, {
        "sample_id": str(sample.sample_id),
        "correct": str(sample.answer),
        "correct_index": int(correct_index),
        "correct_rank": rank,
        "correct_margin": margin,
        "candidate_count": len(candidates),
    }


def _train_ranking_branch(
    model: nn.Module,
    block: nn.Module,
    parent_experts: nn.Module,
    tokenizer: Any,
    train_samples: Sequence[Any],
    candidate_universe: Sequence[str],
    selected_cells: Sequence[str],
    *,
    device: str,
    config: BranchTrainingConfig,
) -> tuple[ForkedCellularExperts, dict[str, Any]]:
    """Train K64 using exact 16-way ranking, effective batch=8.

    One training sample (16 candidate sequences) is materialized at a time to
    bound T4 activation memory.  Eight sample losses are each scaled by 1/8 and
    accumulated before one optimizer.step(), so this is exactly the mean
    ranking loss for the frozen effective batch of eight samples.
    """
    runtime = ForkedCellularExperts(
        parent_experts,
        _selected_map(selected_cells, TARGET_LAYER),
    ).to(device)
    parameters = selected_delta_parameters(runtime)
    if not parameters:
        raise RuntimeError("objective-alignment branch has no trainable Cell deltas")
    block.experts = runtime
    optimizer = torch.optim.AdamW(parameters, lr=float(config.learning_rate))
    torch.manual_seed(int(config.seed))

    samples = list(train_samples)
    if len(samples) < int(config.batch_size):
        raise RuntimeError("objective-alignment training set is smaller than effective batch")
    batches_per_epoch = len(samples) // int(config.batch_size)
    if batches_per_epoch <= 0:
        raise RuntimeError("objective-alignment has no full training batches")

    steps = 0
    correct_completion_tokens = 0
    candidate_completion_tokens = 0
    final_loss = float("nan")
    final_batch_ranking_accuracy = 0.0
    while steps < int(config.max_optimizer_steps):
        batch_index = steps % batches_per_epoch
        start = batch_index * int(config.batch_size)
        batch = samples[start : start + int(config.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        batch_losses: list[float] = []
        batch_exact = 0
        batch_correct_tokens = 0
        batch_candidate_tokens = 0

        for sample in batch:
            candidates = _candidate_pool(
                candidate_universe,
                str(sample.answer),
                str(sample.sample_id),
                size=CANDIDATE_POOL_SIZE,
            )
            for candidate in candidates:
                _, completion = _completion_encoding(tokenizer, str(sample.prompt), str(candidate))
                batch_candidate_tokens += len(completion)
                if str(candidate) == str(sample.answer):
                    batch_correct_tokens += len(completion)
            loss, diagnostic = _ranking_loss_for_sample(
                model,
                tokenizer,
                sample,
                candidate_universe,
                device=device,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite objective-alignment ranking loss")
            (loss / float(config.batch_size)).backward()
            batch_losses.append(float(loss.detach()))
            batch_exact += int(diagnostic["correct_rank"] == 1)

        if correct_completion_tokens + batch_correct_tokens > int(config.max_training_tokens):
            raise RuntimeError("objective-alignment unexpectedly hit the inherited token budget")
        optimizer.step()
        steps += 1
        correct_completion_tokens += batch_correct_tokens
        candidate_completion_tokens += batch_candidate_tokens
        final_loss = sum(batch_losses) / len(batch_losses)
        final_batch_ranking_accuracy = batch_exact / len(batch)
        if steps % 32 == 0 or steps == int(config.max_optimizer_steps):
            print(
                f"[pcu-objective] step={steps} rank_loss={final_loss:.6f} "
                f"batch_rank_acc={final_batch_ranking_accuracy:.3f} "
                f"correct_tokens={correct_completion_tokens}",
                flush=True,
            )

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return runtime, {
        "optimizer": config.optimizer,
        "learning_rate": float(config.learning_rate),
        "max_optimizer_steps": int(config.max_optimizer_steps),
        "max_training_tokens": int(config.max_training_tokens),
        "batch_size": int(config.batch_size),
        "training_steps": steps,
        "training_tokens": correct_completion_tokens,
        "candidate_compute_completion_tokens": candidate_completion_tokens,
        "objective": "16-way-candidate-ranking-cross-entropy-over-mean-completion-loglikelihood",
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "ranking_temperature": RANKING_TEMPERATURE,
        "sample_microbatch": RANKING_SAMPLE_MICROBATCH,
        "final_loss": final_loss,
        "final_batch_ranking_accuracy": final_batch_ranking_accuracy,
        "selected_cells": list(selected_cells),
    }


def evaluate_candidate_ranking(
    model: nn.Module,
    tokenizer: Any,
    samples: Sequence[Any],
    candidate_universe: Sequence[str],
    *,
    device: str,
) -> RankingSummary:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        candidates = _candidate_pool(
            candidate_universe,
            str(sample.answer),
            str(sample.sample_id),
            size=CANDIDATE_POOL_SIZE,
        )
        ranked = _rank_candidate(
            model,
            tokenizer,
            str(sample.prompt),
            candidates,
            str(sample.answer),
            device=device,
        )
        correct_score = float(ranked["correct_score"])
        wrong_scores = []
        # _rank_candidate intentionally exposes only aggregate ranking fields.
        # winner_margin is not the correct-vs-best-wrong margin when correct is
        # not rank 1, so recompute exact candidate scores only for diagnostics.
        with torch.inference_mode():
            scores = _candidate_scores_tensor(
                model,
                tokenizer,
                str(sample.prompt),
                candidates,
                device=device,
            ).detach().cpu().tolist()
        for candidate, score in zip(candidates, scores):
            if str(candidate) != str(sample.answer):
                wrong_scores.append(float(score))
        margin = correct_score - max(wrong_scores)
        rows.append({
            "sample_id": str(sample.sample_id),
            "correct": str(sample.answer),
            "winner": str(ranked["winner"]),
            "correct_rank": int(ranked["correct_rank"]),
            "correct_score": correct_score,
            "correct_margin": float(margin),
            "exact": bool(ranked["exact"]),
        })
    n = max(1, len(rows))
    return RankingSummary(
        accuracy=sum(bool(row["exact"]) for row in rows) / n,
        mean_correct_rank=sum(int(row["correct_rank"]) for row in rows) / n,
        mean_correct_margin=sum(float(row["correct_margin"]) for row in rows) / n,
        rows=tuple(rows),
    )


def _classify(
    *,
    direct_accuracy: float,
    eval_ranking_accuracy: float,
    ce_baseline_accuracy: float,
) -> str:
    if direct_accuracy >= DIRECT_CAPABILITY_FLOOR:
        return "OBJECTIVE_ALIGNMENT_RESCUES_LOCAL_CELL_MUTATION"
    if eval_ranking_accuracy >= ASSOCIATION_FLOOR:
        return "ASSOCIATION_LEARNED_GENERATION_UNRESOLVED"
    if direct_accuracy > ce_baseline_accuracy:
        return "OBJECTIVE_ALIGNMENT_IMPROVES_BUT_DOES_NOT_RESCUE"
    return "OBJECTIVE_ALIGNMENT_DID_NOT_RESCUE"


def run_objective_alignment_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    baseline_root: Path = LOCALITY_BASELINE_ROOT,
    device: str = "cuda:0",
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-OBJECTIVE-ALIGNMENT-001 is engineering-seed only")
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("objective-alignment requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load_locality_baseline(Path(baseline_root))
    source = git_provenance(Path(__file__).resolve().parents[3])
    if source.get("source_dirty") is not False:
        raise RuntimeError("objective-alignment diagnostic requires a clean source tree")

    design = {
        "schema": "minicells.pcu-objective-alignment-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_variable": "training_objective_only",
        "fixed": {
            "task": "A_only_U_to_V",
            "target_layer": TARGET_LAYER,
            "selected_k": TARGET_K,
            "selected_cells": list(baseline["selected"]),
            "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "effective_batch_size": BATCH_SIZE,
            "routing": "inherited_parent_router",
            "direct_evaluation": "A_eval_greedy_exact",
            "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
        },
        "changed": {
            "from": "answer-token-causal-cross-entropy",
            "to": "16-way-candidate-ranking-cross-entropy-over-mean-completion-loglikelihood",
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "candidate_source": "same-domain V identifiers with deterministic context-oracle-v2 pool",
            "candidate_score": "mean exact-completion-token log-likelihood",
            "ranking_temperature": RANKING_TEMPERATURE,
            "prompt": "unchanged A_train prompt; answer absent from prompt",
        },
        "secondary_evaluation": {
            "A_train_candidate_ranking": True,
            "A_eval_candidate_ranking": True,
            "association_floor": ASSOCIATION_FLOOR,
        },
        "baseline": {
            "artifact": baseline["artifact_source"],
            "ce_direct_accuracy": baseline["ce_direct_accuracy"],
            "ce_final_loss": baseline["ce_final_loss"],
            "gradient_mass_at_k": baseline["gradient_mass_at_k"],
            "effective_count": baseline["effective_count"],
        },
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-objective-alignment-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output.name,
        "source": source,
        "baseline_source": baseline["baseline_source"],
        "formal_execution_not_started": True,
    })

    tokenizer, model, manifest = load_granite(
        str(baseline["foundation"]["model_repo"]),
        revision=str(baseline["foundation"]["model_revision"]),
        device=str(device),
    )
    try:
        _validate_foundation_manifest(manifest, baseline["foundation"])
        model.requires_grad_(False)
        block = target_module(model, baseline["target_path"])
        projection = extract_expert_projections(block.experts, 0)
        cellular_experts = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
        model.requires_grad_(False)
        cellular_experts.requires_grad_(False)

        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed:
            raise RuntimeError(f"objective-alignment dataset audit failed: {audit.errors}")
        if world.manifest_sha256() != baseline["dataset_manifest_sha256"]:
            raise RuntimeError("objective-alignment dataset differs from K64 baseline")
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
            f"[pcu-objective] training L{TARGET_LAYER}/K{TARGET_K} with 16-way ranking on {device}",
            flush=True,
        )
        runtime, training = _train_ranking_branch(
            model,
            block,
            cellular_experts,
            tokenizer,
            train_samples,
            candidate_universe,
            baseline["selected"],
            device=str(device),
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime)
        if tuple(training["selected_cells"]) != tuple(baseline["selected"]):
            raise RuntimeError("OBJECTIVE_ALIGNMENT_ALLOCATION_DRIFT")

        print("[pcu-objective] evaluating candidate ranking on A_train", flush=True)
        train_ranking = evaluate_candidate_ranking(
            model, tokenizer, train_samples, candidate_universe, device=str(device)
        )
        print("[pcu-objective] evaluating candidate ranking on A_eval", flush=True)
        eval_ranking = evaluate_candidate_ranking(
            model, tokenizer, eval_samples, candidate_universe, device=str(device)
        )
        print("[pcu-objective] evaluating greedy A_eval direct capability", flush=True)
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
            direct_accuracy=float(direct.exact),
            eval_ranking_accuracy=float(eval_ranking.accuracy),
            ce_baseline_accuracy=float(baseline["ce_direct_accuracy"]),
        )
        result = {
            "schema": "minicells.pcu-objective-alignment-001.result.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": source,
            "foundation": dict(manifest),
            "dataset_manifest_sha256": world.manifest_sha256(),
            "selected_cells": list(baseline["selected"]),
            "selected_k": TARGET_K,
            "baseline": {
                "ce_direct_accuracy": baseline["ce_direct_accuracy"],
                "ce_final_loss": baseline["ce_final_loss"],
                "gradient_mass_at_k": baseline["gradient_mass_at_k"],
                "effective_count": baseline["effective_count"],
            },
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
            "interpretation": (
                "aligned ranking objective rescues the inherited direct-capability gate"
                if status == "OBJECTIVE_ALIGNMENT_RESCUES_LOCAL_CELL_MUTATION"
                else "association is learned under constrained ranking but greedy generation remains unresolved"
                if status == "ASSOCIATION_LEARNED_GENERATION_UNRESOLVED"
                else "aligned ranking objective improves direct behavior but does not rescue local Cell mutation"
                if status == "OBJECTIVE_ALIGNMENT_IMPROVES_BUT_DOES_NOT_RESCUE"
                else "aligned ranking objective does not rescue or improve the permissive L7/K64 local mutation condition"
            ),
        }
        write_json(output / "RESULT.json", result)
        write_json(output / "DECISION.json", {
            "schema": "minicells.pcu-objective-alignment-001.decision.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "ce_baseline_direct_accuracy": baseline["ce_direct_accuracy"],
            "ranking_train_accuracy": train_ranking.accuracy,
            "ranking_eval_accuracy": eval_ranking.accuracy,
            "direct_accuracy": direct.exact,
            "association_floor": ASSOCIATION_FLOOR,
            "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
            "selected_k": TARGET_K,
            "selected_cells_exact_baseline_match": True,
            "source": source,
            "interpretation": result["interpretation"],
        })
        return result
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "EXPERIMENT_ID",
    "TARGET_K",
    "CANDIDATE_POOL_SIZE",
    "ASSOCIATION_FLOOR",
    "LOCALITY_BASELINE_ROOT",
    "DEFAULT_OUTPUT",
    "evaluate_candidate_ranking",
    "run_objective_alignment_diagnostic",
]
