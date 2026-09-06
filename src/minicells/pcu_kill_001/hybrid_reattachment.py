"""Causal reattachment diagnostic for the ranking-only PCU mutation.

PCU-HYBRID-REATTACHMENT-001 asks whether the mature frozen Granite model can
*consume* an already-learned PCU mutation.  It deliberately does not ask the
Cell to become an autonomous language model.

The primary source mutation is the exact published PCU-OBJECTIVE-ALIGNMENT-001
L7/K64 ranking-only state: A_eval association ranking 0.8203125 while greedy
generation is 0.0.  The historical artifact did not publish restorable delta
weights, so the exact pinned training protocol is deterministically replayed.
No CE/readout regularizer is added.

The same memory-resident model is then measured with the mutation ON, with only
its delta tensors zeroed (OFF), and after byte-exact restoration (RESTORED).
Engineering evidence only; formal PCU seeds remain untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .hybrid_objective import OBJECTIVE_BASELINE_ROOT, _load_baselines
from .layer_placement import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    MAX_TRAINING_TOKENS,
    _assert_only_selected_deltas_trainable,
    _validate_foundation_manifest,
)
from .locality_width import ENGINEERING_SEED, TARGET_LAYER
from .model import load_granite, target_module
from .objective_alignment import (
    ASSOCIATION_FLOOR,
    CANDIDATE_POOL_SIZE,
    RANKING_TEMPERATURE,
    _train_ranking_branch,
    evaluate_candidate_ranking,
)
from .synthetic import audit_dataset, generate_world
from .task import IGNORE_INDEX, TaskSequences, build_task_sequences, validate_answer_only_labels
from .training import BranchTrainingConfig


EXPERIMENT_ID = "PCU-HYBRID-REATTACHMENT-001"
TARGET_K = 64
PUBLISHED_SOURCE_ROOT = OBJECTIVE_BASELINE_ROOT
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-hybrid-reattachment-001/engineering/26090501-l7-k64-ranking-causal-reattach"
)
EXPECTED_PUBLISHED_RANKING_ACCURACY = 0.8203125
EXPECTED_PUBLISHED_DIRECT_ACCURACY = 0.0

# Frozen before the first real GPU run.
EQUIVALENCE_MAX_ABS_LOGIT_DIFF = 1e-5
RESTORATION_MAX_ABS_LOGIT_DIFF = 1e-5
MIN_CAUSAL_RANKING_GAIN = 0.50
MIN_CAUSAL_MARGIN_GAIN = 0.0
MAX_CONTROL_ANSWER_NLL_INCREASE = 0.10
LOGIT_PROBE_ROWS = 8
EVAL_BATCH_SIZE = 16


@dataclass(frozen=True)
class LogitDiff:
    max_abs: float
    mean_abs: float
    elements: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs": float(self.max_abs),
            "mean_abs": float(self.mean_abs),
            "elements": int(self.elements),
        }


@dataclass(frozen=True)
class AnswerMetrics:
    answer_tokens: int
    token_top1_accuracy: float
    answer_nll: float
    mean_target_logit: float
    mean_target_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_tokens": int(self.answer_tokens),
            "token_top1_accuracy": float(self.token_top1_accuracy),
            "answer_nll": float(self.answer_nll),
            "mean_target_logit": float(self.mean_target_logit),
            "mean_target_margin": float(self.mean_target_margin),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _delta_parameters(runtime: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = [
        (name, parameter)
        for name, parameter in runtime.named_parameters()
        if name.startswith("delta_") or ".delta_" in name
    ]
    if not values:
        raise RuntimeError("reattachment runtime has no Cell delta parameters")
    return values


def _tensor_bytes(value: Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def delta_sha256(runtime: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(_delta_parameters(runtime), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(_tensor_bytes(parameter))
    return digest.hexdigest()


def frozen_parent_sha256(runtime: nn.Module) -> str:
    digest = hashlib.sha256()
    found = False
    for name, value in sorted(runtime.named_buffers(), key=lambda item: item[0]):
        if "parent_" not in name and not name.endswith("down_bias"):
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_tensor_bytes(value))
    if not found:
        raise RuntimeError("reattachment runtime exposes no frozen parent buffers")
    return digest.hexdigest()


@contextmanager
def temporarily_zero_cell_deltas(runtime: nn.Module) -> Iterator[dict[str, str]]:
    """Make the exact causal intervention and restore deltas byte-for-byte."""
    parameters = _delta_parameters(runtime)
    snapshots = [(name, parameter, parameter.detach().clone()) for name, parameter in parameters]
    before = delta_sha256(runtime)
    with torch.no_grad():
        for _, parameter, _ in snapshots:
            parameter.zero_()
    zero = delta_sha256(runtime)
    try:
        yield {"trained_sha256": before, "zero_sha256": zero}
    finally:
        with torch.no_grad():
            for _, parameter, snapshot in snapshots:
                parameter.copy_(snapshot)
        if delta_sha256(runtime) != before:
            raise RuntimeError("CELL_DELTA_RESTORATION_MISMATCH")


def _capture_logits(
    model: nn.Module,
    sequences: TaskSequences,
    *,
    device: str,
    rows: int = LOGIT_PROBE_ROWS,
) -> Tensor:
    count = min(int(rows), int(sequences.input_ids.shape[0]))
    if count <= 0:
        raise ValueError("logit probe needs at least one row")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model(
                input_ids=sequences.input_ids[:count].to(device),
                attention_mask=sequences.attention_mask[:count].to(device),
                use_cache=False,
            )
            logits = getattr(output, "logits", None)
            if not isinstance(logits, Tensor):
                raise RuntimeError("reattachment probe produced no logits tensor")
            return logits.detach().float().cpu()
    finally:
        model.train(was_training)


def compare_logits(left: Tensor, right: Tensor) -> LogitDiff:
    if left.shape != right.shape:
        raise ValueError(f"logit shapes differ: {tuple(left.shape)} != {tuple(right.shape)}")
    diff = (left.float() - right.float()).abs()
    return LogitDiff(
        max_abs=float(diff.max()) if diff.numel() else 0.0,
        mean_abs=float(diff.mean()) if diff.numel() else 0.0,
        elements=int(diff.numel()),
    )


def evaluate_answer_metrics(
    model: nn.Module,
    sequences: TaskSequences,
    *,
    device: str,
    batch_size: int = EVAL_BATCH_SIZE,
) -> AnswerMetrics:
    """Teacher-forced final-Granite readout on answer positions only."""
    validate_answer_only_labels(sequences)
    was_training = model.training
    model.eval()
    total_tokens = 0
    total_correct = 0
    nll_sum = 0.0
    target_logit_sum = 0.0
    margin_sum = 0.0
    try:
        with torch.inference_mode():
            for start in range(0, int(sequences.input_ids.shape[0]), int(batch_size)):
                end = min(int(sequences.input_ids.shape[0]), start + int(batch_size))
                labels = sequences.labels[start:end].to(device)
                output = model(
                    input_ids=sequences.input_ids[start:end].to(device),
                    attention_mask=sequences.attention_mask[start:end].to(device),
                    use_cache=False,
                )
                logits = getattr(output, "logits", None)
                if not isinstance(logits, Tensor):
                    raise RuntimeError("answer probe produced no logits tensor")
                shifted_logits = logits[:, :-1, :].float()
                shifted_labels = labels[:, 1:]
                mask = shifted_labels.ne(IGNORE_INDEX)
                if not bool(mask.any()):
                    continue
                active_logits = shifted_logits[mask]
                targets = shifted_labels[mask]
                target_logits = active_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                top2_values, top2_indices = active_logits.topk(k=2, dim=-1)
                competitors = torch.where(
                    top2_indices[:, 0].eq(targets), top2_values[:, 1], top2_values[:, 0]
                )
                losses = F.cross_entropy(active_logits, targets, reduction="none")
                total_tokens += int(targets.numel())
                total_correct += int(active_logits.argmax(dim=-1).eq(targets).sum())
                nll_sum += float(losses.sum())
                target_logit_sum += float(target_logits.sum())
                margin_sum += float((target_logits - competitors).sum())
    finally:
        model.train(was_training)
    if total_tokens <= 0:
        raise RuntimeError("answer probe saw no supervised answer tokens")
    return AnswerMetrics(
        answer_tokens=total_tokens,
        token_top1_accuracy=total_correct / total_tokens,
        answer_nll=nll_sum / total_tokens,
        mean_target_logit=target_logit_sum / total_tokens,
        mean_target_margin=margin_sum / total_tokens,
    )


def _load_published_source(root: Path) -> dict[str, Any]:
    decision_path = Path(root) / "DECISION.json"
    result_path = Path(root) / "RESULT.json"
    if not decision_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"missing objective-alignment source under {root}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if decision.get("experiment") != "PCU-OBJECTIVE-ALIGNMENT-001":
        raise RuntimeError("reattachment source has wrong experiment identity")
    if decision.get("status") != "ASSOCIATION_LEARNED_GENERATION_UNRESOLVED":
        raise RuntimeError("reattachment requires the association-learned/generation-unresolved source")
    if decision.get("valid_run") is not True or decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("objective-alignment source is invalid or crossed the formal boundary")
    if int(decision.get("selected_k", -1)) != TARGET_K:
        raise RuntimeError("objective-alignment source K changed")
    if abs(float(decision.get("ranking_eval_accuracy", -1)) - EXPECTED_PUBLISHED_RANKING_ACCURACY) > 1e-12:
        raise RuntimeError("published objective-alignment ranking result changed")
    if abs(float(decision.get("direct_accuracy", -1)) - EXPECTED_PUBLISHED_DIRECT_ACCURACY) > 1e-12:
        raise RuntimeError("published objective-alignment direct result changed")
    return {"decision": decision, "result": result, "root": str(root)}


def classify_reattachment(
    *,
    replay_matches: bool,
    equivalence_max_abs: float,
    off_equivalence_max_abs: float,
    restoration_max_abs: float,
    ranking_on: float,
    ranking_off: float,
    margin_gain: float,
    control_nll_increase: float,
) -> str:
    """Engineering classifier; it intentionally cannot emit a formal PASS."""
    if not replay_matches:
        return "REPLAY_DID_NOT_MATCH_PUBLISHED_MUTATION"
    if equivalence_max_abs > EQUIVALENCE_MAX_ABS_LOGIT_DIFF:
        return "ZERO_STATE_EQUIVALENCE_FAILED"
    if off_equivalence_max_abs > EQUIVALENCE_MAX_ABS_LOGIT_DIFF:
        return "OFF_STATE_EQUIVALENCE_FAILED"
    if restoration_max_abs > RESTORATION_MAX_ABS_LOGIT_DIFF:
        return "REVERSIBILITY_FAILED"
    ranking_gain = float(ranking_on) - float(ranking_off)
    if (
        ranking_on >= ASSOCIATION_FLOOR
        and ranking_gain >= MIN_CAUSAL_RANKING_GAIN
        and margin_gain > MIN_CAUSAL_MARGIN_GAIN
        and control_nll_increase <= MAX_CONTROL_ANSWER_NLL_INCREASE
    ):
        return "ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED"
    if ranking_gain > 0.0 or margin_gain > 0.0:
        return "CAUSAL_EXPRESSION_PRESENT_GATES_UNRESOLVED"
    return "NO_CAUSAL_EXPRESSION_ENGINEERING"


def run_hybrid_reattachment_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    source_root: Path = PUBLISHED_SOURCE_ROOT,
    device: str = "cuda:0",
    seed: int = ENGINEERING_SEED,
    run_direct_generation: bool = True,
) -> dict[str, Any]:
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("engineering runner may not consume formal PCU seeds")
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("reattachment diagnostic requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load_baselines(Path(source_root))
    published = _load_published_source(Path(source_root))
    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("reattachment diagnostic requires a clean source tree")

    design = {
        "schema": "minicells.pcu-hybrid-reattachment-001.design.v2",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_question": "can_frozen_Granite_consume_an_already_learned_PCU_mutation",
        "causal_variable": "exact_same_trained_Cell_deltas_ON_vs_temporarily_zeroed_OFF",
        "source_mutation": {
            "experiment": "PCU-OBJECTIVE-ALIGNMENT-001",
            "artifact": str(source_root),
            "reconstruction": "deterministic_exact_protocol_replay_weights_were_not_published",
            "target_layer": TARGET_LAYER,
            "selected_k": TARGET_K,
            "selected_cells": list(baseline.selected_cells),
            "objective": "16-way-candidate-ranking-only",
            "published_ranking_eval_accuracy": EXPECTED_PUBLISHED_RANKING_ACCURACY,
            "published_greedy_direct_accuracy": EXPECTED_PUBLISHED_DIRECT_ACCURACY,
        },
        "fixed": {
            "foundation": dict(baseline.foundation),
            "dataset_manifest_sha256": baseline.dataset_manifest_sha256,
            "router": "native_frozen_Granite_router",
            "downstream_tail": "native_frozen_Granite",
            "new_bridge": False,
            "new_readout": False,
            "new_allocation": False,
            "ce_readout_regularizer": False,
            "foundation_trainable": False,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "batch_size": BATCH_SIZE,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "ranking_temperature": RANKING_TEMPERATURE,
        },
        "states": ["BASE", "PARENT_ZERO_DELTA", "CELL_ON", "CELL_OFF", "CELL_RESTORED"],
        "primary_gates": {
            "equivalence_max_abs_logit_diff": EQUIVALENCE_MAX_ABS_LOGIT_DIFF,
            "restoration_max_abs_logit_diff": RESTORATION_MAX_ABS_LOGIT_DIFF,
            "association_floor": ASSOCIATION_FLOOR,
            "minimum_causal_ranking_gain": MIN_CAUSAL_RANKING_GAIN,
            "minimum_causal_margin_gain": MIN_CAUSAL_MARGIN_GAIN,
            "maximum_B_control_answer_nll_increase": MAX_CONTROL_ANSWER_NLL_INCREASE,
        },
        "non_gate_diagnostic": "free_generation_exact_accuracy",
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-hybrid-reattachment-001.run-identity.v2",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output.name,
        "source": source,
        "source_mutation_artifact": str(source_root),
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    })

    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=str(device),
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        model.requires_grad_(False)
        model.eval()
        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed:
            raise RuntimeError(f"reattachment dataset audit failed: {audit.errors}")
        if world.manifest_sha256() != baseline.dataset_manifest_sha256:
            raise RuntimeError("reattachment dataset differs from published mutation source")
        train_samples = list(world.splits["A_train"])
        a_eval = list(world.splits["A_eval"])
        b_eval = list(world.splits["B_eval"])
        a_sequences = build_task_sequences(tokenizer, a_eval, "A_eval", max_length=128)
        b_sequences = build_task_sequences(tokenizer, b_eval, "B_eval", max_length=128)
        candidate_universe = tuple(item.v for item in world.triples)

        # BASE: untouched mature Granite.
        base_a_logits = _capture_logits(model, a_sequences, device=str(device))
        base_b_logits = _capture_logits(model, b_sequences, device=str(device))
        base_a_answer = evaluate_answer_metrics(model, a_sequences, device=str(device))
        base_b_answer = evaluate_answer_metrics(model, b_sequences, device=str(device))

        # PARENT_ZERO_DELTA: G0 cellularization, no learned mutation.
        block = target_module(model, baseline.target_path)
        projection = extract_expert_projections(block.experts, 0)
        cellular_experts = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
        model.requires_grad_(False)
        cellular_experts.requires_grad_(False)
        parent_a_logits = _capture_logits(model, a_sequences, device=str(device))
        parent_b_logits = _capture_logits(model, b_sequences, device=str(device))
        base_parent_a = compare_logits(base_a_logits, parent_a_logits)
        base_parent_b = compare_logits(base_b_logits, parent_b_logits)

        # Reconstruct the exact ranking-only mutation.  No CE/readout term.
        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        print("[pcu-reattach] replaying published ranking-only L7/K64 mutation", flush=True)
        runtime, training = _train_ranking_branch(
            model,
            block,
            cellular_experts,
            tokenizer,
            train_samples,
            candidate_universe,
            baseline.selected_cells,
            device=str(device),
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime)
        if tuple(training["selected_cells"]) != baseline.selected_cells:
            raise RuntimeError("REATTACHMENT_ALLOCATION_DRIFT")
        parent_hash_before = frozen_parent_sha256(runtime)
        trained_delta_hash = delta_sha256(runtime)

        # CELL_ON.
        model.eval()
        on_a_logits = _capture_logits(model, a_sequences, device=str(device))
        on_b_logits = _capture_logits(model, b_sequences, device=str(device))
        on_a_answer = evaluate_answer_metrics(model, a_sequences, device=str(device))
        on_b_answer = evaluate_answer_metrics(model, b_sequences, device=str(device))
        on_ranking = evaluate_candidate_ranking(
            model, tokenizer, a_eval, candidate_universe, device=str(device)
        )
        on_direct = None
        if run_direct_generation:
            on_direct = evaluate_samples(
                model, tokenizer, a_eval, split="A_eval", device=str(device),
                max_new_tokens=16, batch_size=16,
            )

        # CELL_OFF: zero only mutable deltas, leaving router/base/tail identical.
        with temporarily_zero_cell_deltas(runtime) as intervention:
            off_a_logits = _capture_logits(model, a_sequences, device=str(device))
            off_b_logits = _capture_logits(model, b_sequences, device=str(device))
            off_a_answer = evaluate_answer_metrics(model, a_sequences, device=str(device))
            off_b_answer = evaluate_answer_metrics(model, b_sequences, device=str(device))
            off_ranking = evaluate_candidate_ranking(
                model, tokenizer, a_eval, candidate_universe, device=str(device)
            )
            off_direct = None
            if run_direct_generation:
                off_direct = evaluate_samples(
                    model, tokenizer, a_eval, split="A_eval", device=str(device),
                    max_new_tokens=16, batch_size=16,
                )

        # CELL_RESTORED.
        restored_delta_hash = delta_sha256(runtime)
        restored_a_logits = _capture_logits(model, a_sequences, device=str(device))
        restored_a_answer = evaluate_answer_metrics(model, a_sequences, device=str(device))
        parent_hash_after = frozen_parent_sha256(runtime)
        if parent_hash_after != parent_hash_before:
            raise RuntimeError("FROZEN_PARENT_MUTATION_DETECTED")
        if restored_delta_hash != trained_delta_hash:
            raise RuntimeError("TRAINED_DELTA_NOT_RESTORED")

        diffs = {
            "base_vs_parent_A": base_parent_a,
            "base_vs_parent_B": base_parent_b,
            "base_vs_off_A": compare_logits(base_a_logits, off_a_logits),
            "base_vs_off_B": compare_logits(base_b_logits, off_b_logits),
            "on_vs_off_A": compare_logits(on_a_logits, off_a_logits),
            "on_vs_off_B": compare_logits(on_b_logits, off_b_logits),
            "on_vs_restored_A": compare_logits(on_a_logits, restored_a_logits),
        }
        ranking_gain = float(on_ranking.accuracy) - float(off_ranking.accuracy)
        margin_gain = float(on_a_answer.mean_target_margin) - float(off_a_answer.mean_target_margin)
        target_nll_gain = float(off_a_answer.answer_nll) - float(on_a_answer.answer_nll)
        control_nll_increase = float(on_b_answer.answer_nll) - float(off_b_answer.answer_nll)
        replay_ranking_matches = (
            abs(float(on_ranking.accuracy) - EXPECTED_PUBLISHED_RANKING_ACCURACY) <= 1e-12
        )
        replay_direct_matches = True
        if on_direct is not None:
            replay_direct_matches = abs(float(on_direct.exact) - EXPECTED_PUBLISHED_DIRECT_ACCURACY) <= 1e-12
        replay_matches = bool(replay_ranking_matches and replay_direct_matches)

        status = classify_reattachment(
            replay_matches=replay_matches,
            equivalence_max_abs=max(base_parent_a.max_abs, base_parent_b.max_abs),
            off_equivalence_max_abs=max(diffs["base_vs_off_A"].max_abs, diffs["base_vs_off_B"].max_abs),
            restoration_max_abs=diffs["on_vs_restored_A"].max_abs,
            ranking_on=float(on_ranking.accuracy),
            ranking_off=float(off_ranking.accuracy),
            margin_gain=margin_gain,
            control_nll_increase=control_nll_increase,
        )
        result = {
            "schema": "minicells.pcu-hybrid-reattachment-001.result.v2",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": bool(replay_matches),
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": source,
            "foundation": dict(manifest),
            "dataset_manifest_sha256": world.manifest_sha256(),
            "source_mutation": {
                "experiment": "PCU-OBJECTIVE-ALIGNMENT-001",
                "artifact": published["root"],
                "objective": "ranking_only",
            },
            "selected_k": TARGET_K,
            "selected_cells": list(baseline.selected_cells),
            "training": training,
            "mutation": {
                "trained_delta_sha256": trained_delta_hash,
                "restored_delta_sha256": restored_delta_hash,
                "zero_intervention_sha256": intervention["zero_sha256"],
                "frozen_parent_sha256_before": parent_hash_before,
                "frozen_parent_sha256_after": parent_hash_after,
                "restoration_exact": restored_delta_hash == trained_delta_hash,
                "parent_unchanged": parent_hash_after == parent_hash_before,
            },
            "logit_differences": {name: value.to_dict() for name, value in diffs.items()},
            "answer_metrics": {
                "A_base": base_a_answer.to_dict(),
                "B_base": base_b_answer.to_dict(),
                "A_on": on_a_answer.to_dict(),
                "B_on": on_b_answer.to_dict(),
                "A_off": off_a_answer.to_dict(),
                "B_off": off_b_answer.to_dict(),
                "A_restored": restored_a_answer.to_dict(),
            },
            "causal_effect": {
                "ranking_on": float(on_ranking.accuracy),
                "ranking_off": float(off_ranking.accuracy),
                "ranking_gain": ranking_gain,
                "answer_margin_gain": margin_gain,
                "answer_nll_gain": target_nll_gain,
                "B_control_answer_nll_increase": control_nll_increase,
                "direct_on": float(on_direct.exact) if on_direct is not None else None,
                "direct_off": float(off_direct.exact) if off_direct is not None else None,
                "direct_gain": (
                    float(on_direct.exact) - float(off_direct.exact)
                    if on_direct is not None and off_direct is not None else None
                ),
            },
            "ranking": {"on": on_ranking.to_dict(), "off": off_ranking.to_dict()},
            "direct_generation": {
                "enabled": bool(run_direct_generation),
                "on": on_direct.to_dict() if on_direct is not None else None,
                "off": off_direct.to_dict() if off_direct is not None else None,
                "primary_success_metric": False,
            },
            "published_replay": {
                "expected_ranking_accuracy": EXPECTED_PUBLISHED_RANKING_ACCURACY,
                "observed_ranking_accuracy": float(on_ranking.accuracy),
                "ranking_matches": replay_ranking_matches,
                "expected_direct_accuracy": EXPECTED_PUBLISHED_DIRECT_ACCURACY,
                "observed_direct_accuracy": float(on_direct.exact) if on_direct is not None else None,
                "direct_matches": replay_direct_matches if on_direct is not None else None,
                "replay_matches": replay_matches,
            },
        }
        decision = {
            "schema": "minicells.pcu-hybrid-reattachment-001.decision.v2",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": bool(replay_matches),
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "formal_decision": "RESERVED_UNRUN",
            "source_mutation": "PCU-OBJECTIVE-ALIGNMENT-001/ranking-only/L7/K64",
            "zero_state_equivalence_passes": max(base_parent_a.max_abs, base_parent_b.max_abs) <= EQUIVALENCE_MAX_ABS_LOGIT_DIFF,
            "off_state_equivalence_passes": max(diffs["base_vs_off_A"].max_abs, diffs["base_vs_off_B"].max_abs) <= EQUIVALENCE_MAX_ABS_LOGIT_DIFF,
            "reversibility_passes": diffs["on_vs_restored_A"].max_abs <= RESTORATION_MAX_ABS_LOGIT_DIFF,
            "association_on_passes": float(on_ranking.accuracy) >= ASSOCIATION_FLOOR,
            "causal_ranking_gain_passes": ranking_gain >= MIN_CAUSAL_RANKING_GAIN,
            "causal_margin_gain_passes": margin_gain > MIN_CAUSAL_MARGIN_GAIN,
            "control_locality_passes": control_nll_increase <= MAX_CONTROL_ANSWER_NLL_INCREASE,
            "ranking_on": float(on_ranking.accuracy),
            "ranking_off": float(off_ranking.accuracy),
            "ranking_gain": ranking_gain,
            "answer_margin_gain": margin_gain,
            "answer_nll_gain": target_nll_gain,
            "B_control_answer_nll_increase": control_nll_increase,
            "cell_alone_takeover_required": False,
            "new_bridge_used": False,
            "ce_readout_regularizer_used": False,
            "source": source,
        }
        write_json(output / "RESULT.json", result)
        write_json(output / "DECISION.json", decision)
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
    "DEFAULT_OUTPUT",
    "PUBLISHED_SOURCE_ROOT",
    "EQUIVALENCE_MAX_ABS_LOGIT_DIFF",
    "RESTORATION_MAX_ABS_LOGIT_DIFF",
    "MIN_CAUSAL_RANKING_GAIN",
    "MAX_CONTROL_ANSWER_NLL_INCREASE",
    "LogitDiff",
    "AnswerMetrics",
    "delta_sha256",
    "frozen_parent_sha256",
    "temporarily_zero_cell_deltas",
    "compare_logits",
    "evaluate_answer_metrics",
    "classify_reattachment",
    "run_hybrid_reattachment_diagnostic",
]
