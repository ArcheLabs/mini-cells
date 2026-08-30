from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Iterator, Literal

import torch
from torch.nn import functional as F

from .language_data import batch_from_starts
from .language_models import TextNCALM
from .sparse_cellular_textnca import SparseCellularTextNCA
from .textnca_to_clm import convert_textnca_to_sparse_cellular


NUM_PROGRAMS = 8
CONDITIONALITY_THRESHOLD = 0.05
NORMALIZED_ADVANTAGE_THRESHOLD = 0.002
QUALITY_RATIO_THRESHOLD = 1.03
RECEPTOR_RATIO_THRESHOLD = 0.05
DENSE_PPL_RATIO_ATOL = 1e-5
DENSE_LOGITS_ATOL = 5e-5
DENSE_RECURRENT_STATE_ATOL = 1e-6
Arm = Literal["dense", "dynamic", "static", "shuffled"]


@dataclass(frozen=True)
class EvaluationResult:
    arm: Arm
    top_k: int
    validation_nll: float
    validation_ppl: float
    dense_executor_flops: int
    receptor_flops: int
    active_executor_flops: int
    executor_ratio: float
    effective_compute_ratio: float
    tokens_per_second: float
    milliseconds_per_batch: float
    peak_vram_bytes: int
    program_usage: torch.Tensor
    program_coactivation: torch.Tensor
    structural_variation: float
    position_variation: float
    temporal_variation: float


def load_experiment_006_teacher(
    checkpoint_path: str,
    *,
    device: torch.device,
    model_config_path: str | None = None,
) -> TextNCALM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "minicells.language-checkpoint.v1":
        raise RuntimeError(
            f"unexpected Experiment 006 checkpoint format: {checkpoint.get('format')!r}"
        )
    if (
        checkpoint.get("model_name") != "minicells-v2"
        or checkpoint.get("consumed_tokens") != 10_000_000
    ):
        raise RuntimeError("checkpoint is not the locked Experiment 006 minicells-v2 10M source")
    locked = {
        "vocab_size": 2048,
        "context_length": 128,
        "dim": 128,
        "heads": 4,
        "ffn_dim": 512,
        "windows": [8, 32, 128],
        "iterations": [4, 4, 4],
        "gru_carry_bias": 2.0,
        "normalization": "LayerNorm",
    }
    if model_config_path is not None:
        documented = json.loads(Path(model_config_path).read_text(encoding="utf-8"))
        candidate = documented["minicells-v2"]
        shared = documented["shared"]
        observed = {
            "vocab_size": shared["vocab_size"],
            "context_length": shared["context_length"],
            **{key: candidate[key] for key in
               ("dim", "heads", "ffn_dim", "windows", "iterations",
                "gru_carry_bias", "normalization")},
        }
        if observed != locked:
            raise RuntimeError(f"Experiment 006 model configuration drifted: {observed}")
    teacher = TextNCALM(
        vocab_size=locked["vocab_size"],
        max_context=locked["context_length"],
        dim=locked["dim"],
        heads=locked["heads"],
        ffn_dim=locked["ffn_dim"],
        windows=tuple(locked["windows"]),
        iterations=tuple(locked["iterations"]),
        rms_norm=False,
        carry_bias=locked["gru_carry_bias"],
        tie_embeddings=True,
        stage_supervision=False,
    )
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.to(device).eval().requires_grad_(False)
    return teacher


def _record_gru_outputs(
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    outputs: list[torch.Tensor] = []
    handles = [
        stage.gru.register_forward_hook(
            lambda _module, _inputs, output: outputs.append(output.detach().clone())
        )
        for stage in model.stages
    ]
    return outputs, handles


def dense_equivalence_passes(
    *,
    ppl_ratio: float,
    max_logits_abs_diff: float,
    max_recurrent_state_abs_diff: float,
) -> bool:
    """Apply the preregistered FP32 parity tolerances, including GPU reduction drift."""
    return (
        abs(ppl_ratio - 1.0) <= DENSE_PPL_RATIO_ATOL
        and max_logits_abs_diff <= DENSE_LOGITS_ATOL
        and max_recurrent_state_abs_diff <= DENSE_RECURRENT_STATE_ATOL
    )


@torch.no_grad()
def validate_real_conversion(
    teacher: TextNCALM,
    student: SparseCellularTextNCA,
    validation_stream: torch.Tensor,
    validation_starts: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    teacher.eval()
    student.eval()
    student.set_routing_mode("dense")
    teacher_loss = 0.0
    student_loss = 0.0
    tokens = 0
    max_logits_diff = 0.0
    max_state_diff = 0.0
    dense_flops = 0
    receptor_flops = 0
    for starts in validation_starts:
        inputs, targets = batch_from_starts(validation_stream, starts, sequence_length, device)
        teacher_states, teacher_handles = _record_gru_outputs(teacher)
        student_states, student_handles = _record_gru_outputs(student)
        expected = teacher(inputs)
        actual, stats = student(inputs, return_stats=True)
        for handle in [*teacher_handles, *student_handles]:
            handle.remove()
        if len(teacher_states) != len(student_states):
            raise RuntimeError("teacher/student recurrent trace lengths differ")
        max_logits_diff = max(max_logits_diff, float((expected.logits - actual.logits).abs().max()))
        for expected_state, actual_state in zip(teacher_states, student_states):
            max_state_diff = max(max_state_diff, float((expected_state - actual_state).abs().max()))
        teacher_loss += float(
            F.cross_entropy(expected.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        student_loss += float(
            F.cross_entropy(actual.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        tokens += targets.numel()
        dense_flops += stats.dense_executor_flops
        receptor_flops += stats.receptor_flops
    teacher_nll = teacher_loss / tokens
    student_nll = student_loss / tokens
    ratio = math.exp(student_nll - teacher_nll)
    passed = dense_equivalence_passes(
        ppl_ratio=ratio,
        max_logits_abs_diff=max_logits_diff,
        max_recurrent_state_abs_diff=max_state_diff,
    )
    return {
        "status": "CLM_DENSE_EQUIVALENCE" if passed else "CLM_DENSE_EQUIVALENCE_FAILURE",
        "teacher_nll": teacher_nll,
        "student_nll": student_nll,
        "teacher_ppl": math.exp(teacher_nll),
        "student_ppl": math.exp(student_nll),
        "ppl_ratio": ratio,
        "max_logits_abs_diff": max_logits_diff,
        "max_recurrent_state_abs_diff": max_state_diff,
        "dense_executor_flops": dense_flops,
        "receptor_flops": receptor_flops,
        "receptor_ratio": receptor_flops / dense_flops,
        "tolerances": {
            "ppl_ratio_atol": DENSE_PPL_RATIO_ATOL,
            "logits_atol": DENSE_LOGITS_ATOL,
            "recurrent_state_atol": DENSE_RECURRENT_STATE_ATOL,
        },
    }


class RoutingRecorder:
    def __init__(self, model: SparseCellularTextNCA) -> None:
        self.model = model
        self.masks: list[torch.Tensor] = []
        self._originals: list[object] = []

    def __enter__(self) -> RoutingRecorder:
        for stage in self.model.stages:
            original = stage._gates
            self._originals.append(original)

            def wrapped(this: object, perception: torch.Tensor, phenotype: torch.Tensor | None,
                        original: object = original) -> tuple[torch.Tensor, torch.Tensor]:
                cell, programs = original(perception, phenotype)
                self.masks.append(programs.detach().clone())
                return cell, programs

            stage._gates = MethodType(wrapped, stage)
        return self

    def __exit__(self, *_exc: object) -> None:
        for stage, original in zip(self.model.stages, self._originals):
            stage._gates = original


@contextlib.contextmanager
def replay_routing(
    model: SparseCellularTextNCA,
    masks: list[torch.Tensor],
) -> Iterator[None]:
    cursor = 0
    originals: list[object] = []
    for stage in model.stages:
        original = stage._gates
        originals.append(original)

        def replay(this: object, perception: torch.Tensor, phenotype: torch.Tensor | None,
                   original: object = original) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal cursor
            cell, _ = original(perception, phenotype)
            programs = masks[cursor].to(device=perception.device, dtype=perception.dtype)
            cursor += 1
            return cell, programs

        stage._gates = MethodType(replay, stage)
    try:
        yield
        if cursor != len(masks):
            raise RuntimeError(f"consumed {cursor} routing masks, expected {len(masks)}")
    finally:
        for stage, original in zip(model.stages, originals):
            stage._gates = original


@torch.no_grad()
def collect_dynamic_masks(model: SparseCellularTextNCA, inputs: torch.Tensor) -> list[torch.Tensor]:
    with RoutingRecorder(model) as recorder:
        model(inputs)
    return recorder.masks


def static_topk_masks(mask_batches: list[list[torch.Tensor]], top_k: int) -> torch.Tensor:
    if not mask_batches:
        raise ValueError("calibration masks must not be empty")
    usage = torch.stack(
        [torch.cat([mask.reshape(-1, mask.shape[-1]) for mask in masks]).mean(0)
         for masks in mask_batches]
    ).mean(0)
    indices = usage.topk(top_k).indices
    return torch.zeros_like(usage).scatter(0, indices, 1.0)


def expand_static_mask(model: SparseCellularTextNCA, inputs: torch.Tensor,
                       mask: torch.Tensor) -> list[torch.Tensor]:
    count = sum(stage.iterations for stage in model.stages)
    expanded = mask.view(1, 1, -1).expand(inputs.shape[0], inputs.shape[1], -1)
    return [expanded for _ in range(count)]


def shuffled_masks(masks: list[torch.Tensor], permutation: torch.Tensor) -> list[torch.Tensor]:
    return [mask.index_select(0, permutation.to(mask.device)) for mask in masks]


def routing_variation_metrics(masks: list[torch.Tensor]) -> dict[str, float]:
    """Measure routing differences without mixing sample, position, or time axes."""
    if not masks:
        raise ValueError("routing masks must not be empty")
    expected_shape = masks[0].shape
    if len(expected_shape) != 3:
        raise ValueError("routing masks must have shape [batch, position, program]")
    if any(mask.shape != expected_shape for mask in masks):
        raise ValueError("all routing masks must have the same shape")
    stacked = torch.stack([mask.detach().float().cpu() for mask in masks])
    # [step, batch, position, program] -> aggregate position and recurrent time.
    sample_profiles = stacked.mean((0, 2))
    if sample_profiles.shape[0] < 2:
        sample = 0.0
    else:
        sample = float((sample_profiles[1:] - sample_profiles[:-1]).abs().mean())
    position = (
        float((stacked[:, :, 1:] - stacked[:, :, :-1]).abs().mean())
        if stacked.shape[2] > 1
        else 0.0
    )
    temporal = float((stacked[1:] - stacked[:-1]).abs().mean()) if len(masks) > 1 else 0.0
    return {"sample": sample, "position": position, "temporal": temporal}


def routing_variation(masks: list[torch.Tensor]) -> tuple[float, float]:
    metrics = routing_variation_metrics(masks)
    return metrics["sample"], metrics["temporal"]


def reset_program_routing_logits(
    model: SparseCellularTextNCA,
    *,
    seed: int,
    symmetry_breaking_scale: float = 1e-4,
) -> None:
    """Reset only program-head biases; cell logits and every executor weight stay untouched."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for stage in model.stages:
            bias = stage.receptor.out_proj.bias[1:]
            noise = torch.randn(bias.numel(), generator=generator, dtype=torch.float32)
            noise = (noise - noise.mean()) * symmetry_breaking_scale
            bias.copy_(noise.to(device=bias.device, dtype=bias.dtype))


@torch.no_grad()
def router_diagnostics(
    model: SparseCellularTextNCA,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float]:
    captured: list[torch.Tensor] = []
    handles = [
        stage.receptor.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[..., 1:].detach().float().cpu())
        )
        for stage in model.stages
    ]
    try:
        for starts in starts_batches:
            inputs, _ = batch_from_starts(stream, starts, sequence_length, device)
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    logits = torch.stack(captured)
    probabilities = logits.sigmoid()
    sample_profiles = logits.mean((0, 2))
    cross_sample_variance = float(sample_profiles.var(0, unbiased=False).mean())
    rankings = sample_profiles.argsort(dim=-1, descending=True)
    if rankings.shape[0] < 2:
        ranking_disagreement = 0.0
    else:
        ranking_disagreement = float((rankings[1:] != rankings[:-1]).float().mean())
    winners = logits.argmax(-1).reshape(-1)
    frequency = torch.bincount(winners, minlength=NUM_PROGRAMS).float()
    frequency = frequency / frequency.sum().clamp_min(1)
    nonzero = frequency > 0
    ranking_entropy = float(
        -(frequency[nonzero] * frequency[nonzero].log()).sum() / math.log(NUM_PROGRAMS)
    )
    return {
        "program_logit_mean": float(logits.mean()),
        "program_logit_std": float(logits.std(unbiased=False)),
        "program_probability_mean": float(probabilities.mean()),
        "program_probability_std": float(probabilities.std(unbiased=False)),
        "program_ranking_entropy": ranking_entropy,
        "mean_cross_sample_program_logit_variance": cross_sample_variance,
        "mean_pairwise_ranking_disagreement": ranking_disagreement,
    }


def configure_hard_program_stage(
    model: SparseCellularTextNCA,
    optimizer: torch.optim.Optimizer,
    *,
    top_k: int,
) -> torch.optim.Optimizer:
    """Configure a discrete continuation stage without replacing optimizer state."""
    model.set_routing_mode("hard_program")
    model.set_execution_backend("masked_dense")
    model.set_program_top_k(top_k)
    return optimizer


def minimum_quality_safe_k(
    progression: list[tuple[int, float]],
    *,
    quality_ratio_max: float = QUALITY_RATIO_THRESHOLD,
) -> int:
    """Return the last safe K, stopping logically at the first failed stage."""
    if not progression or progression[0][0] != NUM_PROGRAMS:
        raise ValueError("progression must begin at top-8")
    safe = NUM_PROGRAMS
    expected = NUM_PROGRAMS
    for top_k, ratio in progression:
        if top_k != expected:
            raise ValueError("progression must be contiguous and descending")
        if ratio > quality_ratio_max:
            break
        safe = top_k
        expected -= 1
    return safe


def quality_gated_progression(
    progression: list[tuple[int, float]],
    *,
    quality_ratio_max: float = QUALITY_RATIO_THRESHOLD,
) -> list[int]:
    """Return stages that execute through the first failure; later stages must not run."""
    executed: list[int] = []
    for top_k, ratio in progression:
        executed.append(top_k)
        if ratio > quality_ratio_max:
            break
    return executed


@torch.no_grad()
def evaluate_arm(
    model: SparseCellularTextNCA,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
    arm: Arm,
    top_k: int,
    static_mask: torch.Tensor | None = None,
    permutation_seed: int = 0,
) -> EvaluationResult:
    model.eval()
    model.set_program_top_k(top_k)
    model.set_routing_mode("dense" if arm == "dense" else "hard_program")
    model.set_execution_backend("sparse_dispatch" if arm != "dense" else "masked_dense")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    elapsed = 0.0
    total_loss = 0.0
    total_tokens = 0
    dense_flops = receptor_flops = active_flops = 0
    usage: list[torch.Tensor] = []
    coactivation: list[torch.Tensor] = []
    structural: list[float] = []
    position: list[float] = []
    temporal: list[float] = []
    generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, sequence_length, device)
        if arm == "dense":
            batch_started = time.perf_counter()
            output, stats = model(inputs, return_stats=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - batch_started
            masks: list[torch.Tensor] = []
        elif arm == "dynamic":
            batch_started = time.perf_counter()
            with RoutingRecorder(model) as recorder:
                output, stats = model(inputs, return_stats=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - batch_started
            masks = recorder.masks
        else:
            if arm == "static":
                if static_mask is None:
                    raise ValueError("static arm requires static_mask")
                masks = expand_static_mask(model, inputs, static_mask)
            else:
                masks = collect_dynamic_masks(model, inputs)
                permutation = torch.randperm(inputs.shape[0], generator=generator)
                if inputs.shape[0] > 1 and torch.equal(permutation, torch.arange(inputs.shape[0])):
                    permutation = permutation.roll(1)
                masks = shuffled_masks(masks, permutation)
            batch_started = time.perf_counter()
            with replay_routing(model, masks):
                output, stats = model(inputs, return_stats=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - batch_started
        if arm != "dense":
            variation = routing_variation_metrics(masks)
            structural.append(variation["sample"])
            position.append(variation["position"])
            temporal.append(variation["temporal"])
        total_loss += float(
            F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        total_tokens += targets.numel()
        dense_flops += stats.dense_executor_flops
        receptor_flops += stats.receptor_flops
        active_flops += stats.active_executor_flops
        usage.append(stats.program_usage)
        coactivation.append(stats.program_coactivation)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    nll = total_loss / total_tokens
    batches = len(starts_batches)
    return EvaluationResult(
        arm, top_k, nll, math.exp(min(nll, 20)), dense_flops, receptor_flops, active_flops,
        active_flops / dense_flops, (receptor_flops + active_flops) / dense_flops,
        total_tokens / elapsed, elapsed * 1000 / batches,
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        torch.stack(usage).mean(0), torch.stack(coactivation).mean(0),
        sum(structural) / len(structural) if structural else 0.0,
        sum(position) / len(position) if position else 0.0,
        sum(temporal) / len(temporal) if temporal else 0.0,
    )


def make_validation_001b_decision(
    arms_rows: list[dict[str, object]],
    progression_rows: list[dict[str, object]],
    *,
    router_warmup_ok: dict[int, bool] | None = None,
) -> dict[str, object]:
    by_replicate: dict[int, dict[str, dict[str, object]]] = {}
    for row in arms_rows:
        by_replicate.setdefault(int(row["replicate"]), {})[str(row["arm"])] = row
    safe_k = {
        int(row["replicate"]): int(row["quality_safe_k"])
        for row in progression_rows
        if bool(row.get("selected", False))
    }
    evidence: list[dict[str, object]] = []
    success_count = safe_count = variation_count = causal_count = 0
    for replicate, arms in sorted(by_replicate.items()):
        if set(arms) != {"dense", "dynamic", "static", "shuffled"}:
            raise ValueError(f"replicate {replicate} is missing one or more final arms")
        if replicate not in safe_k:
            raise ValueError(f"replicate {replicate} has no selected quality-safe K")
        dense = arms["dense"]
        dynamic = arms["dynamic"]
        quality_ratio = float(dynamic["validation_ppl"]) / float(dense["validation_ppl"])
        static_advantage = (
            float(arms["static"]["validation_nll"]) - float(dynamic["validation_nll"])
        ) / float(dense["validation_nll"])
        shuffled_advantage = (
            float(arms["shuffled"]["validation_nll"]) - float(dynamic["validation_nll"])
        ) / float(dense["validation_nll"])
        safe = safe_k[replicate] <= 6 and quality_ratio <= QUALITY_RATIO_THRESHOLD
        varied = float(dynamic["structural_variation"]) >= CONDITIONALITY_THRESHOLD
        causal = (
            static_advantage >= NORMALIZED_ADVANTAGE_THRESHOLD
            and shuffled_advantage >= NORMALIZED_ADVANTAGE_THRESHOLD
        )
        receptor_ok = float(dynamic["receptor_ratio"]) <= RECEPTOR_RATIO_THRESHOLD
        passed = safe and varied and causal and receptor_ok
        safe_count += safe
        variation_count += varied
        causal_count += causal
        success_count += passed
        evidence.append(
            {
                "replicate": replicate,
                "quality_safe_k": safe_k[replicate],
                "quality_ratio": quality_ratio,
                "sample_variation": float(dynamic["structural_variation"]),
                "temporal_variation": float(dynamic["temporal_variation"]),
                "static_advantage": static_advantage,
                "shuffled_advantage": shuffled_advantage,
                "receptor_ratio": float(dynamic["receptor_ratio"]),
                "passed": passed,
            }
        )
    warmup_failures = sum(not value for value in (router_warmup_ok or {}).values())
    if warmup_failures >= 2:
        diagnosis = "CLM_ROUTER_WARMUP_FAILURE"
    elif success_count >= 2:
        diagnosis = "CLM_PROGRAM_CONDITIONALITY_SIGNAL"
    elif safe_count < 2:
        diagnosis = "CLM_NO_QUALITY_SAFE_SPARSITY"
    elif variation_count < 2:
        diagnosis = "CLM_STATIC_PRUNING_ONLY"
    elif causal_count < 2:
        diagnosis = "CLM_DYNAMIC_ROUTING_WITHOUT_CAUSAL_VALUE"
    else:
        diagnosis = "CLM_STATIC_PRUNING_ONLY"
    return {
        "format": "minicells.clm-validation-001b.v1",
        "experiment": "CLM Validation 001b — Stable Program Conditionality",
        "status": "PASS" if diagnosis == "CLM_PROGRAM_CONDITIONALITY_SIGNAL" else "FAIL",
        "diagnosis": diagnosis,
        "successful_replicates": success_count,
        "thresholds": {
            "quality_ratio_max": QUALITY_RATIO_THRESHOLD,
            "required_safe_k_max": 6,
            "sample_variation_min": CONDITIONALITY_THRESHOLD,
            "normalized_advantage_min": NORMALIZED_ADVANTAGE_THRESHOLD,
            "receptor_ratio_max": RECEPTOR_RATIO_THRESHOLD,
        },
        "evidence": evidence,
    }


def make_validation_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    k4 = [row for row in rows if int(row["top_k"]) == 4]
    by_replicate: dict[int, dict[str, dict[str, object]]] = {}
    for row in k4:
        by_replicate.setdefault(int(row["replicate"]), {})[str(row["arm"])] = row
    quality = causal = variation = sparse = 0
    evidence: list[dict[str, object]] = []
    for replicate, arms in sorted(by_replicate.items()):
        required = {"dense", "dynamic", "static", "shuffled"}
        if set(arms) != required:
            raise ValueError(f"replicate {replicate} is missing validation arms")
        dense, dynamic = arms["dense"], arms["dynamic"]
        quality_ratio = float(dynamic["validation_ppl"]) / float(dense["validation_ppl"])
        static_advantage = (float(arms["static"]["validation_nll"]) -
                            float(dynamic["validation_nll"])) / float(dense["validation_nll"])
        shuffle_advantage = (float(arms["shuffled"]["validation_nll"]) -
                             float(dynamic["validation_nll"])) / float(dense["validation_nll"])
        has_quality = quality_ratio <= QUALITY_RATIO_THRESHOLD
        has_variation = float(dynamic["structural_variation"]) >= CONDITIONALITY_THRESHOLD
        has_causal = (static_advantage >= NORMALIZED_ADVANTAGE_THRESHOLD and
                      shuffle_advantage >= NORMALIZED_ADVANTAGE_THRESHOLD)
        has_sparse = float(dynamic["executor_ratio"]) <= 0.5
        quality += has_quality
        variation += has_variation
        causal += has_causal
        sparse += has_sparse
        evidence.append({"replicate": replicate, "quality_ratio": quality_ratio,
                         "static_advantage": static_advantage,
                         "shuffle_advantage": shuffle_advantage,
                         "structural_variation": float(dynamic["structural_variation"])})
    if quality >= 2 and variation >= 2 and causal >= 2:
        diagnosis = "CLM_PROGRAM_CONDITIONALITY_SIGNAL"
    elif quality >= 2 and sparse >= 2:
        diagnosis = "CLM_PROGRAM_SPARSITY_ONLY"
    else:
        diagnosis = "CLM_PROGRAM_SPARSITY_QUALITY_FAILURE"
    return {
        "format": "minicells.clm-validation-001.v1",
        "experiment": "CLM Validation 001 — Program Conditionality",
        "status": "PASS" if diagnosis != "CLM_PROGRAM_SPARSITY_QUALITY_FAILURE" else "FAIL",
        "diagnosis": diagnosis,
        "replicates": len(by_replicate),
        "counts": {"quality": quality, "non_static": variation, "causal": causal,
                   "executor_sparse": sparse},
        "thresholds": {"quality_ppl_ratio": QUALITY_RATIO_THRESHOLD,
                       "structural_variation": CONDITIONALITY_THRESHOLD,
                       "normalized_advantage": NORMALIZED_ADVANTAGE_THRESHOLD,
                       "receptor_ratio": RECEPTOR_RATIO_THRESHOLD},
        "evidence": evidence,
    }
