from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Iterator, Literal

import torch
from torch.nn import functional as F

from .language_clm_validation import routing_variation_metrics
from .language_data import batch_from_starts
from .language_models import TextNCALM
from .overcomplete_cellular_textnca import OvercompleteCellularTextNCA


V2Arm = Literal["dense", "dynamic", "static", "shuffled"]


@torch.no_grad()
def validate_v2_scaffold_parity(
    teacher: TextNCALM,
    student: OvercompleteCellularTextNCA,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
) -> dict[str, object]:
    student.set_scaffold_alpha(1)
    teacher.eval()
    student.eval()
    teacher_loss = student_loss = 0.0
    tokens = 0
    max_logits = max_state = 0.0
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, sequence_length, device)
        teacher_states: list[torch.Tensor] = []
        student_states: list[torch.Tensor] = []
        teacher_hooks = [stage.gru.register_forward_hook(
            lambda _module, _inputs, output: teacher_states.append(output.detach().clone())
        ) for stage in teacher.stages]
        student_hooks = [stage.gru.register_forward_hook(
            lambda _module, _inputs, output: student_states.append(output.detach().clone())
        ) for stage in student.stages]
        expected = teacher(inputs)
        actual = student(inputs)
        for hook in [*teacher_hooks, *student_hooks]:
            hook.remove()
        max_logits = max(max_logits, float((expected.logits - actual.logits).abs().max()))
        for left, right in zip(teacher_states, student_states):
            max_state = max(max_state, float((left - right).abs().max()))
        teacher_loss += float(F.cross_entropy(
            expected.logits.flatten(0, 1), targets.flatten(), reduction="sum"
        ))
        student_loss += float(F.cross_entropy(
            actual.logits.flatten(0, 1), targets.flatten(), reduction="sum"
        ))
        tokens += targets.numel()
    teacher_nll = teacher_loss / tokens
    student_nll = student_loss / tokens
    ratio = math.exp(student_nll - teacher_nll)
    passed = abs(ratio - 1) <= 1e-5 and max_logits <= 5e-5 and max_state <= 1e-6
    return {
        "status": "CLMV2_SCAFFOLD_EQUIVALENCE" if passed
        else "CLMV2_SCAFFOLD_EQUIVALENCE_FAILURE",
        "teacher_nll": teacher_nll, "student_nll": student_nll,
        "teacher_ppl": math.exp(teacher_nll), "student_ppl": math.exp(student_nll),
        "ppl_ratio": ratio, "max_logits_abs_diff": max_logits,
        "max_recurrent_state_abs_diff": max_state,
    }


@dataclass(frozen=True)
class V2Evaluation:
    arm: V2Arm
    top_k: int
    nll: float
    ppl: float
    sample_variation: float
    position_variation: float
    temporal_variation: float
    program_usage: torch.Tensor
    program_coactivation: torch.Tensor
    receptor_ratio: float
    active_ffn_ratio: float
    effective_ffn_ratio: float
    tokens_per_second: float


class V2RoutingRecorder:
    def __init__(self, model: OvercompleteCellularTextNCA) -> None:
        self.model = model
        self.masks: list[torch.Tensor] = []
        self.originals: list[object] = []

    def __enter__(self):
        for stage in self.model.stages:
            bank = stage.program_bank
            original = bank.route
            self.originals.append(original)

            def wrapped(this: object, perception: torch.Tensor, original: object = original):
                gates, probabilities, logits = original(perception)
                self.masks.append(gates.detach().clone())
                return gates, probabilities, logits

            bank.route = MethodType(wrapped, bank)
        return self

    def __exit__(self, *_exc: object) -> None:
        for stage, original in zip(self.model.stages, self.originals):
            stage.program_bank.route = original


@contextlib.contextmanager
def replay_v2_masks(
    model: OvercompleteCellularTextNCA,
    masks: list[torch.Tensor],
) -> Iterator[None]:
    cursor = 0
    originals = []
    for stage in model.stages:
        bank = stage.program_bank
        original = bank.route
        originals.append(original)

        def replay(this: object, perception: torch.Tensor, original: object = original):
            nonlocal cursor
            _, probabilities, logits = original(perception)
            gates = masks[cursor].to(perception)
            cursor += 1
            return gates, probabilities, logits

        bank.route = MethodType(replay, bank)
    try:
        yield
        if cursor != len(masks):
            raise RuntimeError("not all v2 routing masks were consumed")
    finally:
        for stage, original in zip(model.stages, originals):
            stage.program_bank.route = original


def static_mask(mask_batches: list[list[torch.Tensor]], top_k: int) -> torch.Tensor:
    usage = torch.cat(
        [mask.reshape(-1, mask.shape[-1]) for batch in mask_batches for mask in batch]
    ).mean(0)
    indices = usage.topk(top_k).indices
    return torch.zeros_like(usage).scatter(0, indices, 1.0)


@torch.no_grad()
def v2_router_diagnostics(
    model: OvercompleteCellularTextNCA,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
) -> dict[str, object]:
    captured = []
    handles = [stage.program_bank.receptor.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().float().cpu())
    ) for stage in model.stages]
    masks = []
    try:
        for starts in starts_batches:
            inputs, _ = batch_from_starts(stream, starts, sequence_length, device)
            with V2RoutingRecorder(model) as recorder:
                model(inputs)
            masks.extend(recorder.masks)
    finally:
        for handle in handles:
            handle.remove()
    logits = torch.stack(captured)
    profiles = logits.mean((0, 2))
    rankings = profiles.argsort(-1, descending=True)
    disagreement = (
        float((rankings[1:] != rankings[:-1]).float().mean())
        if rankings.shape[0] > 1 else 0.0
    )
    hard = torch.cat([mask.reshape(-1, mask.shape[-1]) for mask in masks]).mean(0)
    soft = logits.sigmoid().reshape(-1, logits.shape[-1]).mean(0)
    frequency = hard / hard.sum().clamp_min(1)
    nonzero = frequency > 0
    entropy = float(
        -(frequency[nonzero] * frequency[nonzero].log()).sum() / math.log(hard.numel())
    )
    return {
        "mean_logits": float(logits.mean()),
        "logits_std": float(logits.std(unbiased=False)),
        "cross_sample_logit_variance": float(profiles.var(0, unbiased=False).mean()),
        "ranking_disagreement": disagreement,
        "usage_entropy": entropy,
        "hard_usage": hard.tolist(),
        "soft_usage": soft.tolist(),
    }


def expanded_static(model: OvercompleteCellularTextNCA, inputs: torch.Tensor,
                    mask: torch.Tensor) -> list[torch.Tensor]:
    expanded = mask.view(1, 1, -1).expand(inputs.shape[0], inputs.shape[1], -1)
    return [expanded for _ in range(sum(stage.iterations for stage in model.stages))]


@torch.no_grad()
def evaluate_v2(
    model: OvercompleteCellularTextNCA,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
    arm: V2Arm,
    top_k: int,
    fixed_mask: torch.Tensor | None = None,
    permutation_seed: int = 0,
) -> V2Evaluation:
    model.eval()
    model.set_program_top_k(top_k)
    model.set_scaffold_alpha(1 if arm == "dense" else 0)
    model.set_execution_backend("masked_dense" if arm == "dense" else "sparse_dispatch")
    generator = torch.Generator().manual_seed(permutation_seed)
    loss = tokens = shared = experts = receptor = 0
    elapsed = 0.0
    usages = []
    coactivations = []
    variations = []
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, sequence_length, device)
        masks: list[torch.Tensor] = []
        if arm == "dynamic":
            started = time.perf_counter()
            with V2RoutingRecorder(model) as recorder:
                output, stats = model(inputs, return_stats=True)
            masks = recorder.masks
        elif arm in ("static", "shuffled"):
            if arm == "static":
                if fixed_mask is None:
                    raise ValueError("static evaluation requires a fixed mask")
                masks = expanded_static(model, inputs, fixed_mask)
            else:
                with V2RoutingRecorder(model) as recorder:
                    model(inputs)
                permutation = torch.randperm(inputs.shape[0], generator=generator)
                if torch.equal(permutation, torch.arange(inputs.shape[0])):
                    permutation = permutation.roll(1)
                masks = [
                    mask.index_select(0, permutation.to(mask.device))
                    for mask in recorder.masks
                ]
            started = time.perf_counter()
            with replay_v2_masks(model, masks):
                output, stats = model(inputs, return_stats=True)
        else:
            started = time.perf_counter()
            output, stats = model(inputs, return_stats=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - started
        loss += float(
            F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        tokens += targets.numel()
        shared += stats.shared_flops
        experts += stats.expert_flops
        receptor += stats.receptor_flops
        usages.append(stats.program_usage.detach().cpu())
        coactivations.append(stats.program_coactivation.detach().cpu())
        if masks:
            variations.append(routing_variation_metrics(masks))
    nll = loss / tokens
    dense_ffn = shared + linear_expert_dense_equivalent(experts, top_k)
    return V2Evaluation(
        arm, top_k, nll, math.exp(min(nll, 20)),
        sum(row["sample"] for row in variations) / len(variations) if variations else 0.0,
        sum(row["position"] for row in variations) / len(variations) if variations else 0.0,
        sum(row["temporal"] for row in variations) / len(variations) if variations else 0.0,
        torch.stack(usages).mean(0), torch.stack(coactivations).mean(0),
        receptor / dense_ffn, (shared + experts) / dense_ffn,
        (shared + experts + receptor) / dense_ffn, tokens / elapsed,
    )


def linear_expert_dense_equivalent(expert_flops: int, top_k: int) -> int:
    # Expert width is 64; six experts plus the 128-wide shared path equal width 512.
    return round(expert_flops * 6 / top_k)


def evaluation_dict(result: V2Evaluation) -> dict[str, object]:
    row = asdict(result)
    row["program_usage"] = result.program_usage.tolist()
    row["program_coactivation"] = result.program_coactivation.tolist()
    return row


def make_v2_decision(
    workers: list[dict[str, object]],
    arms: list[dict[str, object]],
    *,
    teacher_nll: float,
) -> dict[str, object]:
    if any(row["status"] == "CLMV2_SCAFFOLD_EQUIVALENCE_FAILURE" for row in workers):
        diagnosis = "CLMV2_SCAFFOLD_EQUIVALENCE_FAILURE"
        successes = 0
    elif sum(row["status"] == "CLMV2_LOCAL_IMITATION_FAILURE" for row in workers) >= 2:
        diagnosis = "CLMV2_LOCAL_IMITATION_FAILURE"
        successes = 0
    elif sum(row["status"] == "CLMV2_SCAFFOLD_HANDOFF_FAILURE" for row in workers) >= 2:
        diagnosis = "CLMV2_SCAFFOLD_HANDOFF_FAILURE"
        successes = 0
    else:
        grouped: dict[int, dict[str, dict[str, object]]] = {}
        for row in arms:
            grouped.setdefault(int(row["replicate"]), {})[str(row["arm"])] = row
        successes = 0
        handoffs = 0
        safe_capacity = 0
        varied = 0
        causal = 0
        evidence = []
        for worker in workers:
            replicate = int(worker["replicate"])
            if worker["status"] != "CLMV2_SCAFFOLD_HANDOFF_SIGNAL":
                continue
            handoffs += 1
            replicate_arms = grouped.get(replicate, {})
            if set(replicate_arms) != {"dense", "dynamic", "static", "shuffled"}:
                continue
            dynamic = replicate_arms["dynamic"]
            top_k = int(worker["quality_safe_k"])
            quality = float(dynamic["ppl"]) / math.exp(teacher_nll)
            static_advantage = (
                float(replicate_arms["static"]["nll"]) - float(dynamic["nll"])
            ) / teacher_nll
            shuffled_advantage = (
                float(replicate_arms["shuffled"]["nll"]) - float(dynamic["nll"])
            ) / teacher_nll
            is_safe = top_k <= 5 and quality <= 1.03
            is_varied = float(dynamic["sample_variation"]) >= 0.05
            is_causal = static_advantage >= 0.002 and shuffled_advantage >= 0.002
            receptor_ok = float(dynamic["receptor_ratio"]) <= 0.05
            passed = is_safe and is_varied and is_causal and receptor_ok
            safe_capacity += is_safe
            varied += is_varied
            causal += is_causal
            successes += passed
            evidence.append({
                "replicate": replicate, "quality_safe_k": top_k,
                "quality_ratio_to_teacher": quality,
                "sample_variation": float(dynamic["sample_variation"]),
                "static_advantage": static_advantage,
                "shuffled_advantage": shuffled_advantage,
                "receptor_ratio": float(dynamic["receptor_ratio"]), "passed": passed,
            })
        if successes >= 2:
            diagnosis = "CLMV2_PROGRAM_CONDITIONALITY_SIGNAL"
        elif safe_capacity >= 2:
            diagnosis = "CLMV2_CONDITIONAL_CAPACITY_WITHOUT_CAUSAL_ROUTING"
        elif handoffs >= 2:
            diagnosis = "CLMV2_SCAFFOLD_HANDOFF_SIGNAL"
        else:
            diagnosis = "CLMV2_SCAFFOLD_HANDOFF_FAILURE"
    result = {
        "format": "minicells.clm-v2-validation-001.v1",
        "experiment": "CLM v2 Validation 001 — Scaffold Handoff",
        "status": "PASS" if diagnosis in (
            "CLMV2_SCAFFOLD_HANDOFF_SIGNAL",
            "CLMV2_CONDITIONAL_CAPACITY_WITHOUT_CAUSAL_ROUTING",
            "CLMV2_PROGRAM_CONDITIONALITY_SIGNAL",
        ) else "FAIL",
        "diagnosis": diagnosis,
        "successful_replicates": successes,
        "strong_program_sparsity": sum(
            int(row.get("quality_safe_k", 12)) <= 4 for row in workers
        ) >= 2,
        "thresholds": {
            "handoff_ratio_max": 1.03, "quality_safe_k_max": 5,
            "sample_variation_min": 0.05, "normalized_advantage_min": 0.002,
            "receptor_ratio_max": 0.05,
        },
    }
    if "evidence" in locals():
        result["evidence"] = evidence
    return result
