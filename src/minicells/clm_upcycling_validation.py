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
from .upcycled_cellular_textnca import UpcycledCellularTextNCA


UpcyclingArm = Literal["dynamic", "static", "shuffled"]


@torch.no_grad()
def validate_upcycled_parity(
    teacher: TextNCALM,
    student: UpcycledCellularTextNCA,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
) -> dict[str, object]:
    teacher.eval()
    student.eval()
    student.set_execution_backend("masked_dense")
    teacher_loss = student_loss = 0.0
    tokens = 0
    max_logits = max_state = 0.0
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, sequence_length, device)
        teacher_states: list[torch.Tensor] = []
        student_states: list[torch.Tensor] = []
        teacher_hooks = [
            stage.gru.register_forward_hook(
                lambda _module, _inputs, output: teacher_states.append(output.detach().clone())
            )
            for stage in teacher.stages
        ]
        student_hooks = [
            stage.gru.register_forward_hook(
                lambda _module, _inputs, output: student_states.append(output.detach().clone())
            )
            for stage in student.stages
        ]
        expected = teacher(inputs)
        actual = student(inputs)
        for hook in [*teacher_hooks, *student_hooks]:
            hook.remove()
        max_logits = max(max_logits, float((expected.logits - actual.logits).abs().max()))
        for left, right in zip(teacher_states, student_states):
            max_state = max(max_state, float((left - right).abs().max()))
        teacher_loss += float(
            F.cross_entropy(expected.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        student_loss += float(
            F.cross_entropy(actual.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        tokens += targets.numel()
    teacher_nll = teacher_loss / tokens
    student_nll = student_loss / tokens
    ratio = math.exp(student_nll - teacher_nll)
    passed = abs(ratio - 1.0) <= 1e-5 and max_logits <= 5e-5 and max_state <= 1e-6
    return {
        "status": "CLM_UPCYCLING_EQUIVALENCE" if passed else "CLM_UPCYCLING_EQUIVALENCE_FAILURE",
        "teacher_nll": teacher_nll,
        "student_nll": student_nll,
        "teacher_ppl": math.exp(teacher_nll),
        "student_ppl": math.exp(student_nll),
        "ppl_ratio": ratio,
        "max_logits_abs_diff": max_logits,
        "max_recurrent_state_abs_diff": max_state,
    }


@torch.no_grad()
def collect_stage_perceptions(
    teacher: TextNCALM,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
    max_samples_per_stage: int = 8192,
    seed: int = 0,
) -> list[torch.Tensor]:
    teacher.eval()
    captured: list[list[torch.Tensor]] = [[] for _ in teacher.stages]
    handles = []
    for index, stage in enumerate(teacher.stages):
        def hook(_module, _inputs, output, index=index):
            captured[index].append(output.detach().float().cpu().reshape(-1, output.shape[-1]))
        handles.append(stage.norm_ffn.register_forward_hook(hook))
    try:
        for starts in starts_batches:
            inputs, _ = batch_from_starts(stream, starts, sequence_length, device)
            teacher(inputs)
    finally:
        for handle in handles:
            handle.remove()
    generator = torch.Generator().manual_seed(seed)
    samples: list[torch.Tensor] = []
    for stage_rows in captured:
        if not stage_rows:
            raise RuntimeError("geometry calibration captured no local perceptions")
        merged = torch.cat(stage_rows, dim=0)
        if merged.shape[0] > max_samples_per_stage:
            indices = torch.randperm(merged.shape[0], generator=generator)[:max_samples_per_stage]
            merged = merged.index_select(0, indices)
        samples.append(merged)
    return samples


def cosine_kmeans(
    samples: torch.Tensor,
    clusters: int,
    *,
    seed: int,
    iterations: int = 25,
) -> tuple[torch.Tensor, dict[str, object]]:
    if samples.ndim != 2 or samples.shape[0] < clusters:
        raise ValueError("cosine_kmeans requires [samples, dim] with samples >= clusters")
    values = F.normalize(samples.float(), dim=-1)
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(values.shape[0], generator=generator)[:clusters]
    centroids = values.index_select(0, initial).clone()
    assignments = torch.full((values.shape[0],), -1, dtype=torch.long)
    for _ in range(iterations):
        scores = values @ centroids.T
        updated_assignments = scores.argmax(dim=-1)
        if torch.equal(updated_assignments, assignments):
            assignments = updated_assignments
            break
        assignments = updated_assignments
        next_centroids = []
        for cluster in range(clusters):
            members = values[assignments == cluster]
            if members.numel() == 0:
                # Deterministic recovery: take the currently worst represented sample.
                best = scores.max(dim=-1).values
                replacement = values[best.argmin()].clone()
                next_centroids.append(replacement)
            else:
                next_centroids.append(F.normalize(members.mean(0), dim=0))
        centroids = torch.stack(next_centroids)
    final_scores = values @ centroids.T
    assignments = final_scores.argmax(dim=-1)
    occupancy = torch.bincount(assignments, minlength=clusters).float()
    occupancy = occupancy / occupancy.sum()
    cosine_distance = 1.0 - final_scores.gather(1, assignments[:, None]).mean()
    return centroids, {
        "samples": int(values.shape[0]),
        "mean_cosine_distance": float(cosine_distance),
        "occupancy": occupancy.tolist(),
        "occupancy_entropy": float(
            -(occupancy[occupancy > 0] * occupancy[occupancy > 0].log()).sum()
            / math.log(clusters)
        ),
    }


def geometry_prototypes(
    stage_samples: list[torch.Tensor],
    num_experts: int,
    *,
    seed: int,
) -> tuple[list[torch.Tensor], list[dict[str, object]]]:
    prototypes = []
    diagnostics = []
    for stage_index, samples in enumerate(stage_samples):
        centroids, row = cosine_kmeans(
            samples, num_experts, seed=seed + 97 * stage_index
        )
        prototypes.append(centroids)
        diagnostics.append({"stage": stage_index, **row})
    return prototypes, diagnostics


class UpcyclingRoutingRecorder:
    def __init__(self, model: UpcycledCellularTextNCA) -> None:
        self.model = model
        self.masks: list[torch.Tensor] = []
        self.originals: list[object] = []

    def __enter__(self):
        for stage in self.model.stages:
            bank = stage.program_bank
            original = bank.route
            self.originals.append(original)

            def wrapped(this: object, perception: torch.Tensor, original=original):
                gates, probabilities, logits = original(perception)
                self.masks.append(gates.detach().clone())
                return gates, probabilities, logits

            bank.route = MethodType(wrapped, bank)
        return self

    def __exit__(self, *_exc: object) -> None:
        for stage, original in zip(self.model.stages, self.originals):
            stage.program_bank.route = original


@contextlib.contextmanager
def replay_upcycling_masks(
    model: UpcycledCellularTextNCA,
    masks: list[torch.Tensor],
) -> Iterator[None]:
    cursor = 0
    originals = []
    for stage in model.stages:
        bank = stage.program_bank
        original = bank.route
        originals.append(original)

        def replay(this: object, perception: torch.Tensor, original=original):
            nonlocal cursor
            _, probabilities, logits = original(perception)
            gates = masks[cursor].to(perception)
            cursor += 1
            return gates, probabilities, logits

        bank.route = MethodType(replay, bank)
    try:
        yield
        if cursor != len(masks):
            raise RuntimeError("not all upcycling routing masks were consumed")
    finally:
        for stage, original in zip(model.stages, originals):
            stage.program_bank.route = original


def static_templates(
    mask_batches: list[list[torch.Tensor]],
) -> list[torch.Tensor]:
    if not mask_batches:
        raise ValueError("static templates require calibration masks")
    routes = len(mask_batches[0])
    if any(len(batch) != routes for batch in mask_batches):
        raise ValueError("calibration batches have inconsistent route counts")
    templates = []
    for route_index in range(routes):
        usage = torch.cat(
            [batch[route_index].reshape(-1, batch[route_index].shape[-1])
             for batch in mask_batches], dim=0
        ).mean(0)
        chosen = usage.argmax()
        templates.append(torch.zeros_like(usage).scatter(0, chosen.view(1), 1.0))
    return templates


def expand_templates(
    templates: list[torch.Tensor], inputs: torch.Tensor
) -> list[torch.Tensor]:
    return [
        template.to(inputs).view(1, 1, -1).expand(inputs.shape[0], inputs.shape[1], -1)
        for template in templates
    ]


@dataclass(frozen=True)
class UpcyclingEvaluation:
    arm: UpcyclingArm
    nll: float
    ppl: float
    sample_variation: float
    position_variation: float
    temporal_variation: float
    usage_entropy: float
    router_logit_variance: float
    program_usage: torch.Tensor
    program_coactivation: torch.Tensor
    tokens_per_second: float


@torch.no_grad()
def evaluate_upcycled(
    model: UpcycledCellularTextNCA,
    stream: torch.Tensor,
    starts_batches: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
    arm: UpcyclingArm,
    templates: list[torch.Tensor] | None = None,
    permutation_seed: int = 0,
) -> UpcyclingEvaluation:
    model.eval()
    model.set_execution_backend("sparse_dispatch")
    generator = torch.Generator().manual_seed(permutation_seed)
    total_loss = 0.0
    tokens = 0
    elapsed = 0.0
    variations = []
    usages = []
    coactivations = []
    entropies = []
    logit_variances = []
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, sequence_length, device)
        masks: list[torch.Tensor] = []
        if arm == "dynamic":
            started = time.perf_counter()
            with UpcyclingRoutingRecorder(model) as recorder:
                output, stats = model(inputs, return_stats=True)
            masks = recorder.masks
        elif arm == "static":
            if templates is None:
                raise ValueError("static evaluation requires templates")
            masks = expand_templates(templates, inputs)
            started = time.perf_counter()
            with replay_upcycling_masks(model, masks):
                output, stats = model(inputs, return_stats=True)
        else:
            with UpcyclingRoutingRecorder(model) as recorder:
                model(inputs)
            permutation = torch.randperm(inputs.shape[0], generator=generator)
            if torch.equal(permutation, torch.arange(inputs.shape[0])):
                permutation = permutation.roll(1)
            masks = [
                mask.index_select(0, permutation.to(mask.device)) for mask in recorder.masks
            ]
            started = time.perf_counter()
            with replay_upcycling_masks(model, masks):
                output, stats = model(inputs, return_stats=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - started
        total_loss += float(
            F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        tokens += targets.numel()
        variations.append(routing_variation_metrics(masks))
        usages.append(stats.program_usage.detach().cpu())
        coactivations.append(stats.program_coactivation.detach().cpu())
        entropies.append(float(stats.usage_entropy))
        logit_variances.append(float(stats.router_logit_variance))
    nll = total_loss / tokens
    return UpcyclingEvaluation(
        arm=arm,
        nll=nll,
        ppl=math.exp(min(nll, 20)),
        sample_variation=sum(row["sample"] for row in variations) / len(variations),
        position_variation=sum(row["position"] for row in variations) / len(variations),
        temporal_variation=sum(row["temporal"] for row in variations) / len(variations),
        usage_entropy=sum(entropies) / len(entropies),
        router_logit_variance=sum(logit_variances) / len(logit_variances),
        program_usage=torch.stack(usages).mean(0),
        program_coactivation=torch.stack(coactivations).mean(0),
        tokens_per_second=tokens / elapsed,
    )


def evaluation_dict(result: UpcyclingEvaluation) -> dict[str, object]:
    row = asdict(result)
    row["program_usage"] = result.program_usage.tolist()
    row["program_coactivation"] = result.program_coactivation.tolist()
    return row


def make_upcycling_decision(
    replicates: list[dict[str, object]],
    controls: list[dict[str, object]],
) -> dict[str, object]:
    if any(
        row.get("random_parity") != "CLM_UPCYCLING_EQUIVALENCE"
        or row.get("geometry_parity") != "CLM_UPCYCLING_EQUIVALENCE"
        for row in replicates
    ):
        diagnosis = "CLM_UPCYCLING_EQUIVALENCE_FAILURE"
        method_evidence: dict[str, list[dict[str, object]]] = {}
    else:
        by_key = {
            (int(row["replicate"]), str(row["method"]), str(row["arm"])): row
            for row in controls
        }
        method_evidence = {"copy_random": [], "copy_geometry": []}
        quality_counts = {"copy_random": 0, "copy_geometry": 0}
        causal_counts = {"copy_random": 0, "copy_geometry": 0}
        for replicate in replicates:
            r = int(replicate["replicate"])
            dense_nll = float(replicate["dense_nll"])
            dense_ppl = float(replicate["dense_ppl"])
            for method in method_evidence:
                dynamic = by_key[(r, method, "dynamic")]
                static = by_key[(r, method, "static")]
                shuffled = by_key[(r, method, "shuffled")]
                quality_ratio = float(dynamic["ppl"]) / dense_ppl
                static_advantage = (float(static["nll"]) - float(dynamic["nll"])) / dense_nll
                shuffled_advantage = (
                    float(shuffled["nll"]) - float(dynamic["nll"])
                ) / dense_nll
                quality = quality_ratio <= 1.03
                varied = float(dynamic["sample_variation"]) >= 0.05
                causal = static_advantage >= 0.002 and shuffled_advantage >= 0.002
                entropy_ok = float(dynamic["usage_entropy"]) >= 0.80
                passed = quality and varied and causal and entropy_ok
                quality_counts[method] += int(quality)
                causal_counts[method] += int(passed)
                method_evidence[method].append({
                    "replicate": r,
                    "quality_ratio_to_dense_continued": quality_ratio,
                    "sample_variation": float(dynamic["sample_variation"]),
                    "usage_entropy": float(dynamic["usage_entropy"]),
                    "static_advantage": static_advantage,
                    "shuffled_advantage": shuffled_advantage,
                    "passed": passed,
                })
        best_quality = max(quality_counts.values())
        best_causal = max(causal_counts.values())
        if best_causal >= 2:
            diagnosis = "CLM_UPCYCLING_CONDITIONALITY_SIGNAL"
        elif best_quality >= 2:
            diagnosis = "CLM_UPCYCLING_QUALITY_SIGNAL"
        else:
            diagnosis = "CLM_UPCYCLING_QUALITY_FAILURE"
    geometry_advantage = False
    if method_evidence:
        random_pass = sum(int(row["passed"]) for row in method_evidence["copy_random"])
        geometry_pass = sum(int(row["passed"]) for row in method_evidence["copy_geometry"])
        geometry_advantage = geometry_pass >= 2 and random_pass < 2
    return {
        "format": "minicells.clm-upcycling-study-001.v1",
        "experiment": "CLM Upcycling Study 001 — Inherit Then Differentiate",
        "status": "PASS" if diagnosis in (
            "CLM_UPCYCLING_QUALITY_SIGNAL",
            "CLM_UPCYCLING_CONDITIONALITY_SIGNAL",
        ) else "FAIL",
        "diagnosis": diagnosis,
        "geometry_advantage": geometry_advantage,
        "thresholds": {
            "quality_ratio_to_dense_continued_max": 1.03,
            "sample_variation_min": 0.05,
            "normalized_advantage_min": 0.002,
            "usage_entropy_min": 0.80,
        },
        "evidence": method_evidence,
    }
