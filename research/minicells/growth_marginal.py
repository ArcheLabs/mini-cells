"""Marginal-capacity calibration and saturation diagnostics for CLM-0.3b."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F

from .growth_pressure import (
    MIN_ROUTED_PERCEPTIONS,
    PREFERRED_ROUTED_PERCEPTIONS,
    cosine_kmeans_2,
    gradient_disagreement,
    pressure_score,
    utilization,
)


@dataclass(frozen=True)
class MarginalCandidate:
    stage: int
    expert_id: str
    usage: float
    gradient_disagreement: float
    legacy_pressure: float
    fisher_per_route: float
    weight_grad_saliency: float
    geometry_separation: float
    marginal_score: float
    routed_samples: int
    eligible: bool

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["eligible"] = "yes" if self.eligible else "no"
        return row


@dataclass(frozen=True)
class SaturationResult:
    detected: bool
    token: int | None
    observations: int
    window_start_token: int | None
    window_end_token: int | None
    slope_log_ppl_per_100k: float | None
    projected_improvement_500k: float | None
    endpoint_ratio: float | None
    threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def geometry_separation(perceptions: torch.Tensor) -> float:
    """Return cosine two-cluster separation in [0, 1]."""

    if perceptions.ndim != 2 or perceptions.shape[0] < 2:
        return 0.0
    prototypes = cosine_kmeans_2(perceptions)
    cosine = float(F.cosine_similarity(prototypes[0:1], prototypes[1:2]).item())
    return float(max(0.0, min(1.0, 0.5 * (1.0 - cosine))))


def marginal_capacity_score(
    fisher_per_route: float,
    weight_grad_saliency: float,
    geometry: float,
) -> float:
    """Preregistered CLM-0.3b WHERE score.

    Sensitivity is the geometric mean of a Fisher-like gradient-energy term and
    weight-gradient saliency. Geometry is a bounded multiplier rather than the
    primary importance signal: a parent must both matter to the objective and
    admit a meaningful local split.
    """

    if fisher_per_route < 0 or weight_grad_saliency < 0 or not 0.0 <= geometry <= 1.0:
        raise ValueError("marginal score inputs are outside their valid domains")
    sensitivity = math.sqrt(fisher_per_route * weight_grad_saliency)
    return float(sensitivity * (0.5 + 0.5 * geometry))


def rank_marginal_candidates(candidates: Iterable[MarginalCandidate]) -> list[MarginalCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.marginal_score,
            -item.fisher_per_route,
            -item.weight_grad_saliency,
            item.stage,
            item.expert_id,
        ),
    )


def select_marginal_parent(candidates: Iterable[MarginalCandidate]) -> MarginalCandidate:
    ranked = [item for item in rank_marginal_candidates(candidates) if item.eligible]
    if not ranked:
        raise RuntimeError("NO_ELIGIBLE_GROWTH_PARENT")
    return ranked[0]


def select_random_parent(candidates: Iterable[MarginalCandidate], *, seed: int) -> MarginalCandidate:
    eligible = sorted((item for item in candidates if item.eligible), key=lambda item: item.expert_id)
    if not eligible:
        raise RuntimeError("NO_ELIGIBLE_GROWTH_PARENT")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return eligible[int(torch.randint(len(eligible), (), generator=generator))]


def write_marginal_table(path: str | Path, candidates: Iterable[MarginalCandidate]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"rank": rank, **item.to_row()}
        for rank, item in enumerate(rank_marginal_candidates(candidates), start=1)
    ]
    fieldnames = [
        "rank",
        "stage",
        "expert_id",
        "usage",
        "gradient_disagreement",
        "legacy_pressure",
        "fisher_per_route",
        "weight_grad_saliency",
        "geometry_separation",
        "marginal_score",
        "routed_samples",
        "eligible",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def calibrate_marginal_candidates(
    model: Any,
    microbatches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    min_samples: int = MIN_ROUTED_PERCEPTIONS,
) -> tuple[list[MarginalCandidate], dict[str, torch.Tensor]]:
    """Measure marginal-capacity signals on recent routed training data.

    The calibration does not mutate optimizer state. Routed perceptions are
    intentionally retained on CPU by the program banks.
    """

    batches = list(microbatches)
    if not batches:
        raise ValueError("at least one marginal-utility microbatch is required")
    banks = [stage.program_bank for stage in model.stages]
    for bank in banks:
        bank.begin_pressure_collection(cap=PREFERRED_ROUTED_PERCEPTIONS)

    gradient_vectors: dict[str, list[torch.Tensor]] = {}
    grad_energy: dict[str, list[float]] = {}
    saliency: dict[str, list[float]] = {}
    parameter_counts: dict[str, int] = {}
    was_training = model.training
    model.train()
    try:
        for inputs, targets in batches:
            model.zero_grad(set_to_none=True)
            output = model(inputs, execution_backend="masked_dense")
            loss = F.cross_entropy(output.logits.flatten(0, 1), targets.reshape(-1))
            loss.backward()
            for bank in banks:
                for expert_id in bank.expert_ids:
                    pieces: list[torch.Tensor] = []
                    energy = 0.0
                    wg = 0.0
                    count = 0
                    for parameter in bank.experts[expert_id].parameters():
                        gradient = parameter.grad
                        if gradient is None:
                            gradient = torch.zeros_like(parameter)
                        detached = gradient.detach().float()
                        pieces.append(detached.reshape(-1).cpu())
                        energy += float(detached.square().sum().item())
                        wg += float((parameter.detach().float() * detached).abs().sum().item())
                        count += parameter.numel()
                    gradient_vectors.setdefault(expert_id, []).append(torch.cat(pieces))
                    grad_energy.setdefault(expert_id, []).append(energy)
                    saliency.setdefault(expert_id, []).append(wg / max(count, 1))
                    parameter_counts[expert_id] = count
    finally:
        for bank in banks:
            bank.end_pressure_collection()
        if not was_training:
            model.eval()

    route_counts = {
        expert_id: int(bank.last_route_counts.get(expert_id, 0))
        for bank in banks
        for expert_id in bank.expert_ids
    }
    stage_by_expert = {
        expert_id: int(bank.stage)
        for bank in banks
        for expert_id in bank.expert_ids
    }
    usages = utilization(route_counts)
    perceptions = {
        expert_id: torch.cat(values) if values else torch.empty(0, model.stages[0].gru.hidden_size)
        for bank in banks
        for expert_id, values in bank.last_perceptions.items()
    }

    candidates: list[MarginalCandidate] = []
    for expert_id, stage in stage_by_expert.items():
        usage = float(usages.get(expert_id, 0.0))
        gradients = gradient_vectors.get(expert_id, [])
        conflict = gradient_disagreement(gradients)
        mean_energy = sum(grad_energy.get(expert_id, ())) / max(len(grad_energy.get(expert_id, ())), 1)
        fisher = mean_energy / max(usage, 1e-12)
        mean_saliency = sum(saliency.get(expert_id, ())) / max(len(saliency.get(expert_id, ())), 1)
        routed = int(route_counts.get(expert_id, 0))
        geometry = geometry_separation(perceptions.get(expert_id, torch.empty(0, 1))) if routed >= min_samples else 0.0
        score = marginal_capacity_score(fisher, mean_saliency, geometry)
        candidates.append(MarginalCandidate(
            stage=stage,
            expert_id=expert_id,
            usage=usage,
            gradient_disagreement=conflict,
            legacy_pressure=pressure_score(usage, conflict),
            fisher_per_route=fisher,
            weight_grad_saliency=mean_saliency,
            geometry_separation=geometry,
            marginal_score=score,
            routed_samples=routed,
            eligible=routed >= min_samples,
        ))

    model.zero_grad(set_to_none=True)
    return rank_marginal_candidates(candidates), perceptions


def detect_saturation(
    rows: Iterable[dict[str, object]],
    *,
    min_tokens: int = 1_500_000,
    window: int = 5,
    projected_horizon_tokens: int = 500_000,
    max_projected_improvement: float = 0.005,
    endpoint_tolerance: float = 0.01,
) -> SaturationResult:
    """Detect a preregistered low-slope regime from evaluation PPL history.

    A five-point log-PPL regression spans 400K tokens at the formal 100K eval
    cadence. Saturation requires projected improvement over the next 500K to be
    at most 0.5%, while the latest point may not be >1% worse than the first
    point in the window. A temporary catastrophic regression therefore cannot
    masquerade as a plateau.
    """

    ordered = sorted(
        (
            (int(row["tokens"]), float(row["ppl"]))
            for row in rows
            if row.get("tokens") is not None and row.get("ppl") is not None
        ),
        key=lambda item: item[0],
    )
    if len(ordered) < window or ordered[-1][0] < min_tokens:
        return SaturationResult(
            False, None, len(ordered), None, None, None, None, None,
            max_projected_improvement,
        )
    selected = ordered[-window:]
    x = torch.tensor([(token - selected[0][0]) / 100_000.0 for token, _ in selected], dtype=torch.float64)
    y = torch.tensor([math.log(ppl) for _, ppl in selected], dtype=torch.float64)
    centered = x - x.mean()
    denom = float(centered.square().sum().item())
    slope = float((centered * (y - y.mean())).sum().item() / max(denom, 1e-12))
    horizon_units = projected_horizon_tokens / 100_000.0
    projected_improvement = 0.0 if slope >= 0 else 1.0 - math.exp(slope * horizon_units)
    endpoint_ratio = selected[-1][1] / selected[0][1]
    detected = (
        projected_improvement <= max_projected_improvement
        and endpoint_ratio <= 1.0 + endpoint_tolerance
    )
    return SaturationResult(
        detected=bool(detected),
        token=selected[-1][0] if detected else None,
        observations=len(ordered),
        window_start_token=selected[0][0],
        window_end_token=selected[-1][0],
        slope_log_ppl_per_100k=slope,
        projected_improvement_500k=float(projected_improvement),
        endpoint_ratio=float(endpoint_ratio),
        threshold=max_projected_improvement,
    )


@torch.no_grad()
def mergeback_bootstrap_ci(
    model: Any,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    stage: int,
    child_id: str,
    seed: int,
    bootstrap_samples: int = 1000,
) -> dict[str, float | int]:
    """Deterministic batch bootstrap for newborn causal merge-back penalty."""

    if not batches:
        raise ValueError("at least one validation batch is required")
    was_training = model.training
    model.eval()
    dynamic: list[float] = []
    merged: list[float] = []
    try:
        for inputs, targets in batches:
            output = model(inputs, execution_backend="sparse_dispatch")
            merged_output = model(
                inputs,
                execution_backend="sparse_dispatch",
                merge_back=(stage, child_id),
            )
            dynamic.append(float(F.cross_entropy(output.logits.flatten(0, 1), targets.reshape(-1))))
            merged.append(float(F.cross_entropy(merged_output.logits.flatten(0, 1), targets.reshape(-1))))
    finally:
        if was_training:
            model.train()

    d = torch.tensor(dynamic, dtype=torch.float64)
    m = torch.tensor(merged, dtype=torch.float64)
    point = float((m.mean() - d.mean()) / d.mean().clamp_min(1e-12))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        len(batches),
        (int(bootstrap_samples), len(batches)),
        generator=generator,
    )
    sampled_d = d[indices].mean(dim=1)
    sampled_m = m[indices].mean(dim=1)
    penalties = (sampled_m - sampled_d) / sampled_d.clamp_min(1e-12)
    lower, upper = torch.quantile(
        penalties,
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    ).tolist()
    return {
        "causal_merge_back_penalty_bootstrap": point,
        "causal_merge_back_ci95_low": float(lower),
        "causal_merge_back_ci95_high": float(upper),
        "causal_merge_back_batches": len(batches),
        "causal_merge_back_bootstrap_samples": int(bootstrap_samples),
    }
