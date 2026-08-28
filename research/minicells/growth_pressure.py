"""Deterministic, label-free pressure calibration for CLM-0.3."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch.nn import functional as F


MIN_ROUTED_PERCEPTIONS = 512
PREFERRED_ROUTED_PERCEPTIONS = 8192


@dataclass(frozen=True)
class PressureCandidate:
    stage: int
    expert_id: str
    usage: float
    grad_conflict: float
    pressure: float
    routed_samples: int
    eligible: bool

    def to_row(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "expert_id": self.expert_id,
            "usage": self.usage,
            "gradient_disagreement": self.grad_conflict,
            "pressure": self.pressure,
            "routed_samples": self.routed_samples,
            "eligible": "yes" if self.eligible else "no",
        }


def utilization(route_counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(max(0, int(value)) for value in route_counts.values())
    if total == 0:
        return {key: 0.0 for key in route_counts}
    return {key: max(0, int(value)) / total for key, value in route_counts.items()}


def gradient_disagreement(gradients: Iterable[torch.Tensor], *, eps: float = 1e-12) -> float:
    """Return ``1 - ||sum(g)|| / (sum(||g||) + eps)`` exactly as preregistered."""

    vectors = [gradient.detach().float().reshape(-1) for gradient in gradients]
    if not vectors:
        return 0.0
    if len({vector.numel() for vector in vectors}) != 1:
        raise ValueError("all gradient vectors must have the same size")
    stacked = torch.stack(vectors)
    numerator = stacked.sum(0).norm()
    denominator = stacked.norm(dim=1).sum() + eps
    return float((1.0 - numerator / denominator).clamp(0.0, 1.0))


def pressure_score(usage: float, grad_conflict: float) -> float:
    if usage < 0 or grad_conflict < 0:
        raise ValueError("pressure inputs must be non-negative")
    return float(usage * (1.0 + grad_conflict))


def rank_pressure_candidates(candidates: Iterable[PressureCandidate]) -> list[PressureCandidate]:
    """Rank eligible lineages reproducibly, with stable IDs as tie breakers."""

    return sorted(
        candidates,
        key=lambda item: (-item.pressure, -item.usage, item.stage, item.expert_id),
    )


def select_pressure_parent(candidates: Iterable[PressureCandidate]) -> PressureCandidate:
    ranked = [item for item in rank_pressure_candidates(candidates) if item.eligible]
    if not ranked:
        raise RuntimeError("NO_ELIGIBLE_GROWTH_PARENT")
    return ranked[0]


def select_random_parent(
    candidates: Iterable[PressureCandidate], *, seed: int
) -> PressureCandidate:
    eligible = sorted((item for item in candidates if item.eligible), key=lambda item: item.expert_id)
    if not eligible:
        raise RuntimeError("NO_ELIGIBLE_GROWTH_PARENT")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return eligible[int(torch.randint(len(eligible), (), generator=generator))]


def write_pressure_table(path: str | Path, candidates: Iterable[PressureCandidate]) -> None:
    rows = [item.to_row() for item in rank_pressure_candidates(candidates)]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "expert_id", "usage", "gradient_disagreement", "pressure", "routed_samples", "eligible"],
        )
        writer.writeheader()
        writer.writerows(rows)


def cosine_kmeans_2(
    perceptions: torch.Tensor,
    *,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Deterministic unlabeled cosine 2-means.

    Initialization uses the first normalized sample and its farthest sample,
    making geometry initialization independent of global RNG state.
    """

    if perceptions.ndim != 2 or perceptions.shape[0] < 2:
        raise ValueError("cosine k-means requires at least two [sample, dim] perceptions")
    x = F.normalize(perceptions.detach().float(), dim=-1)
    first = x[0]
    farthest = torch.argmin(x @ first)
    centroids = torch.stack((first, x[farthest]))
    for _ in range(max_iter):
        assignments = (x @ centroids.T).argmax(-1)
        updated = []
        for cluster in range(2):
            members = x[assignments == cluster]
            if members.numel() == 0:
                # This deterministic fallback preserves both branches without
                # introducing random split initialization.
                updated.append(centroids[cluster])
            else:
                updated.append(F.normalize(members.mean(0, keepdim=True), dim=-1).squeeze(0))
        new_centroids = torch.stack(updated)
        if torch.max((new_centroids - centroids).abs()) <= tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def make_pressure_candidates(
    *,
    stage_by_expert: Mapping[str, int],
    route_counts: Mapping[str, int],
    gradient_batches: Mapping[str, Iterable[torch.Tensor]],
    min_samples: int = MIN_ROUTED_PERCEPTIONS,
) -> list[PressureCandidate]:
    usage = utilization(route_counts)
    result = []
    for expert_id, stage in stage_by_expert.items():
        conflict = gradient_disagreement(gradient_batches.get(expert_id, ()))
        samples = int(route_counts.get(expert_id, 0))
        result.append(PressureCandidate(
            int(stage), expert_id, usage.get(expert_id, 0.0), conflict,
            pressure_score(usage.get(expert_id, 0.0), conflict), samples,
            samples >= min_samples,
        ))
    return result


def calibrate_model_pressure(
    model: object,
    microbatches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    min_samples: int = MIN_ROUTED_PERCEPTIONS,
) -> tuple[list[PressureCandidate], dict[str, torch.Tensor]]:
    """Calibrate utilization/conflict without retaining autograd graphs.

    ``model`` is intentionally duck-typed so this helper remains useful for
    tiny CPU fixtures as well as the full CLM.  Each backward pass is cleared
    before the next microbatch.
    """

    batches = list(microbatches)
    if not batches:
        raise ValueError("at least one pressure microbatch is required")
    banks = [stage.program_bank for stage in model.stages]
    for bank in banks:
        bank.begin_pressure_collection(cap=PREFERRED_ROUTED_PERCEPTIONS)
    gradients: dict[str, list[torch.Tensor]] = {}
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
                    pieces = []
                    for parameter in bank.experts[expert_id].parameters():
                        gradient = (
                            parameter.grad
                            if parameter.grad is not None
                            else torch.zeros_like(parameter)
                        )
                        pieces.append(gradient.detach().reshape(-1))
                    gradients.setdefault(expert_id, []).append(torch.cat(pieces))
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
        expert_id: bank.stage
        for bank in banks for expert_id in bank.expert_ids
    }
    candidates = make_pressure_candidates(
        stage_by_expert=stage_by_expert,
        route_counts=route_counts,
        gradient_batches=gradients,
        min_samples=min_samples,
    )
    perceptions = {
        expert_id: torch.cat(values) if values else torch.empty(0, model.stages[0].gru.hidden_size)
        for bank in banks for expert_id, values in bank.last_perceptions.items()
    }
    model.zero_grad(set_to_none=True)
    return rank_pressure_candidates(candidates), perceptions


compute_gradient_disagreement = gradient_disagreement
compute_pressure = pressure_score
select_parent = select_pressure_parent
