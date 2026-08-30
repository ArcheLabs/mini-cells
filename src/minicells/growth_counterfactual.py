"""Counterfactual mitosis utilities for CLM-0.3c.

The analytic score is intentionally only a proxy. Formal WHEN/WHERE decisions
are made from realized short-horizon shadow continuations.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.nn import functional as F

from .growth_pressure import MIN_ROUTED_PERCEPTIONS, PREFERRED_ROUTED_PERCEPTIONS, cosine_kmeans_2


@dataclass(frozen=True)
class SplitRegretCandidate:
    stage: int
    expert_id: str
    usage: float
    routed_samples: int
    cluster0_samples: int
    cluster1_samples: int
    pi0: float
    pi1: float
    geometry_separation: float
    adam_metric_disagreement: float
    split_regret: float
    eligible: bool

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["eligible"] = "yes" if self.eligible else "no"
        return row


@dataclass(frozen=True)
class PairedUtility:
    control_nll: float
    candidate_nll: float
    delta_nll: float
    relative_improvement: float
    ci95_low: float
    ci95_high: float
    bootstrap_samples: int
    batches: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def geometry_separation(prototypes: torch.Tensor) -> float:
    if prototypes.shape != (2, prototypes.shape[-1]):
        raise ValueError("expected exactly two prototypes")
    cosine = float(F.cosine_similarity(prototypes[0:1], prototypes[1:2]).item())
    return float(max(0.0, min(1.0, 0.5 * (1.0 - cosine))))


def split_regret_score(
    *,
    usage: float,
    pi0: float,
    pi1: float,
    gradient0: torch.Tensor,
    gradient1: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """Return Adam-metric gradient disagreement and route-weighted split regret."""

    if gradient0.shape != gradient1.shape or gradient0.shape != exp_avg_sq.shape:
        raise ValueError("gradient and Adam metric vectors must have the same shape")
    if usage < 0 or pi0 < 0 or pi1 < 0 or pi0 + pi1 <= 0:
        raise ValueError("invalid split-regret masses")
    delta = gradient0.double() - gradient1.double()
    metric = exp_avg_sq.double().clamp_min(0).sqrt().add(float(eps))
    disagreement = float((delta.square() / metric).mean().item())
    regret = float(usage * pi0 * pi1 * disagreement)
    return disagreement, regret


def _optimizer_second_moment(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        value = state.get("exp_avg_sq")
        if value is None:
            value = torch.zeros_like(parameter)
        pieces.append(value.detach().float().reshape(-1).cpu())
    return torch.cat(pieces) if pieces else torch.empty(0)


def _expert_gradient(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        pieces.append(gradient.detach().float().reshape(-1).cpu())
    return torch.cat(pieces) if pieces else torch.empty(0)


def _assign_counts(perceptions: torch.Tensor, prototypes: torch.Tensor) -> tuple[int, int]:
    if perceptions.numel() == 0:
        return 0, 0
    values = F.normalize(perceptions.float(), dim=-1)
    centers = F.normalize(prototypes.float(), dim=-1)
    choices = (values @ centers.T).argmax(dim=-1)
    counts = torch.bincount(choices, minlength=2)
    return int(counts[0].item()), int(counts[1].item())


def calibrate_split_regret(
    model: Any,
    optimizer: torch.optim.Optimizer,
    microbatches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    min_samples: int = MIN_ROUTED_PERCEPTIONS,
    min_cluster_samples: int = 128,
) -> tuple[list[SplitRegretCandidate], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Estimate split regret using route-conditioned microbatch gradients.

    Per-microbatch expert gradients are divided by that expert's routed mass.
    The prospective geometry split supplies cluster weights; route-count-weighted
    gradient means then approximate the two gradients that would be untied by a
    birth. No optimizer state is mutated.
    """

    batches = list(microbatches)
    if not batches:
        raise ValueError("at least one calibration microbatch is required")

    banks = [stage.program_bank for stage in model.stages]
    expert_meta = {
        expert_id: (int(bank.stage), bank, list(bank.experts[expert_id].parameters()))
        for bank in banks
        for expert_id in bank.expert_ids
    }
    second_moments = {
        expert_id: _optimizer_second_moment(optimizer, parameters)
        for expert_id, (_stage, _bank, parameters) in expert_meta.items()
    }
    batch_records: dict[str, list[dict[str, object]]] = {expert_id: [] for expert_id in expert_meta}
    all_perceptions: dict[str, list[torch.Tensor]] = {expert_id: [] for expert_id in expert_meta}
    total_routes: dict[str, int] = {expert_id: 0 for expert_id in expert_meta}

    was_training = model.training
    model.train()
    try:
        for bank in banks:
            bank.begin_pressure_collection(cap=PREFERRED_ROUTED_PERCEPTIONS)
        for inputs, targets in batches:
            for bank in banks:
                bank.reset_pressure_window()
            model.zero_grad(set_to_none=True)
            output = model(inputs, execution_backend="masked_dense")
            loss = F.cross_entropy(output.logits.flatten(0, 1), targets.reshape(-1))
            loss.backward()
            for expert_id, (_stage, bank, parameters) in expert_meta.items():
                routed = int(bank.last_route_counts.get(expert_id, 0))
                perceptions = bank.last_perceptions.get(expert_id, [])
                perception = torch.cat(perceptions, dim=0) if perceptions else torch.empty(0, inputs.shape[-1] if inputs.ndim > 2 else model.stages[0].gru.hidden_size)
                gradient = _expert_gradient(parameters)
                per_route_gradient = gradient / max(routed, 1)
                batch_records[expert_id].append({
                    "routed": routed,
                    "gradient": per_route_gradient,
                    "perceptions": perception,
                })
                total_routes[expert_id] += routed
                if perception.numel():
                    all_perceptions[expert_id].append(perception)
    finally:
        for bank in banks:
            bank.end_pressure_collection()
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()

    total_all_routes = max(sum(total_routes.values()), 1)
    candidates: list[SplitRegretCandidate] = []
    prototypes_by_expert: dict[str, torch.Tensor] = {}
    perceptions_by_expert: dict[str, torch.Tensor] = {}

    for expert_id, (stage, _bank, _parameters) in expert_meta.items():
        perception = (
            torch.cat(all_perceptions[expert_id], dim=0)
            if all_perceptions[expert_id]
            else torch.empty(0, model.stages[0].gru.hidden_size)
        )
        perceptions_by_expert[expert_id] = perception
        routed = int(total_routes[expert_id])
        eligible_geometry = routed >= min_samples and perception.shape[0] >= min_samples
        if eligible_geometry:
            prototypes = cosine_kmeans_2(perception)
        else:
            prototypes = torch.zeros(2, model.stages[0].gru.hidden_size)
        prototypes_by_expert[expert_id] = prototypes

        weighted0: torch.Tensor | None = None
        weighted1: torch.Tensor | None = None
        count0 = 0
        count1 = 0
        if eligible_geometry:
            for record in batch_records[expert_id]:
                batch_perception = record["perceptions"]
                assert isinstance(batch_perception, torch.Tensor)
                c0, c1 = _assign_counts(batch_perception, prototypes)
                gradient = record["gradient"]
                assert isinstance(gradient, torch.Tensor)
                if weighted0 is None:
                    weighted0 = torch.zeros_like(gradient)
                    weighted1 = torch.zeros_like(gradient)
                weighted0 += gradient * c0
                weighted1 += gradient * c1
                count0 += c0
                count1 += c1

        eligible = bool(
            eligible_geometry
            and count0 >= min_cluster_samples
            and count1 >= min_cluster_samples
            and second_moments[expert_id].numel() > 0
        )
        if eligible:
            assert weighted0 is not None and weighted1 is not None
            gradient0 = weighted0 / count0
            gradient1 = weighted1 / count1
            pi0 = count0 / max(count0 + count1, 1)
            pi1 = count1 / max(count0 + count1, 1)
            disagreement, regret = split_regret_score(
                usage=routed / total_all_routes,
                pi0=pi0,
                pi1=pi1,
                gradient0=gradient0,
                gradient1=gradient1,
                exp_avg_sq=second_moments[expert_id],
            )
            geometry = geometry_separation(prototypes)
        else:
            pi0 = count0 / max(count0 + count1, 1)
            pi1 = count1 / max(count0 + count1, 1)
            disagreement = 0.0
            regret = 0.0
            geometry = geometry_separation(prototypes) if eligible_geometry else 0.0

        candidates.append(SplitRegretCandidate(
            stage=stage,
            expert_id=expert_id,
            usage=routed / total_all_routes,
            routed_samples=routed,
            cluster0_samples=count0,
            cluster1_samples=count1,
            pi0=pi0,
            pi1=pi1,
            geometry_separation=geometry,
            adam_metric_disagreement=disagreement,
            split_regret=regret,
            eligible=eligible,
        ))

    candidates.sort(key=lambda item: (-item.split_regret, item.stage, item.expert_id))
    return candidates, perceptions_by_expert, prototypes_by_expert


def write_split_regret_table(path: str | Path, candidates: Iterable[SplitRegretCandidate]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"rank": rank, **item.to_row()} for rank, item in enumerate(candidates, start=1)]
    fieldnames = [
        "rank", "stage", "expert_id", "usage", "routed_samples", "cluster0_samples",
        "cluster1_samples", "pi0", "pi1", "geometry_separation",
        "adam_metric_disagreement", "split_regret", "eligible",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def paired_bootstrap_utility(
    control_batch_nlls: Sequence[float],
    candidate_batch_nlls: Sequence[float],
    *,
    seed: int,
    bootstrap_samples: int = 2000,
) -> PairedUtility:
    if len(control_batch_nlls) != len(candidate_batch_nlls) or not control_batch_nlls:
        raise ValueError("paired utility requires equally sized non-empty batch losses")
    control = torch.tensor(control_batch_nlls, dtype=torch.float64)
    candidate = torch.tensor(candidate_batch_nlls, dtype=torch.float64)
    control_mean = control.mean()
    candidate_mean = candidate.mean()
    relative = (control_mean - candidate_mean) / control_mean.clamp_min(1e-12)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        len(control_batch_nlls),
        (int(bootstrap_samples), len(control_batch_nlls)),
        generator=generator,
    )
    sampled_control = control[indices].mean(dim=1)
    sampled_candidate = candidate[indices].mean(dim=1)
    sampled_relative = (sampled_control - sampled_candidate) / sampled_control.clamp_min(1e-12)
    low, high = torch.quantile(
        sampled_relative,
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    ).tolist()
    return PairedUtility(
        control_nll=float(control_mean.item()),
        candidate_nll=float(candidate_mean.item()),
        delta_nll=float((control_mean - candidate_mean).item()),
        relative_improvement=float(relative.item()),
        ci95_low=float(low),
        ci95_high=float(high),
        bootstrap_samples=int(bootstrap_samples),
        batches=len(control_batch_nlls),
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in range(position, end):
            ranks[indexed[index][0]] = rank
        position = end
    return ranks


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Spearman correlation requires paired vectors with at least two entries")
    rx = torch.tensor(_average_ranks(xs), dtype=torch.float64)
    ry = torch.tensor(_average_ranks(ys), dtype=torch.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.square().sum().sqrt() * ry.square().sum().sqrt()
    if float(denom) == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def select_counterfactual_action(probe_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in probe_rows if row.get("eligible", True)]
    if not eligible:
        raise RuntimeError("NO_COUNTERFACTUAL_CANDIDATE")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["ci95_low"]),
            float(row["relative_improvement"]),
            -int(row.get("analytic_rank", 10**9)),
        ),
    )
    lcb = float(selected["ci95_low"])
    return {
        "action": "GROW" if lcb > 0.0 else "NO_GROW",
        "selected_expert": str(selected["expert_id"]),
        "selected_stage": int(selected["stage"]),
        "probe_relative_improvement": float(selected["relative_improvement"]),
        "probe_ci95_low": lcb,
        "probe_ci95_high": float(selected["ci95_high"]),
        "analytic_rank": int(selected.get("analytic_rank", -1)),
    }
