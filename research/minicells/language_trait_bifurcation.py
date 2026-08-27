from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .language_conflict_differentiation import (
    DOMAINS,
    IDENTITY_NORMALIZED_MARGIN_MIN,
    ROUTING_PURITY_MIN,
    summarize_identity,
)

ARMS = ("unified", "stratified-capacity-fork", "geometry-bifurcation-fork")
BIFURCATION_GAIN_MIN = 0.20
BIFURCATION_SPLIT_BALANCE_MIN = 0.25
BIFURCATION_AXIS_STABILITY_MIN = 0.50
BIFURCATION_WINDOWS_MIN = 2
GEOMETRY_MARGIN_ADVANTAGE_MIN = 0.05
POSITIVE_REPLICATES_MIN = 2
KMEANS_STEPS = 32


@dataclass(frozen=True)
class BifurcationGeometry:
    one_mode_centroid: torch.Tensor
    centroids: torch.Tensor
    axis: torch.Tensor
    residual_k1: float
    residual_k2: float
    bifurcation_gain: float
    split_balance: float
    centroid_separation: float


@dataclass(frozen=True)
class PersistentBifurcationSummary:
    windows_passed: int
    axis_stability: float
    combined_gain: float
    combined_split_balance: float
    persistent: bool


def _canonical_axis(axis: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis.float(), dim=0, eps=1e-8)
    pivot = int(axis.abs().argmax().item())
    if float(axis[pivot]) < 0.0:
        axis = -axis
    return axis


def fit_two_mode_gradient_field(gradients: torch.Tensor) -> BifurcationGeometry:
    """Fit deterministic task-label-free K=1 and K=2 models to phenotype gradients."""
    if gradients.ndim != 2 or gradients.shape[0] < 4:
        raise ValueError("gradients must be [microbatches, phenotype_dim]")
    if not torch.isfinite(gradients).all():
        raise ValueError("bifurcation gradients must be finite")

    unit = F.normalize(gradients.float(), dim=1, eps=1e-8)
    mean = unit.mean(dim=0)
    residual_k1_tensor = (unit - mean).square().sum()

    centered = unit - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    seed_axis = _canonical_axis(vh[0])
    projection = centered @ seed_axis
    median = projection.median()
    assignment = (projection < median).long()
    if int((assignment == 0).sum()) == 0 or int((assignment == 1).sum()) == 0:
        order = torch.argsort(projection)
        assignment = torch.zeros(len(unit), dtype=torch.long, device=unit.device)
        assignment[order[len(unit) // 2 :]] = 1

    centroids = torch.stack(
        [unit[assignment == cluster].mean(dim=0) for cluster in (0, 1)], dim=0
    )
    for _ in range(KMEANS_STEPS):
        distance = (unit[:, None, :] - centroids[None, :, :]).square().sum(dim=-1)
        updated = distance.argmin(dim=1)
        if int((updated == 0).sum()) == 0 or int((updated == 1).sum()) == 0:
            break
        new_centroids = torch.stack(
            [unit[updated == cluster].mean(dim=0) for cluster in (0, 1)], dim=0
        )
        if torch.equal(updated, assignment):
            assignment = updated
            centroids = new_centroids
            break
        assignment = updated
        centroids = new_centroids

    if float(torch.dot(centroids[0] - centroids[1], seed_axis)) < 0.0:
        centroids = centroids.flip(0)
        assignment = 1 - assignment

    residual_k2_tensor = (unit - centroids[assignment]).square().sum()
    residual_k1 = float(residual_k1_tensor)
    residual_k2 = float(residual_k2_tensor)
    gain = (residual_k1 - residual_k2) / max(residual_k1, 1e-12)
    fractions = [float((assignment == cluster).float().mean()) for cluster in (0, 1)]
    axis = F.normalize(centroids[0] - centroids[1], dim=0, eps=1e-8)

    return BifurcationGeometry(
        one_mode_centroid=mean.detach(),
        centroids=centroids.detach(),
        axis=axis.detach(),
        residual_k1=residual_k1,
        residual_k2=residual_k2,
        bifurcation_gain=float(gain),
        split_balance=min(fractions),
        centroid_separation=float((centroids[0] - centroids[1]).norm()),
    )


def bifurcation_window_pass(geometry: BifurcationGeometry) -> bool:
    return bool(
        geometry.bifurcation_gain >= BIFURCATION_GAIN_MIN
        and geometry.split_balance >= BIFURCATION_SPLIT_BALANCE_MIN
    )


def axis_stability(geometries: list[BifurcationGeometry]) -> float:
    if len(geometries) < 2:
        raise ValueError("axis stability requires at least two windows")
    values = []
    for left in range(len(geometries)):
        for right in range(left + 1, len(geometries)):
            values.append(
                abs(
                    float(
                        torch.dot(
                            F.normalize(geometries[left].axis.float(), dim=0, eps=1e-8),
                            F.normalize(geometries[right].axis.float(), dim=0, eps=1e-8),
                        )
                    )
                )
            )
    return float(min(values))


def summarize_persistent_bifurcation(
    windows: list[BifurcationGeometry],
    combined: BifurcationGeometry,
) -> PersistentBifurcationSummary:
    passed = sum(int(bifurcation_window_pass(window)) for window in windows)
    stability = axis_stability(windows)
    persistent = bool(
        passed >= BIFURCATION_WINDOWS_MIN
        and stability >= BIFURCATION_AXIS_STABILITY_MIN
        and bifurcation_window_pass(combined)
    )
    return PersistentBifurcationSummary(
        windows_passed=passed,
        axis_stability=stability,
        combined_gain=combined.bifurcation_gain,
        combined_split_balance=combined.split_balance,
        persistent=persistent,
    )


def route_gradient_to_mode(
    gradient: torch.Tensor,
    geometry: BifurcationGeometry,
) -> tuple[int, float]:
    unit = F.normalize(gradient.detach().float(), dim=0, eps=1e-8)
    centroids = geometry.centroids.to(unit.device)
    distances = (centroids - unit[None, :]).square().sum(dim=1)
    branch = int(distances.argmin().item())
    score = float(distances[1] - distances[0])
    return branch, score


def routing_purity_from_branches(branches: list[int], labels: list[str]) -> float:
    if len(branches) != len(labels) or not branches:
        raise ValueError("branches and labels must align")
    predicted = np.asarray(branches, dtype=np.int64)
    truth = np.asarray([0 if label == DOMAINS[0] else 1 for label in labels], dtype=np.int64)
    direct = float((predicted == truth).mean())
    swapped = float(((1 - predicted) == truth).mean())
    return max(direct, swapped)


def geometry_advantage(geometry_identity_margin: float, capacity_identity_margin: float) -> float:
    return float(geometry_identity_margin - capacity_identity_margin)
