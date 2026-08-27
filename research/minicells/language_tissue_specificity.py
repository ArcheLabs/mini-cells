from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .language_localized_learning import LocalizedLearningState, conservative_fork


TISSUE_ARMS = ("one-cell", "three-cell-chain")
ONE_CELL_SIZE = 1
THREE_CELL_SIZE = 3
SPECIFICITY_NORM_MIN = 0.10
EXAMPLE_TOP1_MIN = 0.50
REPLICATE_TOP1_MIN = 2
GENERAL_FAMILIES_MIN = 4
RETENTION_RATIO_MAX = 1.10
CAUSAL_FRACTION_MIN = 0.90
TRANSPLANT_RECOVERY_MIN = 0.90


@dataclass(frozen=True)
class SpecificitySummary:
    matching_value: float
    mean_wrong_value: float
    best_wrong_value: float
    specificity: float
    normalized_specificity: float
    strict_margin: float
    matching_rank: int


def summarize_specificity(candidate_names: list[str], values: np.ndarray, matching: str) -> SpecificitySummary:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(candidate_names):
        raise ValueError("candidate_names and values must be aligned")
    if matching not in candidate_names:
        raise ValueError("matching candidate is missing")
    if not np.isfinite(values).all():
        raise ValueError("specificity values must be finite")
    match_index = candidate_names.index(matching)
    matching_value = float(values[match_index])
    wrong = np.delete(values, match_index)
    mean_wrong = float(wrong.mean())
    best_wrong = float(wrong.max())
    specificity = matching_value - mean_wrong
    normalized = specificity / max(abs(matching_value), 1e-6)
    strict_margin = matching_value - best_wrong
    order = np.argsort(-values, kind="stable")
    rank = int(np.flatnonzero(order == match_index)[0]) + 1
    return SpecificitySummary(
        matching_value=matching_value,
        mean_wrong_value=mean_wrong,
        best_wrong_value=best_wrong,
        specificity=specificity,
        normalized_specificity=normalized,
        strict_margin=strict_margin,
        matching_rank=rank,
    )


def family_pass(
    normalized_specificity: float,
    replicate_top1_count: int,
    example_top1: float,
) -> bool:
    return bool(
        normalized_specificity >= SPECIFICITY_NORM_MIN
        and replicate_top1_count >= REPLICATE_TOP1_MIN
        and example_top1 >= EXAMPLE_TOP1_MIN
    )


def _orthogonal_direction(direction: torch.Tensor) -> torch.Tensor:
    vector = direction.detach().float()
    vector = vector / vector.norm().clamp_min(1e-8)
    axis = int(vector.abs().argmin().item())
    basis = torch.zeros_like(vector)
    basis[axis] = 1.0
    orthogonal = basis - torch.dot(basis, vector) * vector
    return orthogonal / orthogonal.norm().clamp_min(1e-8)


@torch.no_grad()
def allocate_fixed_tissue(
    model,
    state: LocalizedLearningState,
    probe,
    *,
    tissue_size: int,
    step: int = 0,
) -> tuple[int, list[int]]:
    """Allocate a fixed skill tissue without enabling autonomous topology changes.

    The base parent is selected identically for both arms from the same frozen-organism
    rewrite-pressure probe. One-cell receives one bidirectional parent-child edge.
    Three-cell receives a protected chain parent<->c1<->c2<->c3. The old phenotype is
    never modified. All later learning is restricted to the newborn memories.
    """
    if tissue_size not in (ONE_CELL_SIZE, THREE_CELL_SIZE):
        raise ValueError("Experiment 020 supports tissue sizes 1 and 3 only")
    base = state.base_alive.detach().cpu().clone()
    base &= model.alive_mask.detach().cpu()
    base[0] = False
    if not bool(base.any()):
        raise RuntimeError("fixed tissue allocation requires a non-interface base cell")
    cells = torch.nonzero(base, as_tuple=False).flatten()
    parent = int(cells[probe.pressure[cells].argmax()].item())
    primary = probe.split_direction[parent].detach().clone()
    orthogonal = _orthogonal_direction(primary)
    directions = (primary, orthogonal, -primary)

    newborn: list[int] = []
    current_parent = parent
    for index in range(tissue_size):
        child = conservative_fork(
            model,
            current_parent,
            step=step,
            direction=directions[index],
        )
        if child is None:
            raise RuntimeError(f"failed to allocate fixed tissue cell {index + 1}/{tissue_size}")
        newborn.append(child)
        current_parent = child
    return parent, newborn
