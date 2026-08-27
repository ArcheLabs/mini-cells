from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


CONDITIONS = (
    "STORY_ONLY",
    "DUPLICATED_STORY",
    "STORY_ARITHMETIC",
    "WEAK_ARITHMETIC",
)
ARMS = ("parent", "capacity-shadow", "geometry-shadow")
PROPOSAL_BATCHES = 64
PROBATION_WINDOWS = 4
STEPS_PER_WINDOW = 64
PROBATION_STEPS = PROBATION_WINDOWS * STEPS_PER_WINDOW
STRUCTURAL_COST_FRACTION = 0.005
GEOMETRY_ADVANTAGE_MIN = 0.005
ROUTING_PURITY_MIN = 0.75
IDENTITY_NORMALIZED_MARGIN_MIN = 0.01
POSITIVE_REPLICATES_MIN = 2


@dataclass(frozen=True)
class ProbationDecision:
    geometry_window_net_utility: tuple[float, ...]
    capacity_window_net_utility: tuple[float, ...]
    geometry_advantage: tuple[float, ...]
    sustained_positive: bool
    cumulative_positive: bool
    beats_capacity: bool
    accepted: bool


def condition_counts(condition: str, *, steps: int = PROBATION_STEPS) -> dict[str, int]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if condition == "STORY_ONLY":
        return {"STORY_A": steps}
    if condition == "DUPLICATED_STORY":
        left = steps // 2
        return {"STORY_A": left, "STORY_B": steps - left}
    if condition == "STORY_ARITHMETIC":
        left = steps // 2
        return {"STORY_A": left, "ARITHMETIC": steps - left}
    if condition == "WEAK_ARITHMETIC":
        arithmetic = max(1, int(round(0.10 * steps)))
        return {"STORY_A": steps - arithmetic, "ARITHMETIC": arithmetic}
    raise ValueError(f"unknown probation condition: {condition}")


def condition_schedule(condition: str, *, replicate: int, steps: int = PROBATION_STEPS) -> tuple[str, ...]:
    counts = condition_counts(condition, steps=steps)
    rows = [name for name, count in counts.items() for _ in range(count)]
    rng = random.Random(323_100 + 10_000 * replicate + 97 * CONDITIONS.index(condition) + steps)
    rng.shuffle(rows)
    return tuple(rows)


def semantic_family(stream_key: str) -> str:
    if stream_key.startswith("STORY"):
        return "STORY"
    if stream_key == "ARITHMETIC":
        return "ARITHMETIC"
    raise ValueError(stream_key)


def capacity_branch(step: int, replicate: int) -> int:
    """Task-agnostic matched-compute control: exactly alternating descendants."""
    if step < 0:
        raise ValueError("step must be non-negative")
    return int((step + replicate) % 2)


def normalized_net_utility(
    parent_loss: float,
    candidate_loss: float,
    *,
    structural_cost_fraction: float = STRUCTURAL_COST_FRACTION,
) -> float:
    if not np.isfinite(parent_loss) or not np.isfinite(candidate_loss):
        raise ValueError("probation losses must be finite")
    if structural_cost_fraction < 0.0:
        raise ValueError("structural cost must be non-negative")
    gain = (float(parent_loss) - float(candidate_loss)) / max(abs(float(parent_loss)), 1e-8)
    return float(gain - structural_cost_fraction)


def summarize_probation(
    parent_window_losses: list[float] | tuple[float, ...],
    capacity_window_losses: list[float] | tuple[float, ...],
    geometry_window_losses: list[float] | tuple[float, ...],
    *,
    structural_cost_fraction: float = STRUCTURAL_COST_FRACTION,
    geometry_advantage_min: float = GEOMETRY_ADVANTAGE_MIN,
) -> ProbationDecision:
    if not (
        len(parent_window_losses)
        == len(capacity_window_losses)
        == len(geometry_window_losses)
        == PROBATION_WINDOWS
    ):
        raise ValueError(f"probation requires exactly {PROBATION_WINDOWS} aligned windows")
    geometry = tuple(
        normalized_net_utility(parent, candidate, structural_cost_fraction=structural_cost_fraction)
        for parent, candidate in zip(parent_window_losses, geometry_window_losses)
    )
    capacity = tuple(
        normalized_net_utility(parent, candidate, structural_cost_fraction=structural_cost_fraction)
        for parent, candidate in zip(parent_window_losses, capacity_window_losses)
    )
    advantage = tuple(float(g - c) for g, c in zip(geometry, capacity))
    sustained = bool(geometry[-1] > 0.0 and geometry[-2] > 0.0)
    cumulative = bool(float(np.mean(geometry[-3:])) > 0.0)
    beats_capacity = bool(float(np.mean(advantage[-3:])) >= geometry_advantage_min)
    accepted = bool(sustained and cumulative and beats_capacity)
    return ProbationDecision(
        geometry_window_net_utility=geometry,
        capacity_window_net_utility=capacity,
        geometry_advantage=advantage,
        sustained_positive=sustained,
        cumulative_positive=cumulative,
        beats_capacity=beats_capacity,
        accepted=accepted,
    )


def expected_condition_outcome(condition: str) -> str:
    if condition in ("STORY_ONLY", "DUPLICATED_STORY"):
        return "REJECT"
    if condition == "STORY_ARITHMETIC":
        return "ACCEPT"
    if condition == "WEAK_ARITHMETIC":
        return "DISCOVER"
    raise ValueError(condition)
