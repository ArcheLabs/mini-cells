from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .language_probationary_trait_genesis import (
    GEOMETRY_ADVANTAGE_MIN,
    PROBATION_STEPS,
    PROBATION_WINDOWS,
    ROUTING_PURITY_MIN,
    STEPS_PER_WINDOW,
    STRUCTURAL_COST_FRACTION,
    summarize_probation,
)


STAGES = (
    "A_STORY_NULL",
    "B_ARITHMETIC_BIRTH",
    "C_DUPLICATE_ARITHMETIC",
    "D_WEAK_TRANSFORM",
    "E_TRANSFORM_BIRTH",
)
MAX_TRAITS = 3
PROPOSAL_BATCHES = 64
POSITIVE_REPLICATES_MIN = 2
IDENTITY_NORMALIZED_MARGIN_MIN = 0.01


@dataclass(frozen=True)
class StageSpec:
    name: str
    counts: dict[str, int]
    expected_outcome: str
    expected_start_k: int
    expected_end_k: int


@dataclass(frozen=True)
class StageDecision:
    accepted: bool
    expected_outcome: str
    start_k: int
    end_k: int
    geometry_mean_net_utility_last3: float
    capacity_mean_net_utility_last3: float
    geometry_advantage_last3: float
    sustained_positive: bool
    cumulative_positive: bool
    beats_capacity: bool


def stage_spec(stage: str, *, steps: int = PROBATION_STEPS) -> StageSpec:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if stage == "A_STORY_NULL":
        counts = {"STORY": steps}
        return StageSpec(stage, counts, "REJECT", 1, 1)
    if stage == "B_ARITHMETIC_BIRTH":
        story = steps // 2
        counts = {"STORY": story, "ARITH_A": steps - story}
        return StageSpec(stage, counts, "ACCEPT", 1, 2)
    if stage == "C_DUPLICATE_ARITHMETIC":
        story = steps // 2
        remaining = steps - story
        left = remaining // 2
        counts = {"STORY": story, "ARITH_A": left, "ARITH_B": remaining - left}
        return StageSpec(stage, counts, "REJECT", 2, 2)
    if stage == "D_WEAK_TRANSFORM":
        transform = max(1, int(round(0.10 * steps)))
        remaining = steps - transform
        story = remaining // 2
        counts = {"STORY": story, "ARITH_A": remaining - story, "TRANSFORM": transform}
        return StageSpec(stage, counts, "DISCOVER", 2, 2)
    if stage == "E_TRANSFORM_BIRTH":
        story = steps // 3 + (1 if steps % 3 > 0 else 0)
        arithmetic = steps // 3 + (1 if steps % 3 > 1 else 0)
        transform = steps - story - arithmetic
        counts = {"STORY": story, "ARITH_A": arithmetic, "TRANSFORM": transform}
        return StageSpec(stage, counts, "ACCEPT", 2, 3)
    raise ValueError(f"unknown stage: {stage}")


def stage_schedule(stage: str, *, replicate: int, steps: int = PROBATION_STEPS) -> tuple[str, ...]:
    spec = stage_spec(stage, steps=steps)
    values = [name for name, count in spec.counts.items() for _ in range(count)]
    rng = random.Random(324_000 + 10_000 * replicate + 131 * STAGES.index(stage) + steps)
    rng.shuffle(values)
    return tuple(values)


def semantic_family(stream_key: str) -> str:
    if stream_key == "STORY":
        return "STORY"
    if stream_key.startswith("ARITH"):
        return "ARITHMETIC"
    if stream_key == "TRANSFORM":
        return "TRANSFORM"
    raise ValueError(stream_key)


def capacity_shadow_branch(
    *,
    incumbent_branch: int,
    parent_branch: int,
    newborn_branch: int,
    occurrence: int,
    replicate: int,
) -> int:
    """Split only the proposed parent branch; leave every other incumbent branch unchanged."""
    if occurrence < 0:
        raise ValueError("occurrence must be non-negative")
    if incumbent_branch != parent_branch:
        return incumbent_branch
    return parent_branch if (occurrence + replicate) % 2 == 0 else newborn_branch


def summarize_stage_decision(
    *,
    stage: str,
    start_k: int,
    parent_window_losses: list[float] | tuple[float, ...],
    capacity_window_losses: list[float] | tuple[float, ...],
    geometry_window_losses: list[float] | tuple[float, ...],
    structural_cost_fraction: float = STRUCTURAL_COST_FRACTION,
    geometry_advantage_min: float = GEOMETRY_ADVANTAGE_MIN,
) -> StageDecision:
    probation = summarize_probation(
        parent_window_losses,
        capacity_window_losses,
        geometry_window_losses,
        structural_cost_fraction=structural_cost_fraction,
        geometry_advantage_min=geometry_advantage_min,
    )
    accepted = bool(probation.accepted and start_k < MAX_TRAITS)
    return StageDecision(
        accepted=accepted,
        expected_outcome=stage_spec(stage).expected_outcome,
        start_k=int(start_k),
        end_k=int(start_k + 1 if accepted else start_k),
        geometry_mean_net_utility_last3=float(np.mean(probation.geometry_window_net_utility[-3:])),
        capacity_mean_net_utility_last3=float(np.mean(probation.capacity_window_net_utility[-3:])),
        geometry_advantage_last3=float(np.mean(probation.geometry_advantage[-3:])),
        sustained_positive=bool(probation.sustained_positive),
        cumulative_positive=bool(probation.cumulative_positive),
        beats_capacity=bool(probation.beats_capacity),
    )


def expected_trajectory() -> tuple[int, ...]:
    return (1, 1, 2, 2, 2, 3)


def classify_replicate(stages: list[dict[str, object]]) -> dict[str, object]:
    by_name = {str(row["stage"]): row for row in stages}
    missing = [stage for stage in STAGES if stage not in by_name]
    if missing:
        raise ValueError(f"missing stages: {missing}")
    a = by_name["A_STORY_NULL"]
    b = by_name["B_ARITHMETIC_BIRTH"]
    c = by_name["C_DUPLICATE_ARITHMETIC"]
    d = by_name["D_WEAK_TRANSFORM"]
    e = by_name["E_TRANSFORM_BIRTH"]
    return {
        "story_null_reject": int(int(a["accepted"]) == 0 and int(a["end_k"]) == 1),
        "arithmetic_birth": int(
            int(b["accepted"]) == 1
            and int(b["start_k"]) == 1
            and int(b["end_k"]) == 2
            and int(b.get("identity_pass", 0) or 0) == 1
            and int(b.get("routing_purity_pass", 0) or 0) == 1
        ),
        "duplicate_reject": int(
            int(c["start_k"]) == 2
            and int(c["accepted"]) == 0
            and int(c["end_k"]) == 2
            and int(c.get("retention_identity_pass", 0) or 0) == 1
        ),
        "weak_transform_accepted": int(int(d["accepted"]) == 1),
        "transform_birth": int(
            int(e["accepted"]) == 1
            and int(e["start_k"]) == 2
            and int(e["end_k"]) == 3
            and int(e.get("identity_pass", 0) or 0) == 1
            and int(e.get("routing_purity_pass", 0) or 0) == 1
        ),
        "final_k": int(e["end_k"]),
    }


def aggregate_status(replicates: list[dict[str, object]]) -> str:
    if len(replicates) == 0:
        raise ValueError("replicate summaries required")
    null_reject = sum(int(row["story_null_reject"]) for row in replicates)
    arithmetic_birth = sum(int(row["arithmetic_birth"]) for row in replicates)
    duplicate_reject = sum(int(row["duplicate_reject"]) for row in replicates)
    transform_birth = sum(int(row["transform_birth"]) for row in replicates)
    final_three = sum(int(row["final_k"] == 3) for row in replicates)
    if null_reject < len(replicates):
        return "FALSE_POSITIVE_SEQUENTIAL_BIRTH"
    if arithmetic_birth < POSITIVE_REPLICATES_MIN:
        return "NO_FIRST_PROBATIONARY_BIRTH"
    if duplicate_reject < POSITIVE_REPLICATES_MIN:
        return "DUPLICATE_SIGNAL_CAUSES_EXTRA_BIRTH"
    if transform_birth >= POSITIVE_REPLICATES_MIN and final_three >= POSITIVE_REPLICATES_MIN:
        return "SEQUENTIAL_PROBATIONARY_GENESIS_SIGNAL"
    return "FIRST_BIRTH_WITHOUT_SECOND_TRAIT_GENESIS"
