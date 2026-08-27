from __future__ import annotations

from collections import Counter

from minicells.language_independent_third_trait import (
    CANDIDATES,
    SCREEN_INDEPENDENCE_MIN,
    aggregate_status,
    screening_score,
    select_candidate,
    selected_stage_schedule,
)
from minicells.language_probationary_trait_genesis import summarize_probation


def test_absorbable_candidate_does_not_qualify_as_new_trait() -> None:
    score = screening_score(
        baseline_candidate_nll=10.0,
        existing_candidate_nll=8.0,
        newborn_candidate_nll=8.0,
        baseline_arithmetic_nll=5.0,
        existing_arithmetic_nll=5.0,
    )
    assert score.existing_candidate_gain == 0.2
    assert score.arithmetic_damage == 0.0
    assert score.independence_advantage < 0.0
    assert not score.qualifies


def test_retention_conflict_can_make_newborn_functionally_independent() -> None:
    score = screening_score(
        baseline_candidate_nll=10.0,
        existing_candidate_nll=8.0,
        newborn_candidate_nll=7.8,
        baseline_arithmetic_nll=5.0,
        existing_arithmetic_nll=6.5,
    )
    assert score.newborn_candidate_gain >= 0.02
    assert score.arithmetic_damage == 0.3
    assert score.independence_advantage > SCREEN_INDEPENDENCE_MIN
    assert score.absorption_ratio == 0.0
    assert score.qualifies


def test_candidate_selection_is_cross_replicate_and_deterministic() -> None:
    rows = []
    for candidate in CANDIDATES:
        for replicate in range(3):
            rows.append(
                {
                    "candidate": candidate,
                    "replicate": replicate,
                    "independence_advantage": 0.002,
                    "qualifies": False,
                }
            )
    for row in rows:
        if row["candidate"] == "DELAY_COPY" and row["replicate"] in (0, 1):
            row["independence_advantage"] = 0.05
            row["qualifies"] = True
        if row["candidate"] == "PARITY" and row["replicate"] in (0, 1):
            row["independence_advantage"] = 0.03
            row["qualifies"] = True
    selection = select_candidate(rows)
    assert selection.qualified
    assert set(selection.qualifying_candidates) == {"DELAY_COPY", "PARITY"}
    assert selection.selected == "DELAY_COPY"


def test_selected_stage_schedules_have_exact_weak_and_strong_counts() -> None:
    weak = Counter(selected_stage_schedule("PARITY", weak=True, replicate=0))
    strong = Counter(selected_stage_schedule("PARITY", weak=False, replicate=0))
    assert weak == {"STORY": 115, "ARITH_A": 115, "PARITY": 26}
    assert strong == {"STORY": 86, "ARITH_A": 85, "PARITY": 85}
    assert sum(weak.values()) == 256
    assert sum(strong.values()) == 256


def test_probation_decision_exposes_last_three_means_without_changing_gate() -> None:
    decision = summarize_probation(
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [9.9, 9.8, 9.8, 9.8],
    )
    assert decision.geometry_mean_net_utility_last3 > 0.0
    assert decision.capacity_mean_net_utility_last3 < 0.0
    assert decision.geometry_advantage_last3 > 0.005
    assert decision.accepted


def test_aggregate_status_requires_screening_and_rejects_early_birth() -> None:
    positive = [
        {"arithmetic_birth": 1, "weak_reject": 1, "strong_birth": 1, "final_k": 3},
        {"arithmetic_birth": 1, "weak_reject": 1, "strong_birth": 1, "final_k": 3},
        {"arithmetic_birth": 0, "weak_reject": 1, "strong_birth": 0, "final_k": 2},
    ]
    assert aggregate_status(positive, screening_qualified=True) == "INDEPENDENT_THIRD_TRAIT_GENESIS_SIGNAL"
    assert aggregate_status(positive, screening_qualified=False) == "NO_FUNCTIONALLY_INDEPENDENT_THIRD_CAPABILITY"
    early = [dict(row) for row in positive]
    early[2]["weak_reject"] = 0
    assert aggregate_status(early, screening_qualified=True) == "INDEPENDENT_CAPABILITY_CAUSES_EARLY_BIRTH"


def test_first_birth_failure_precedes_unexecuted_weak_stage() -> None:
    no_first = [
        {"arithmetic_birth": 1, "weak_reject": 1, "strong_birth": 0, "final_k": 2},
        {"arithmetic_birth": 0, "weak_reject": 0, "strong_birth": 0, "final_k": 1},
        {"arithmetic_birth": 0, "weak_reject": 0, "strong_birth": 0, "final_k": 1},
    ]
    assert aggregate_status(no_first, screening_qualified=True) == "NO_STABLE_FIRST_TRAIT_BIRTH"
