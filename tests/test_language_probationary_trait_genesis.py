from collections import Counter

from minicells.language_probationary_trait_genesis import (
    CONDITIONS,
    GEOMETRY_ADVANTAGE_MIN,
    PROBATION_STEPS,
    capacity_branch,
    condition_counts,
    condition_schedule,
    expected_condition_outcome,
    normalized_net_utility,
    summarize_probation,
)


def test_condition_schedules_have_exact_preregistered_counts() -> None:
    expected = {
        "STORY_ONLY": {"STORY_A": 256},
        "DUPLICATED_STORY": {"STORY_A": 128, "STORY_B": 128},
        "STORY_ARITHMETIC": {"STORY_A": 128, "ARITHMETIC": 128},
        "WEAK_ARITHMETIC": {"STORY_A": 230, "ARITHMETIC": 26},
    }
    for condition in CONDITIONS:
        assert condition_counts(condition) == expected[condition]
        assert Counter(condition_schedule(condition, replicate=0)) == Counter(expected[condition])
        assert len(condition_schedule(condition, replicate=1)) == PROBATION_STEPS


def test_capacity_control_is_exactly_balanced_per_stream() -> None:
    for condition in CONDITIONS:
        schedule = condition_schedule(condition, replicate=0)
        occurrences = Counter()
        exposure = Counter()
        for key in schedule:
            branch = capacity_branch(occurrences[key], 0)
            exposure[(key, branch)] += 1
            occurrences[key] += 1
        for key in occurrences:
            assert exposure[(key, 0)] == exposure[(key, 1)]


def test_normalized_utility_charges_structural_cost() -> None:
    assert normalized_net_utility(5.0, 5.0) < 0.0
    assert normalized_net_utility(5.0, 4.9) > 0.0


def test_probation_rejects_transient_or_late_single_window_gain() -> None:
    parent = [5.0, 5.0, 5.0, 5.0]
    capacity = [5.0, 5.0, 5.0, 5.0]
    geometry = [5.1, 5.1, 5.1, 4.7]
    decision = summarize_probation(parent, capacity, geometry)
    assert not decision.sustained_positive
    assert not decision.accepted


def test_probation_accepts_sustained_geometry_gain_beyond_capacity() -> None:
    parent = [5.0, 5.0, 5.0, 5.0]
    capacity = [5.0, 4.99, 4.98, 4.98]
    geometry = [5.02, 4.92, 4.88, 4.86]
    decision = summarize_probation(parent, capacity, geometry)
    assert decision.sustained_positive
    assert decision.cumulative_positive
    assert decision.beats_capacity
    assert decision.accepted
    assert sum(decision.geometry_advantage[-3:]) / 3.0 >= GEOMETRY_ADVANTAGE_MIN


def test_positive_geometry_without_capacity_advantage_is_not_birth() -> None:
    parent = [5.0, 5.0, 5.0, 5.0]
    capacity = [5.0, 4.90, 4.88, 4.86]
    geometry = [5.0, 4.89, 4.87, 4.85]
    decision = summarize_probation(parent, capacity, geometry)
    assert decision.sustained_positive
    assert decision.cumulative_positive
    assert not decision.beats_capacity
    assert not decision.accepted


def test_null_and_positive_expectations_are_not_used_as_commit_inputs() -> None:
    assert expected_condition_outcome("STORY_ONLY") == "REJECT"
    assert expected_condition_outcome("DUPLICATED_STORY") == "REJECT"
    assert expected_condition_outcome("STORY_ARITHMETIC") == "ACCEPT"
    assert expected_condition_outcome("WEAK_ARITHMETIC") == "DISCOVER"
