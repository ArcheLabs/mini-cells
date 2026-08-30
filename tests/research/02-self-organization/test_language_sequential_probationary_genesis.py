from collections import Counter

from minicells.language_sequential_probationary_genesis import (
    STAGES,
    aggregate_status,
    capacity_shadow_branch,
    classify_replicate,
    expected_trajectory,
    stage_schedule,
    stage_spec,
    summarize_stage_decision,
)


def test_stage_schedules_have_exact_preregistered_counts() -> None:
    expected = {
        "A_STORY_NULL": {"STORY": 256},
        "B_ARITHMETIC_BIRTH": {"STORY": 128, "ARITH_A": 128},
        "C_DUPLICATE_ARITHMETIC": {"STORY": 128, "ARITH_A": 64, "ARITH_B": 64},
        "D_WEAK_TRANSFORM": {"STORY": 115, "ARITH_A": 115, "TRANSFORM": 26},
        "E_TRANSFORM_BIRTH": {"STORY": 86, "ARITH_A": 85, "TRANSFORM": 85},
    }
    for stage in STAGES:
        assert stage_spec(stage).counts == expected[stage]
        assert Counter(stage_schedule(stage, replicate=0)) == Counter(expected[stage])
        assert len(stage_schedule(stage, replicate=2)) == 256


def test_expected_trajectory_is_one_then_two_then_three() -> None:
    assert expected_trajectory() == (1, 1, 2, 2, 2, 3)
    assert stage_spec("A_STORY_NULL").expected_outcome == "REJECT"
    assert stage_spec("B_ARITHMETIC_BIRTH").expected_outcome == "ACCEPT"
    assert stage_spec("C_DUPLICATE_ARITHMETIC").expected_outcome == "REJECT"
    assert stage_spec("D_WEAK_TRANSFORM").expected_outcome == "REJECT"
    assert stage_spec("E_TRANSFORM_BIRTH").expected_outcome == "ACCEPT"


def test_capacity_shadow_only_splits_the_proposed_parent() -> None:
    assert capacity_shadow_branch(
        incumbent_branch=0,
        parent_branch=1,
        newborn_branch=2,
        occurrence=0,
        replicate=0,
    ) == 0
    routed = [
        capacity_shadow_branch(
            incumbent_branch=1,
            parent_branch=1,
            newborn_branch=2,
            occurrence=index,
            replicate=0,
        )
        for index in range(6)
    ]
    assert routed == [1, 2, 1, 2, 1, 2]


def test_stage_utility_can_accept_a_sustained_useful_birth() -> None:
    decision = summarize_stage_decision(
        stage="B_ARITHMETIC_BIRTH",
        start_k=1,
        parent_window_losses=[5.0, 5.0, 5.0, 5.0],
        capacity_window_losses=[5.0, 5.0, 5.0, 5.0],
        geometry_window_losses=[5.0, 4.90, 4.86, 4.84],
    )
    assert decision.accepted
    assert decision.end_k == 2
    assert decision.geometry_mean_net_utility_last3 > 0.0
    assert decision.geometry_advantage_last3 > 0.005


def test_stage_utility_rejects_transient_gain() -> None:
    decision = summarize_stage_decision(
        stage="C_DUPLICATE_ARITHMETIC",
        start_k=2,
        parent_window_losses=[5.0, 5.0, 5.0, 5.0],
        capacity_window_losses=[5.0, 5.0, 5.0, 5.0],
        geometry_window_losses=[4.7, 5.0, 5.0, 5.0],
    )
    assert not decision.accepted
    assert decision.end_k == 2


def _successful_stage_rows() -> list[dict[str, object]]:
    return [
        {"stage": "A_STORY_NULL", "start_k": 1, "end_k": 1, "accepted": 0},
        {
            "stage": "B_ARITHMETIC_BIRTH",
            "start_k": 1,
            "end_k": 2,
            "accepted": 1,
            "identity_pass": 1,
            "routing_purity_pass": 1,
        },
        {
            "stage": "C_DUPLICATE_ARITHMETIC",
            "start_k": 2,
            "end_k": 2,
            "accepted": 0,
            "retention_identity_pass": 1,
        },
        {
            "stage": "D_WEAK_TRANSFORM",
            "start_k": 2,
            "end_k": 2,
            "accepted": 0,
            "retention_identity_pass": 1,
        },
        {
            "stage": "E_TRANSFORM_BIRTH",
            "start_k": 2,
            "end_k": 3,
            "accepted": 1,
            "identity_pass": 1,
            "routing_purity_pass": 1,
        },
    ]


def test_aggregate_requires_clean_negative_controls_and_two_births() -> None:
    replicates = [classify_replicate(_successful_stage_rows()) for _ in range(3)]
    assert aggregate_status(replicates) == "SEQUENTIAL_PROBATIONARY_GENESIS_SIGNAL"


def test_weak_early_birth_cannot_be_called_success() -> None:
    rows = _successful_stage_rows()
    rows[3] = {
        "stage": "D_WEAK_TRANSFORM",
        "start_k": 2,
        "end_k": 3,
        "accepted": 1,
        "retention_identity_pass": 1,
    }
    rows[4] = {
        "stage": "E_TRANSFORM_BIRTH",
        "start_k": 3,
        "end_k": 4,
        "accepted": 1,
        "identity_pass": 1,
        "routing_purity_pass": 1,
    }
    bad = classify_replicate(rows)
    good = classify_replicate(_successful_stage_rows())
    assert aggregate_status([bad, good, good]) == "WEAK_SIGNAL_CAUSES_EARLY_BIRTH"
