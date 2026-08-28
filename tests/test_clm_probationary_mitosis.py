from __future__ import annotations

from minicells.growth_counterfactual import PairedUtility
from minicells.growth_probationary import (
    ProbationPoint,
    absorption_diagnostic,
    condition_domains,
    independent_confirmation,
    select_promotion_candidate,
    shortlist_candidates,
    summarize_probation,
)


def utility(relative: float, low: float, high: float) -> PairedUtility:
    return PairedUtility(
        control_nll=2.0,
        candidate_nll=2.0 * (1.0 - relative),
        delta_nll=2.0 * relative,
        relative_improvement=relative,
        ci95_low=low,
        ci95_high=high,
        bootstrap_samples=2000,
        batches=32,
    )


def point(tokens: int, relative: float, low: float, high: float, ratio: float) -> ProbationPoint:
    return ProbationPoint(
        tokens=tokens,
        utility=utility(relative, low, high),
        control_ppl=10.0,
        candidate_ppl=10.0 * ratio,
    )


def test_condition_schedule_is_deterministic_and_balanced() -> None:
    first = condition_domains("story_arithmetic_shift", steps=500, seed=123)
    second = condition_domains("story_arithmetic_shift", steps=500, seed=123)
    assert first == second
    assert first.count("story") == 250
    assert first.count("arithmetic") == 250
    stationary = condition_domains("stationary_story", steps=17, seed=999)
    assert set(stationary) == {"story"}


def test_shortlist_uses_realized_100k_utility_not_analytic_fields() -> None:
    rows = [
        {"expert_id": "s0-e0", "relative_improvement": 0.001, "split_regret": 100.0},
        {"expert_id": "s0-e1", "relative_improvement": 0.004, "split_regret": 0.0},
        {"expert_id": "s0-e2", "relative_improvement": 0.003, "split_regret": 1.0},
        {"expert_id": "s0-e3", "relative_improvement": 0.002, "split_regret": 2.0},
        {"expert_id": "s1-e0", "relative_improvement": -0.001, "split_regret": 200.0},
    ]
    selected = shortlist_candidates(rows, k=4)
    assert [row["expert_id"] for row in selected] == ["s0-e1", "s0-e2", "s0-e3", "s0-e0"]


def test_late_maturing_shadow_can_pass_probation() -> None:
    points = [
        point(50_000, -0.0002, -0.001, 0.0005, 1.0002),
        point(100_000, 0.0002, -0.0004, 0.0008, 0.9998),
        point(200_000, 0.0010, -0.0001, 0.0020, 0.9980),
        point(300_000, 0.0020, 0.0004, 0.0030, 0.9960),
        point(500_000, 0.0030, 0.0010, 0.0040, 0.9940),
    ]
    decision = summarize_probation("s2-e2", points)
    assert decision.sustained_positive
    assert decision.cumulative_positive
    assert decision.practical_effect
    assert decision.accepted_on_probe_holdout


def test_early_gain_that_decays_is_rejected() -> None:
    points = [
        point(50_000, 0.0030, 0.0010, 0.0050, 0.994),
        point(100_000, 0.0020, 0.0010, 0.0030, 0.995),
        point(200_000, 0.0010, 0.0001, 0.0020, 0.997),
        point(300_000, 0.0001, -0.0004, 0.0005, 0.999),
        point(500_000, -0.0002, -0.0010, 0.0004, 1.001),
    ]
    decision = summarize_probation("s0-e3", points)
    assert not decision.sustained_positive
    assert not decision.practical_effect
    assert not decision.accepted_on_probe_holdout


def test_independent_confirmation_requires_statistics_practicality_and_retention() -> None:
    positive = independent_confirmation(
        utility=utility(0.003, 0.001, 0.005),
        control_ppl=10.0,
        candidate_ppl=9.94,
        story_control_nll=2.0,
        story_candidate_nll=2.01,
    )
    assert positive["confirmed"] is True
    forgetting = independent_confirmation(
        utility=utility(0.003, 0.001, 0.005),
        control_ppl=10.0,
        candidate_ppl=9.94,
        story_control_nll=2.0,
        story_candidate_nll=2.03,
    )
    assert forgetting["confirmed"] is False


def test_absorption_diagnostic_separates_shift_learning_from_story_damage() -> None:
    row = absorption_diagnostic(
        baseline_story_nll=2.0,
        baseline_arithmetic_nll=4.0,
        control_story_nll=2.01,
        control_arithmetic_nll=3.6,
    )
    assert row["arithmetic_gain"] > 0.02
    assert row["story_damage"] <= 0.01
    assert row["absorbable_without_mitosis"] is True


def test_promotion_selects_only_accepted_shadow() -> None:
    accepted = summarize_probation(
        "s1-e1",
        [
            point(50_000, 0.0, -0.001, 0.001, 1.0),
            point(100_000, 0.0, -0.001, 0.001, 1.0),
            point(200_000, 0.001, -0.001, 0.002, 0.998),
            point(300_000, 0.002, 0.001, 0.003, 0.996),
            point(500_000, 0.003, 0.001, 0.004, 0.994),
        ],
    )
    rejected = summarize_probation(
        "s1-e2",
        [
            point(50_000, 0.004, 0.002, 0.005, 0.993),
            point(100_000, 0.004, 0.002, 0.005, 0.993),
            point(200_000, 0.004, 0.002, 0.005, 0.993),
            point(300_000, 0.004, -0.001, 0.005, 0.993),
            point(500_000, 0.004, -0.001, 0.005, 0.993),
        ],
    )
    selected = select_promotion_candidate([rejected, accepted])
    assert selected is not None
    assert selected.expert_id == "s1-e1"
