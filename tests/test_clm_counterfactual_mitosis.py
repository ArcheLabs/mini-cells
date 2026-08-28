from __future__ import annotations

import math

import torch

from minicells.growth_counterfactual import (
    paired_bootstrap_utility,
    select_counterfactual_action,
    spearman_rank_correlation,
    split_regret_score,
)


def test_split_regret_is_zero_when_prospective_gradients_match() -> None:
    gradient = torch.tensor([1.0, -2.0, 0.5])
    disagreement, regret = split_regret_score(
        usage=0.1,
        pi0=0.5,
        pi1=0.5,
        gradient0=gradient,
        gradient1=gradient.clone(),
        exp_avg_sq=torch.ones_like(gradient),
    )
    assert disagreement == 0.0
    assert regret == 0.0


def test_split_regret_rewards_untying_disagreement() -> None:
    variance = torch.ones(4)
    _, small = split_regret_score(
        usage=0.1,
        pi0=0.5,
        pi1=0.5,
        gradient0=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        gradient1=torch.tensor([0.5, 0.0, 0.0, 0.0]),
        exp_avg_sq=variance,
    )
    _, large = split_regret_score(
        usage=0.1,
        pi0=0.5,
        pi1=0.5,
        gradient0=torch.tensor([2.0, 0.0, 0.0, 0.0]),
        gradient1=torch.tensor([-2.0, 0.0, 0.0, 0.0]),
        exp_avg_sq=variance,
    )
    assert large > small > 0.0


def test_paired_bootstrap_detects_consistent_candidate_gain() -> None:
    control = [2.5 + 0.01 * index for index in range(32)]
    candidate = [value - 0.02 for value in control]
    result = paired_bootstrap_utility(control, candidate, seed=7, bootstrap_samples=500)
    assert result.relative_improvement > 0.0
    assert result.ci95_low > 0.0
    assert result.ci95_high > result.ci95_low


def test_spearman_detects_monotone_proxy() -> None:
    assert math.isclose(
        spearman_rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        spearman_rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]),
        -1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_counterfactual_policy_uses_lower_confidence_bound() -> None:
    rows = [
        {
            "expert_id": "s0-e0",
            "stage": 0,
            "analytic_rank": 1,
            "relative_improvement": 0.004,
            "ci95_low": -0.001,
            "ci95_high": 0.008,
        },
        {
            "expert_id": "s1-e2",
            "stage": 1,
            "analytic_rank": 3,
            "relative_improvement": 0.003,
            "ci95_low": 0.001,
            "ci95_high": 0.005,
        },
    ]
    decision = select_counterfactual_action(rows)
    assert decision["action"] == "GROW"
    assert decision["selected_expert"] == "s1-e2"


def test_counterfactual_policy_can_choose_no_growth() -> None:
    rows = [
        {
            "expert_id": "s0-e0",
            "stage": 0,
            "analytic_rank": 1,
            "relative_improvement": 0.001,
            "ci95_low": -0.002,
            "ci95_high": 0.004,
        },
        {
            "expert_id": "s0-e1",
            "stage": 0,
            "analytic_rank": 2,
            "relative_improvement": -0.001,
            "ci95_low": -0.003,
            "ci95_high": 0.001,
        },
    ]
    decision = select_counterfactual_action(rows)
    assert decision["action"] == "NO_GROW"
