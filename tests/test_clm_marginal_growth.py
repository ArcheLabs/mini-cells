from __future__ import annotations

from minicells.growth_marginal import (
    MarginalCandidate,
    detect_saturation,
    marginal_capacity_score,
    rank_marginal_candidates,
)
from minicells.growth_marginal_reporting import marginal_growth_decision


def test_marginal_capacity_score_rewards_sensitivity_and_separability() -> None:
    base = marginal_capacity_score(2.0, 3.0, 0.2)
    assert marginal_capacity_score(4.0, 3.0, 0.2) > base
    assert marginal_capacity_score(2.0, 6.0, 0.2) > base
    assert marginal_capacity_score(2.0, 3.0, 0.8) > base


def test_marginal_candidate_ranking_uses_score_not_legacy_pressure() -> None:
    low = MarginalCandidate(
        stage=0,
        expert_id="s0-e0",
        usage=0.20,
        gradient_disagreement=0.9,
        legacy_pressure=0.38,
        fisher_per_route=1.0,
        weight_grad_saliency=1.0,
        geometry_separation=0.1,
        marginal_score=0.55,
        routed_samples=1000,
        eligible=True,
    )
    high = MarginalCandidate(
        stage=1,
        expert_id="s1-e0",
        usage=0.05,
        gradient_disagreement=0.1,
        legacy_pressure=0.055,
        fisher_per_route=4.0,
        weight_grad_saliency=2.0,
        geometry_separation=0.7,
        marginal_score=2.4,
        routed_samples=1000,
        eligible=True,
    )
    assert rank_marginal_candidates([low, high])[0].expert_id == "s1-e0"


def test_saturation_detects_flat_log_ppl_window() -> None:
    rows = [
        {"tokens": 1_100_000 + index * 100_000, "ppl": 15.0 - index * 0.002}
        for index in range(5)
    ]
    result = detect_saturation(rows, min_tokens=1_500_000)
    assert result.detected is True
    assert result.token == 1_500_000
    assert result.projected_improvement_500k is not None
    assert result.projected_improvement_500k <= 0.005


def test_saturation_rejects_still_improving_model() -> None:
    rows = [
        {"tokens": 1_100_000 + index * 100_000, "ppl": 15.0 * (0.997 ** index)}
        for index in range(5)
    ]
    result = detect_saturation(rows, min_tokens=1_500_000)
    assert result.detected is False
    assert result.projected_improvement_500k is not None
    assert result.projected_improvement_500k > 0.005


def test_formal_decision_requires_paired_utility_and_selector_wins() -> None:
    summaries = []
    fixed = [15.0, 15.2, 14.8]
    marginal = [14.90, 15.10, 14.80]
    random = [14.95, 15.15, 14.75]
    for replicate in range(3):
        summaries.extend([
            {"replicate": replicate, "arm": "fixed4", "ppl": fixed[replicate]},
            {"replicate": replicate, "arm": "marginal_growth", "ppl": marginal[replicate]},
            {"replicate": replicate, "arm": "random_growth", "ppl": random[replicate]},
        ])
    decision = marginal_growth_decision(
        summaries,
        saturation_replicates=3,
        paired_prebirth_replicates=3,
        equivalent_growth_births=6,
        viable_marginal_births=3,
        causal_positive_ci_replicates=2,
        formal_gpu_experiment_run=True,
        training_code_commit="abc",
        training_code_tree_sha="def",
    )
    assert decision["paired_prebirth"]["status"] == "CLM_PAIRED_PREBIRTH_EQUIVALENCE"
    assert decision["saturation_regime"]["status"] == "CLM_SATURATION_REGIME_ESTABLISHED"
    assert decision["growth_equivalence"]["status"] == "CLM_GROWTH_EQUIVALENCE"
    assert decision["marginal_growth_utility"]["status"] == "CLM_MARGINAL_GROWTH_UTILITY_SIGNAL"
    assert decision["marginal_selection"]["status"] == "CLM_MARGINAL_SELECTION_SIGNAL"
    assert decision["causal_utility"]["status"] == "CLM_NEWBORN_CAUSAL_UTILITY_SIGNAL"
    assert decision["training_code_commit"] == "abc"
