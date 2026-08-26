from __future__ import annotations

import pytest

from minicells.language_depth_ablation import variant_by_code
from minicells.language_multiseed_core import (
    CORE_VARIANT_CODES,
    N_REPLICATES,
    core_recipe_confirmation,
    core_recipe_ratio,
    exact_bootstrap_geometric_ci,
    factor_ratio,
    geometric_mean,
    model_seed,
    ratio_summary,
    seed_bundle,
)


def test_experiment_014_keeps_only_scale_one_core_variants() -> None:
    assert CORE_VARIANT_CODES == ("A", "B", "F", "H")
    for code in CORE_VARIANT_CODES:
        variant = variant_by_code(code)
        assert variant.step_embedding_init_scale == 1.0
    assert not variant_by_code("A").random_depth
    assert variant_by_code("B").random_depth
    assert variant_by_code("F").uses_stability_loss
    assert variant_by_code("H").random_depth and variant_by_code("H").uses_stability_loss


def test_seed_bundles_are_reproducible_unique_and_topology_matched() -> None:
    bundles = [seed_bundle(index) for index in range(N_REPLICATES)]
    assert bundles == [seed_bundle(index) for index in range(N_REPLICATES)]
    assert len({bundle.schedule_seed for bundle in bundles}) == N_REPLICATES
    assert len({bundle.depth_seed for bundle in bundles}) == N_REPLICATES
    assert len({bundle.validation_seed for bundle in bundles}) == N_REPLICATES
    assert len({model_seed(bundle, "1d") for bundle in bundles}) == N_REPLICATES
    assert len({model_seed(bundle, "2d") for bundle in bundles}) == N_REPLICATES
    for bundle in bundles:
        assert model_seed(bundle, "2d") - model_seed(bundle, "1d") == 4


def test_seed_bundle_rejects_out_of_range_replicates() -> None:
    with pytest.raises(ValueError):
        seed_bundle(-1)
    with pytest.raises(ValueError):
        seed_bundle(N_REPLICATES)


def test_factor_ratios_match_balanced_two_by_two_log_contrasts() -> None:
    values = {"A": 100.0, "B": 80.0, "F": 90.0, "H": 72.0}
    assert core_recipe_ratio(values) == pytest.approx(0.72)
    assert factor_ratio(values, "random_depth") == pytest.approx(0.8)
    assert factor_ratio(values, "stability_loss") == pytest.approx(0.9)


def test_exact_bootstrap_geometric_ci_is_deterministic() -> None:
    ratios = (0.80, 0.82, 0.84, 0.86, 0.88)
    first = exact_bootstrap_geometric_ci(ratios)
    second = exact_bootstrap_geometric_ci(ratios)
    mean = geometric_mean(ratios)
    assert first == second
    assert first[0] <= mean <= first[1]
    assert first[0] > 0.0


def test_ratio_summary_uses_all_five_replicates() -> None:
    summary = ratio_summary((0.80, 0.82, 0.84, 0.86, 0.88), aggregate_threshold=0.90)
    assert summary["replicates"] == 5
    assert summary["aggregate_pass"] is True
    assert summary["geometric_mean_ratio"] < 0.90


def test_core_recipe_confirmation_requires_aggregate_and_seed_level_evidence() -> None:
    confirmed = core_recipe_confirmation(
        ppl_ratios=(0.99, 0.995, 1.00, 1.005, 1.008),
        cost_ratios=(0.84, 0.86, 0.87, 0.88, 0.89),
    )
    assert confirmed["confirmed"] is True
    assert confirmed["joint_pass_replicates"] == 5
    assert confirmed["ppl"]["aggregate_pass"] is True
    assert confirmed["cost"]["aggregate_pass"] is True

    # A favorable geometric mean is not enough if too many paired seeds miss
    # the pre-registered joint quality/cost envelope.
    mixed = core_recipe_confirmation(
        ppl_ratios=(0.94, 0.95, 1.02, 1.03, 1.04),
        cost_ratios=(0.70, 0.72, 0.90, 0.91, 0.92),
    )
    assert mixed["confirmed"] is False
    assert mixed["joint_pass_replicates"] < 4
