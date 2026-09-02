from __future__ import annotations

from minicells.constructive_clm_004 import model_level_smoke


def test_model_level_smoke_separates_composition_controls() -> None:
    result = model_level_smoke(501)

    acquisition = result["acquisition"]
    simultaneous = result["simultaneous"]
    sequential = result["sequential"]

    assert acquisition["route_accuracy"] == 1.0
    assert acquisition["raw_examples_retained"] == 0
    assert acquisition["mean_operator_relative_error"] <= 0.01

    assert simultaneous["mean_mse"] <= 1e-4
    assert sequential["mean_mse"] <= 1e-4
    assert simultaneous["exact_route_sequence_accuracy"] == 1.0
    assert sequential["exact_route_sequence_accuracy"] == 1.0

    assert sequential["mean_true_order_effect_mse"] >= 1e-3
    assert simultaneous["mean_simultaneous_permutation_mse"] <= 1e-12

    assert sequential["mean_mse"] <= 0.05 * sequential["single_cell_baseline_mse"]
    assert simultaneous["mean_mse"] <= 0.05 * simultaneous["single_cell_baseline_mse"]


def test_model_level_smoke_preserves_certificate_inside_composition() -> None:
    result = model_level_smoke(502)
    mutation = result["protected_mutation"]

    assert mutation["learner_replay_accesses"] == 0
    assert mutation["learner_raw_history_retained"] == 0
    assert mutation["safe_fit_error"] <= 1e-10
    assert mutation["safe_historical_composition_mse"] <= 1e-10
    assert mutation["safe_protected_change"] <= 1e-10

    assert mutation["unsafe_historical_composition_mse"] >= 1e-4
    assert mutation["unsafe_protected_change"] >= 1e-3
    assert mutation["unrelated_cell_parameter_drift"] <= 1e-15
    assert mutation["route_key_drift"] <= 1e-15
