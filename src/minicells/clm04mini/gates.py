"""Registered M1 prerequisite and scientific-gate evaluation."""

from __future__ import annotations

import math
from typing import Mapping

from .engine import VariantHarness


def _ratio(value: float, reference: float) -> float:
    if reference > 0.0:
        return value / reference
    return 0.0 if value <= 0.0 else math.inf


def evaluate_base_prerequisites(
    *,
    protocol: Mapping,
    math_exact_match: float,
    story_exact_match: float,
    cell_activation_counts: Mapping[str, int],
    locked_minimum_activation: int,
    numeric_finite: bool,
    hashes_match_lock: bool,
) -> dict:
    gates = {
        "base_math_exact_match": {
            "value": float(math_exact_match),
            "threshold": float(protocol["base_prerequisites"]["minimum_base_math_exact_match"]),
            "pass": float(math_exact_match)
            >= float(protocol["base_prerequisites"]["minimum_base_math_exact_match"]),
        },
        "base_story_exact_match": {
            "value": float(story_exact_match),
            "threshold": float(protocol["base_prerequisites"]["minimum_base_story_qa_exact_match"]),
            "pass": float(story_exact_match)
            >= float(protocol["base_prerequisites"]["minimum_base_story_qa_exact_match"]),
        },
        "base_cell_activation": {
            "value": min(cell_activation_counts.values()) if cell_activation_counts else 0,
            "threshold": int(locked_minimum_activation),
            "pass": bool(cell_activation_counts)
            and min(cell_activation_counts.values()) >= int(locked_minimum_activation),
        },
        "numeric_finite": {"value": bool(numeric_finite), "threshold": True, "pass": bool(numeric_finite)},
        "hashes_match_lock": {"value": bool(hashes_match_lock), "threshold": True, "pass": bool(hashes_match_lock)},
    }
    return {"gates": gates, "pass": all(item["pass"] for item in gates.values())}


def _maximum_active_private(harness: VariantHarness) -> int:
    maximum = 0
    for record in harness.records:
        for cells in record["active_cells_by_layer"].values():
            maximum = max(maximum, sum(1 for cell in cells if str(cell).startswith("growth:")))
    return maximum


def evaluate_m1_gates(
    *,
    protocol: Mapping,
    harnesses: Mapping[str, VariantHarness],
) -> dict:
    summaries = {name: harness.summary() for name, harness in harnesses.items()}
    always = summaries["local_always"]
    local_tx = summaries["local_tx"]
    growth = summaries["local_tx_growth"]
    damage_ratio = _ratio(
        float(growth["positive_global_regression_damage"]),
        float(always["positive_global_regression_damage"]),
    )
    gain_ratio = _ratio(
        float(growth["committed_new_gain"]), float(always["committed_new_gain"])
    )
    registered = protocol["m1_gates"]
    active_private = _maximum_active_private(harnesses["local_tx_growth"])
    gates = {
        "false_safe_rate": {
            "value": growth["false_safe_rate"],
            "threshold": registered["maximum_false_safe_rate"],
            "pass": growth["false_safe_rate"] <= registered["maximum_false_safe_rate"],
        },
        "structural_escape_rate": {
            "value": growth["maximum_structural_escape_rate"],
            "threshold": registered["maximum_structural_escape_rate"],
            "pass": growth["maximum_structural_escape_rate"]
            <= registered["maximum_structural_escape_rate"],
        },
        "regression_damage_ratio_vs_local_always": {
            "value": damage_ratio,
            "threshold": registered["maximum_regression_damage_ratio_vs_local_always"],
            "pass": damage_ratio <= registered["maximum_regression_damage_ratio_vs_local_always"],
        },
        "effective_acceptance_rate": {
            "value": growth["effective_acceptance_rate"],
            "threshold": registered["minimum_effective_acceptance_rate"],
            "pass": growth["effective_acceptance_rate"] >= registered["minimum_effective_acceptance_rate"],
        },
        "committed_gain_ratio_vs_local_always": {
            "value": gain_ratio,
            "threshold": registered["minimum_committed_gain_ratio_vs_local_always"],
            "pass": gain_ratio >= registered["minimum_committed_gain_ratio_vs_local_always"],
        },
        "final_protected_retention_ratio": {
            "value": growth["final_protected_retention_ratio"],
            "threshold": registered["minimum_final_protected_retention_ratio"],
            "pass": growth["final_protected_retention_ratio"]
            >= registered["minimum_final_protected_retention_ratio"],
        },
        "growth_exceeds_local_tx_gain": {
            "value": growth["committed_new_gain"] - local_tx["committed_new_gain"],
            "threshold": ">0",
            "pass": growth["committed_new_gain"] > local_tx["committed_new_gain"],
        },
        "growth_rescue_rate": {
            "value": growth["growth_rescue_rate"],
            "threshold": registered["minimum_growth_rescue_rate"],
            "pass": growth["growth_rescue_rate"] >= registered["minimum_growth_rescue_rate"],
        },
        "private_reuse_acceptance_rate": {
            "value": growth["private_reuse_acceptance_rate"],
            "threshold": registered["minimum_private_reuse_acceptance_rate"],
            "pass": growth["private_reuse_acceptance_rate"]
            >= registered["minimum_private_reuse_acceptance_rate"],
        },
        "spawned_bundles_per_effective_commit": {
            "value": growth["spawned_bundles_per_effective_commit"],
            "threshold": registered["maximum_spawned_bundles_per_effective_commit"],
            "pass": growth["spawned_bundles_per_effective_commit"]
            <= registered["maximum_spawned_bundles_per_effective_commit"],
        },
        "growth_parameter_overhead_ratio": {
            "value": growth["growth_parameter_overhead_ratio"],
            "threshold": registered["maximum_growth_parameter_overhead_ratio"],
            "pass": growth["growth_parameter_overhead_ratio"]
            <= registered["maximum_growth_parameter_overhead_ratio"],
        },
        "active_private_cells_per_layer_per_input": {
            "value": active_private,
            "threshold": registered["maximum_active_private_cells_per_growth_layer_per_input"],
            "pass": active_private
            <= registered["maximum_active_private_cells_per_growth_layer_per_input"],
        },
        "mean_direct_dependency_coverage": {
            "value": growth["mean_direct_dependency_coverage"],
            "threshold": registered["maximum_mean_direct_dependency_coverage"],
            "pass": growth["mean_direct_dependency_coverage"]
            <= registered["maximum_mean_direct_dependency_coverage"],
        },
    }
    return {
        "variant_summaries": summaries,
        "gates": gates,
        "pass": all(item["pass"] for item in gates.values()),
    }
