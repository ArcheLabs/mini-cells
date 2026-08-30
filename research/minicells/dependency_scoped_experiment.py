"""Formal orchestration for Core Validation 003."""

from __future__ import annotations

from typing import Any

import torch

from .dependency_scoped_transactional import (
    CoreValidation003Config,
    RoutedContinualWorld,
    VARIANTS,
    pretrain_model,
    run_variant,
    summarize_granularity,
)


def run_primary_seed(
    config: CoreValidation003Config,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    world = RoutedContinualWorld(config, seed=seed + 101)
    historical_x, historical_contexts = world.fixed_historical_inputs(
        examples_per_context=config.historical_examples_per_context,
        seed=seed + 151,
    )
    transaction_contexts = world.transaction_stream(
        transactions=config.transactions,
        seed=seed + 181,
    )

    granularity_runs: list[dict[str, Any]] = []
    coarsest_dependency_coverage: float | None = None
    for granularity in config.granularities:
        pretrained, pretraining = pretrain_model(
            config,
            world,
            granularity=granularity,
            seed=seed + granularity * 1009,
            device=device,
        )
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            variants[variant] = run_variant(
                pretrained,
                config,
                world,
                variant=variant,
                transaction_contexts=transaction_contexts,
                seed=seed + granularity * 2003,
                historical_x=historical_x,
                historical_contexts=historical_contexts,
                device=device,
            )

        tx_coverage = float(
            variants["local_tx_frozen"]["summary"]["mean_dependency_coverage"]
        )
        if granularity == config.granularities[0]:
            coarsest_dependency_coverage = tx_coverage

        gate_summary = summarize_granularity(
            config,
            granularity=granularity,
            pretraining=pretraining,
            variant_runs=variants,
            coarsest_dependency_coverage=coarsest_dependency_coverage,
        )
        granularity_runs.append(
            {
                "granularity": granularity,
                "pretraining": pretraining,
                "variants": variants,
                "gate_summary": gate_summary,
            }
        )

    supported = [
        int(run["granularity"])
        for run in granularity_runs
        if bool(run["gate_summary"]["pass"])
    ]
    h1 = all(
        float(run["variants"]["local_tx_frozen"]["summary"]["maximum_structural_escape_rate"])
        <= config.maximum_structural_escape_rate
        for run in granularity_runs
        if float(run["pretraining"]["base_normalized_mse"])
        <= config.maximum_base_normalized_mse
    )
    stress_escape = max(
        float(run["variants"]["local_tx_router_drift"]["summary"]["maximum_structural_escape_rate"])
        for run in granularity_runs
    )
    frozen_escape = max(
        float(run["variants"]["local_tx_frozen"]["summary"]["maximum_structural_escape_rate"])
        for run in granularity_runs
    )
    stress_false_safe = max(
        float(run["variants"]["local_tx_router_drift"]["summary"]["false_safe_rate"])
        for run in granularity_runs
    )
    frozen_false_safe = max(
        float(run["variants"]["local_tx_frozen"]["summary"]["false_safe_rate"])
        for run in granularity_runs
    )

    return {
        "seed": seed,
        "granularities": granularity_runs,
        "supported_granularities": supported,
        "hypothesis_diagnostics": {
            "h1_structural_locality": bool(h1),
            "h5_router_drift_increases_escape": bool(stress_escape > frozen_escape),
            "h5_router_drift_increases_false_safe": bool(stress_false_safe > frozen_false_safe),
            "maximum_frozen_structural_escape_rate": frozen_escape,
            "maximum_router_drift_structural_escape_rate": stress_escape,
            "maximum_frozen_false_safe_rate": frozen_false_safe,
            "maximum_router_drift_false_safe_rate": stress_false_safe,
        },
    }


def summarize_experiment(
    runs: list[dict[str, Any]],
    *,
    granularities: tuple[int, ...],
    positive_status: str,
    negative_status: str,
) -> dict[str, Any]:
    shared_supported = []
    for granularity in granularities:
        if granularity == 1:
            continue
        if all(granularity in run["supported_granularities"] for run in runs):
            shared_supported.append(granularity)

    h1 = all(run["hypothesis_diagnostics"]["h1_structural_locality"] for run in runs)
    h5_escape = all(
        run["hypothesis_diagnostics"]["h5_router_drift_increases_escape"] for run in runs
    )
    h5_false_safe = all(
        run["hypothesis_diagnostics"]["h5_router_drift_increases_false_safe"] for run in runs
    )
    passed = bool(shared_supported)
    return {
        "status": positive_status if passed else negative_status,
        "pass": passed,
        "scientific_decision": True,
        "passed_seeds": sum(bool(run["supported_granularities"]) for run in runs),
        "total_seeds": len(runs),
        "shared_supported_granularities": shared_supported,
        "require_same_granularity_across_seeds": True,
        "hypotheses": {
            "h1_structural_locality": h1,
            "h2_dependency_scoped_validation": passed,
            "h3_transactional_commit_benefit": passed,
            "h4_granularity_benefit": passed,
            "h5_router_stability_stress_escape": h5_escape,
            "h5_router_stability_stress_false_safe": h5_false_safe,
        },
    }
