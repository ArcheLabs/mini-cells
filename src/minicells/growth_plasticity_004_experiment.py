"""Variant orchestration and frozen decision gates for Core Validation 004."""

from __future__ import annotations

import copy
from typing import Any

import torch

from .growth_plasticity_004_config import CoreValidation004Config, _EPS
from .growth_plasticity_004_world import GrowthPlasticityWorld
from .growth_plasticity_004_model import GrowingRoutedModel
from .growth_plasticity_004_ops import (
    evaluate_candidate, evaluate_final_state, pretrain_model,
    train_direct_candidate, train_private_cell_candidate,
)

VARIANTS = ("local_always", "local_tx", "local_tx_growth")


def _summarize_records(
    model: GrowingRoutedModel,
    records: list[dict[str, Any]],
    *,
    config: CoreValidation004Config,
    world: GrowthPlasticityWorld,
    amplitudes: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    effective = [r for r in records if r["effective_commit"]]
    local_passes = [r for r in records if r["decision_local_pass"]]
    rescue_attempts = [r for r in records if r["growth_attempted"]]
    rescue_successes = [r for r in rescue_attempts if r["growth_committed"]]
    private_reuse = [r for r in records if r["used_existing_private_cell"]]
    private_reuse_commits = [r for r in private_reuse if r["effective_commit"]]
    cumulative_damage = sum(max(0.0, float(r["committed_global_regression"])) for r in effective)
    cumulative_gain = sum(max(0.0, float(r["committed_new_gain_fraction"])) for r in effective)
    validation_cost = sum(float(r["transaction_validation_cost"]) for r in records)
    final = evaluate_final_state(
        model, config, world, amplitudes=amplitudes, seed=seed + 50000, device=device
    )
    spawned = len(model.growth_cells)
    return {
        "effective_acceptance_rate": len(effective) / max(len(records), 1),
        "false_safe_rate": sum(bool(r["false_safe"]) for r in local_passes) / max(len(local_passes), 1),
        "maximum_structural_escape_rate": max(
            (float(r["maximum_structural_escape_rate"]) for r in records), default=0.0
        ),
        "cumulative_positive_global_regression": cumulative_damage,
        "cumulative_committed_new_gain": cumulative_gain,
        "growth_attempt_rate": len(rescue_attempts) / max(len(records), 1),
        "growth_rescue_rate": len(rescue_successes) / max(len(rescue_attempts), 1),
        "private_cell_reuse_acceptance_rate": len(private_reuse_commits) / max(len(private_reuse), 1),
        "spawned_cells": spawned,
        "spawned_cells_per_effective_commit": spawned / max(len(effective), 1),
        "maximum_active_growth_cells_per_input": 1 if spawned else 0,
        "normalized_state_validation_cost_per_effective_commit": validation_cost / max(len(effective), 1),
        **final,
    }


def run_variant(
    pretrained: GrowingRoutedModel,
    config: CoreValidation004Config,
    world: GrowthPlasticityWorld,
    *,
    variant: str,
    transaction_contexts: list[int],
    seed: int,
    historical_x: torch.Tensor,
    historical_contexts: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    model = copy.deepcopy(pretrained)
    amplitudes = world.initial_amplitudes()
    records: list[dict[str, Any]] = []

    for index, target_context in enumerate(transaction_contexts):
        old_amplitudes = amplitudes.clone()
        new_amplitudes = amplitudes.clone()
        new_amplitudes[target_context] += config.update_amplitude
        attempts: list[dict[str, Any]] = []
        effective_commit = False
        growth_attempted = False
        growth_committed = False
        used_existing_private = False
        committed_metrics: dict[str, Any] | None = None

        existing_private = model.growth_cell_for_context(target_context)
        if variant == "local_tx_growth" and existing_private is not None:
            used_existing_private = True
            candidate = train_private_cell_candidate(
                model,
                config,
                world,
                amplitudes=new_amplitudes,
                target_context=target_context,
                seed=seed + 10000 + index,
                device=device,
                spawn=False,
            )
            metrics = evaluate_candidate(
                model,
                candidate,
                config,
                world,
                old_amplitudes=old_amplitudes,
                new_amplitudes=new_amplitudes,
                target_context=target_context,
                historical_x=historical_x,
                historical_contexts=historical_contexts,
                new_validation_seed=seed + 30000 + index,
                device=device,
                candidate_kind="private",
            )
            attempts.append(metrics)
            if metrics["local_pass"]:
                model = candidate
                effective_commit = True
                committed_metrics = metrics
        else:
            direct = train_direct_candidate(
                model,
                config,
                world,
                amplitudes=new_amplitudes,
                target_context=target_context,
                seed=seed + 10000 + index,
                device=device,
            )
            direct_metrics = evaluate_candidate(
                model,
                direct,
                config,
                world,
                old_amplitudes=old_amplitudes,
                new_amplitudes=new_amplitudes,
                target_context=target_context,
                historical_x=historical_x,
                historical_contexts=historical_contexts,
                new_validation_seed=seed + 30000 + index,
                device=device,
                candidate_kind="direct",
            )
            attempts.append(direct_metrics)
            if variant == "local_always":
                model = direct
                effective_commit = True
                committed_metrics = direct_metrics
            elif direct_metrics["local_pass"]:
                model = direct
                effective_commit = True
                committed_metrics = direct_metrics
            elif variant == "local_tx_growth":
                growth_attempted = True
                grown = train_private_cell_candidate(
                    model,
                    config,
                    world,
                    amplitudes=new_amplitudes,
                    target_context=target_context,
                    seed=seed + 20000 + index,
                    device=device,
                    spawn=True,
                )
                growth_metrics = evaluate_candidate(
                    model,
                    grown,
                    config,
                    world,
                    old_amplitudes=old_amplitudes,
                    new_amplitudes=new_amplitudes,
                    target_context=target_context,
                    historical_x=historical_x,
                    historical_contexts=historical_contexts,
                    new_validation_seed=seed + 40000 + index,
                    device=device,
                    candidate_kind="spawn",
                )
                attempts.append(growth_metrics)
                if growth_metrics["local_pass"]:
                    model = grown
                    effective_commit = True
                    growth_committed = True
                    committed_metrics = growth_metrics

        amplitudes = new_amplitudes
        decision_metrics = attempts[-1]
        false_safe = any(bool(m["local_pass"] and not m["oracle_pass"]) for m in attempts)
        structural_escape = max(float(m["structural_escape_rate"]) for m in attempts)
        validation_cost = sum(
            float(m["candidate_state_fraction"]) + float(m["dependency_coverage"])
            for m in attempts
        )
        records.append(
            {
                "transaction": index,
                "variant": variant,
                "target_context": int(target_context),
                "effective_commit": effective_commit,
                "growth_attempted": growth_attempted,
                "growth_committed": growth_committed,
                "used_existing_private_cell": used_existing_private,
                "attempt_count": len(attempts),
                "decision_local_pass": bool(decision_metrics["local_pass"]),
                "false_safe": false_safe,
                "maximum_structural_escape_rate": structural_escape,
                "transaction_validation_cost": validation_cost,
                "committed_global_regression": (
                    float(committed_metrics["global_regression"]) if committed_metrics else 0.0
                ),
                "committed_new_gain_fraction": (
                    float(committed_metrics["new_gain_fraction"]) if committed_metrics else 0.0
                ),
                "attempts": attempts,
            }
        )

    return {
        "summary": _summarize_records(
            model,
            records,
            config=config,
            world=world,
            amplitudes=amplitudes,
            seed=seed,
            device=device,
        ),
        "records": records,
    }


def summarize_seed(
    config: CoreValidation004Config,
    *,
    pretraining: dict[str, float],
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    always = variants["local_always"]["summary"]
    tx = variants["local_tx"]["summary"]
    growth = variants["local_tx_growth"]["summary"]
    damage_ratio = growth["cumulative_positive_global_regression"] / max(
        always["cumulative_positive_global_regression"], _EPS
    )
    gain_ratio = growth["cumulative_committed_new_gain"] / max(
        always["cumulative_committed_new_gain"], _EPS
    )
    final_mutable_ratio = growth["mutable_normalized_mse"] / max(
        always["mutable_normalized_mse"], _EPS
    )
    gates = {
        "base_quality": pretraining["base_normalized_mse"] <= config.maximum_base_normalized_mse,
        "scope_safety": growth["false_safe_rate"] <= config.maximum_false_safe_rate,
        "structural_locality": growth["maximum_structural_escape_rate"] <= config.maximum_structural_escape_rate,
        "effective_acceptance": growth["effective_acceptance_rate"] >= config.minimum_effective_acceptance_rate,
        "regression_reduction": damage_ratio <= config.maximum_regression_damage_ratio_vs_local_always,
        "plasticity_recovery": gain_ratio >= config.minimum_committed_gain_ratio_vs_local_always,
        "growth_rescue": growth["growth_rescue_rate"] >= config.minimum_growth_rescue_rate,
        "private_cell_reuse": (
            growth["private_cell_reuse_acceptance_rate"]
            >= config.minimum_private_cell_reuse_acceptance_rate
        ),
        "bounded_growth": (
            growth["spawned_cells_per_effective_commit"]
            <= config.maximum_spawned_cells_per_effective_commit
        ),
        "sparse_active_growth": (
            growth["maximum_active_growth_cells_per_input"]
            <= config.maximum_active_growth_cells_per_input
        ),
        "growth_improves_tx_plasticity": (
            growth["cumulative_committed_new_gain"] > tx["cumulative_committed_new_gain"]
        ),
        "final_mutable_fit": (
            final_mutable_ratio <= config.maximum_final_mutable_nrmse_ratio_vs_local_always
        ),
    }
    return {
        "pass": bool(all(gates.values())),
        "gates": gates,
        "regression_damage_ratio_vs_local_always": float(damage_ratio),
        "committed_gain_ratio_vs_local_always": float(gain_ratio),
        "final_mutable_nrmse_ratio_vs_local_always": float(final_mutable_ratio),
        "pretraining": pretraining,
        "variant_summaries": {name: run["summary"] for name, run in variants.items()},
    }


def run_primary_seed(
    config: CoreValidation004Config,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    world = GrowthPlasticityWorld(config, seed=seed + 101)
    historical_x, historical_contexts = world.fixed_historical_inputs(
        examples_per_context=config.historical_examples_per_context,
        seed=seed + 151,
    )
    stream = world.transaction_stream(transactions=config.transactions, seed=seed + 181)
    pretrained, pretraining = pretrain_model(config, world, seed=seed + 1009, device=device)
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        variants[variant] = run_variant(
            pretrained,
            config,
            world,
            variant=variant,
            transaction_contexts=stream,
            seed=seed + 2003,
            historical_x=historical_x,
            historical_contexts=historical_contexts,
            device=device,
        )
    gate = summarize_seed(config, pretraining=pretraining, variants=variants)
    return {"seed": seed, "variants": variants, "gate_summary": gate}


def summarize_experiment(
    runs: list[dict[str, Any]], *, positive_status: str, negative_status: str
) -> dict[str, Any]:
    passed = all(bool(run["gate_summary"]["pass"]) for run in runs)
    return {
        "status": positive_status if passed else negative_status,
        "pass": passed,
        "scientific_decision": True,
        "passed_seeds": sum(bool(run["gate_summary"]["pass"]) for run in runs),
        "total_seeds": len(runs),
        "hypotheses": {
            "growth_restores_plasticity": all(
                run["gate_summary"]["gates"]["plasticity_recovery"] for run in runs
            ),
            "growth_preserves_dependency_scoped_safety": all(
                run["gate_summary"]["gates"]["scope_safety"]
                and run["gate_summary"]["gates"]["structural_locality"]
                for run in runs
            ),
            "growth_is_bounded_and_reusable": all(
                run["gate_summary"]["gates"]["bounded_growth"]
                and run["gate_summary"]["gates"]["private_cell_reuse"]
                for run in runs
            ),
        },
    }
