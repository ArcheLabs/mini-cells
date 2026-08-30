"""Transactional variants and frozen gates for Core Validation 003."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import torch

from .dependency_scoped_config import CoreValidation003Config, _EPS
from .dependency_scoped_model import RoutedCellModel
from .dependency_scoped_world import RoutedContinualWorld
from .dependency_scoped_ops import (
    evaluate_candidate,
    evaluate_final_state,
    pretrain_model,
    train_candidate,
)

VARIANTS = (
    "standard_moe_always",
    "local_always",
    "local_tx_frozen",
    "local_tx_router_drift",
)


def run_variant(
    pretrained: RoutedCellModel,
    config: CoreValidation003Config,
    world: RoutedContinualWorld,
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

        update_shared = variant == "standard_moe_always"
        candidate = train_candidate(
            model,
            config,
            world,
            amplitudes=new_amplitudes,
            target_context=target_context,
            seed=seed + 10000 + index,
            device=device,
            update_shared=update_shared,
        )
        if variant == "local_tx_router_drift":
            candidate.perturb_router(
                seed=seed + 20000 + index,
                noise_scale=config.router_drift_noise,
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
        )

        if variant in {"standard_moe_always", "local_always"}:
            commit = True
        else:
            commit = bool(metrics["local_pass"])
        if commit:
            model = candidate

        amplitudes = new_amplitudes
        records.append(
            {
                "transaction": index,
                "variant": variant,
                "commit": commit,
                **metrics,
            }
        )

    final = evaluate_final_state(
        model,
        config,
        world,
        amplitudes=amplitudes,
        seed=seed + 40000,
        device=device,
    )
    committed = [r for r in records if r["commit"]]
    local_passes = [r for r in records if r["local_pass"]]
    cumulative_damage = sum(max(0.0, float(r["global_regression"])) for r in committed)
    cumulative_gain = sum(max(0.0, float(r["new_gain_fraction"])) for r in committed)
    state_validation_cost = sum(
        float(r["candidate_state_fraction"]) + float(r["dependency_coverage"])
        for r in records
    )
    summary = {
        "variant": variant,
        "acceptance_rate": len(committed) / max(len(records), 1),
        "false_safe_rate": (
            sum(bool(r["false_safe"]) for r in local_passes) / max(len(local_passes), 1)
        ),
        "mean_dependency_coverage": float(
            sum(float(r["dependency_coverage"]) for r in records) / max(len(records), 1)
        ),
        "maximum_structural_escape_rate": float(
            max((float(r["structural_escape_rate"]) for r in records), default=0.0)
        ),
        "mean_routing_drift_rate": float(
            sum(float(r["routing_drift_rate"]) for r in records) / max(len(records), 1)
        ),
        "cumulative_positive_global_regression": cumulative_damage,
        "cumulative_committed_new_gain": cumulative_gain,
        "normalized_state_validation_cost_per_accepted_update": (
            state_validation_cost / max(len(committed), 1)
        ),
        **final,
    }
    return {"summary": summary, "records": records}


def summarize_granularity(
    config: CoreValidation003Config,
    *,
    granularity: int,
    pretraining: dict[str, float],
    variant_runs: dict[str, dict[str, Any]],
    coarsest_dependency_coverage: float | None,
) -> dict[str, Any]:
    tx = variant_runs["local_tx_frozen"]["summary"]
    always = variant_runs["local_always"]["summary"]
    base_quality = pretraining["base_normalized_mse"] <= config.maximum_base_normalized_mse

    damage_ratio = (
        tx["cumulative_positive_global_regression"]
        / max(always["cumulative_positive_global_regression"], _EPS)
    )
    gain_ratio = (
        tx["cumulative_committed_new_gain"]
        / max(always["cumulative_committed_new_gain"], _EPS)
    )
    dependency_ratio = (
        tx["mean_dependency_coverage"] / max(coarsest_dependency_coverage, _EPS)
        if coarsest_dependency_coverage is not None
        else 1.0
    )

    gates = {
        "base_quality": bool(base_quality),
        "scope_safety": tx["false_safe_rate"] <= config.maximum_false_safe_rate,
        "dependency_scope": tx["mean_dependency_coverage"]
        <= config.maximum_dependency_coverage,
        "structural_locality": tx["maximum_structural_escape_rate"]
        <= config.maximum_structural_escape_rate,
        "acceptance": tx["acceptance_rate"] >= config.minimum_acceptance_rate,
        "transactional_regression_reduction": damage_ratio
        <= config.maximum_regression_damage_ratio_vs_local_always,
        "new_learning_retention": gain_ratio
        >= config.minimum_committed_gain_ratio_vs_local_always,
        "granularity_reduces_dependency": (
            granularity == 1
            or dependency_ratio <= config.maximum_dependency_ratio_vs_coarsest
        ),
    }
    return {
        "granularity": granularity,
        "pass": bool(granularity > 1 and all(gates.values())),
        "gates": gates,
        "regression_damage_ratio_vs_local_always": float(damage_ratio),
        "committed_gain_ratio_vs_local_always": float(gain_ratio),
        "dependency_ratio_vs_coarsest": float(dependency_ratio),
        "base_normalized_mse": float(pretraining["base_normalized_mse"]),
        "local_tx_summary": tx,
        "router_drift_stress_summary": variant_runs["local_tx_router_drift"]["summary"],
    }


def smoke_config(config: CoreValidation003Config) -> CoreValidation003Config:
    return replace(
        config,
        num_contexts=12,
        anchor_contexts=3,
        content_dim=8,
        model_dim=8,
        output_dim=3,
        basis_hidden=2,
        residual_hidden=2,
        router_dim=4,
        base_expert_hidden=16,
        granularities=(1, 4),
        pretrain_steps=8,
        pretrain_batch_size=32,
        pretrain_validation_examples=64,
        transactions=6,
        update_train_examples=12,
        update_validation_examples=24,
        update_steps=3,
        historical_examples_per_context=4,
    )
