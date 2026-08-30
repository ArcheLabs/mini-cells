"""Sequential controls and decision gates for Core Validation 002."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .write_addressability import (
    SuperpositionWorld,
    VariantName,
    WriteAddressabilityConfig,
    _EPS,
)
from .write_addressability_editing import (
    _cyclic_permutation,
    _edit_metrics,
    _gradient_edit,
    _log_log_slope,
    _mean,
    _mechanistic_metrics,
    _median,
    _pearson,
    _predictions,
    _retention_mse,
    _variant_models,
    make_edit_schedule,
)
from .write_addressability_models import (
    SparseFunctionalModel,
    apply_addressed_write,
    infer_write_address,
    latent_feature_correlations,
    least_squares_delta_for_address,
    parameter_count,
    pretrain_models,
)


def run_sequential_edits(
    config: WriteAddressabilityConfig,
    world: SuperpositionWorld,
    pretrained: dict[str, nn.Module],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    world.reset_functions()
    variants = _variant_models(pretrained, config, device=device)
    sparse_reference = variants["inferred_address"]
    assert isinstance(sparse_reference, SparseFunctionalModel)
    correlations = latent_feature_correlations(
        sparse_reference,
        world,
        config,
        seed=seed + 509,
        device=device,
    )
    oracle_address = correlations.argmax(dim=0)
    permutation = _cyclic_permutation(config.latent_dim, seed=seed + 601)
    schedule = make_edit_schedule(config, seed=seed + 701)
    data_generator = torch.Generator().manual_seed(seed + 809)
    retained: dict[VariantName, list[torch.Tensor]] = {name: [] for name in variants}
    records: list[dict[str, Any]] = []

    for task in schedule:
        edit_cpu, affected_cpu, invariant_cpu, retention_cpu = world.edit_batches(
            task,
            generator=data_generator,
        )
        edit = edit_cpu.to(device)
        affected = affected_cpu.to(device)
        invariant = invariant_cpu.to(device)

        pre: dict[VariantName, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, model in variants.items():
            pre[name] = (
                _predictions(model, edit),
                _predictions(model, affected),
                _predictions(model, invariant),
            )

        inferred = variants["inferred_address"]
        assert isinstance(inferred, SparseFunctionalModel)
        inferred_result = infer_write_address(inferred, edit.x, edit.y, config)
        inferred_q = int(inferred_result["address"])
        apply_addressed_write(
            inferred,
            address=inferred_q,
            delta=inferred_result["delta"],
        )

        oracle_model = variants["oracle_address"]
        assert isinstance(oracle_model, SparseFunctionalModel)
        oracle_q = int(oracle_address[task.target_feature].item())
        oracle_delta = least_squares_delta_for_address(oracle_model, edit.x, edit.y, oracle_q)
        apply_addressed_write(oracle_model, address=oracle_q, delta=oracle_delta)

        permuted = variants["permuted_address"]
        assert isinstance(permuted, SparseFunctionalModel)
        permuted_result = infer_write_address(permuted, edit.x, edit.y, config)
        permuted_q = int(permuted_result["address"])
        permuted_destination = int(permutation[permuted_q].item())
        apply_addressed_write(
            permuted,
            address=permuted_q,
            destination=permuted_destination,
            delta=permuted_result["delta"],
        )

        global_model = variants["global_write"]
        assert isinstance(global_model, SparseFunctionalModel)
        global_loss = _gradient_edit(
            global_model,
            edit,
            steps=config.global_edit_steps,
            learning_rate=config.global_edit_learning_rate,
            parameters=[global_model.writer.weight],
        )
        dense_loss = _gradient_edit(
            variants["dense"],
            edit,
            steps=config.dense_edit_steps,
            learning_rate=config.dense_edit_learning_rate,
        )
        moe_loss = _gradient_edit(
            variants["moe"],
            edit,
            steps=config.moe_edit_steps,
            learning_rate=config.moe_edit_learning_rate,
        )

        # Advance evaluator ground truth only after every model has consumed the same edit data.
        world.apply_edit(task)
        for name in retained:
            retained[name].append(retention_cpu.s.clone())

        for name, model in variants.items():
            metrics = _edit_metrics(
                model,
                edit,
                affected,
                invariant,
                target_feature=task.target_feature,
                delta=task.delta.to(device),
                pre_edit_prediction=pre[name][0],
                pre_affected_prediction=pre[name][1],
                pre_invariant_prediction=pre[name][2],
            )
            address: int | None = None
            destination: int | None = None
            address_score: float | None = None
            active_fraction: float | None = None
            if name == "inferred_address":
                address = inferred_q
                destination = inferred_q
                address_score = float(inferred_result["score"])
                active_fraction = float(inferred_result["active_fraction"])
            elif name == "oracle_address":
                address = oracle_q
                destination = oracle_q
            elif name == "permuted_address":
                address = permuted_q
                destination = permuted_destination
                address_score = float(permuted_result["score"])
                active_fraction = float(permuted_result["active_fraction"])

            record: dict[str, Any] = {
                "edit_index": task.index,
                "target_feature": task.target_feature,
                "repeated_target": sum(
                    earlier.target_feature == task.target_feature
                    for earlier in schedule[: task.index]
                )
                > 0,
                "forced_distractor": task.forced_distractor,
                "variant": name,
                **metrics,
                "retention_normalized_mse": _retention_mse(
                    model, retained[name], world, device
                ),
                "address": address,
                "write_destination": destination,
                "address_score": address_score,
                "address_active_fraction": active_fraction,
                "oracle_address": (
                    oracle_q
                    if name in {"inferred_address", "oracle_address", "permuted_address"}
                    else None
                ),
                "address_matches_oracle": (
                    bool(address == oracle_q) if address is not None else None
                ),
                "selected_target_correlation": (
                    float(correlations[address, task.target_feature].item())
                    if address is not None
                    else None
                ),
                "optimizer_final_mse": (
                    global_loss
                    if name == "global_write"
                    else dense_loss
                    if name == "dense"
                    else moe_loss
                    if name == "moe"
                    else None
                ),
            }
            if (
                name in {"inferred_address", "oracle_address", "permuted_address"}
                and address is not None
            ):
                record.update(
                    _mechanistic_metrics(
                        model if isinstance(model, SparseFunctionalModel) else sparse_reference,
                        invariant,
                        affected,
                        target_feature=task.target_feature,
                        address=address,
                    )
                )
            else:
                record.update(
                    {
                        "off_support_squared_activation": None,
                        "target_coefficient_squared": None,
                        "leakage_proxy": None,
                    }
                )
            records.append(record)

    return {
        "records": records,
        "schedule": [
            {
                "edit_index": task.index,
                "target_feature": task.target_feature,
                "forced_distractor": task.forced_distractor,
                "delta_norm": float(task.delta.norm().item()),
            }
            for task in schedule
        ],
        "oracle_mapping": {
            "mean_best_correlation": float(correlations.max(dim=0).values.mean().item()),
            "median_best_correlation": float(correlations.max(dim=0).values.median().item()),
            "unique_latent_addresses": int(torch.unique(oracle_address).numel()),
        },
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_variant.setdefault(str(record["variant"]), []).append(record)
    summary: dict[str, Any] = {}
    for variant, rows in by_variant.items():
        update = [float(row["update_error"]) for row in rows]
        leakage = [float(row["write_leakage"]) for row in rows]
        edit_fit = [float(row["edit_normalized_mse"]) for row in rows]
        retention = [
            float(row["retention_normalized_mse"])
            for row in rows
            if row["retention_normalized_mse"] is not None
        ]
        repeated = [row for row in rows if bool(row["repeated_target"])]
        variant_summary: dict[str, Any] = {
            "mean_update_error": _mean(update),
            "median_update_error": _median(update),
            "mean_write_leakage": _mean(leakage),
            "median_write_leakage": _median(leakage),
            "mean_edit_normalized_mse": _mean(edit_fit),
            "final_retention_normalized_mse": retention[-1] if retention else None,
            "mean_repeated_target_update_error": (
                _mean([float(row["update_error"]) for row in repeated]) if repeated else None
            ),
            "mean_repeated_target_write_leakage": (
                _mean([float(row["write_leakage"]) for row in repeated]) if repeated else None
            ),
        }
        proxy_rows = [
            row
            for row in rows
            if row.get("leakage_proxy") is not None
            and math.isfinite(float(row["leakage_proxy"]))
            and math.isfinite(float(row["write_leakage"]))
        ]
        if proxy_rows:
            proxies = [float(row["leakage_proxy"]) for row in proxy_rows]
            leakages = [float(row["write_leakage"]) for row in proxy_rows]
            variant_summary["mechanistic_pearson"] = _pearson(proxies, leakages)
            variant_summary["mechanistic_log_log_slope"] = _log_log_slope(proxies, leakages)
        else:
            variant_summary["mechanistic_pearson"] = None
            variant_summary["mechanistic_log_log_slope"] = None
        addresses = [row for row in rows if row.get("address") is not None]
        variant_summary["address_match_rate"] = (
            _mean([1.0 if row["address_matches_oracle"] else 0.0 for row in addresses])
            if addresses
            else None
        )
        variant_summary["mean_selected_target_correlation"] = (
            _mean([float(row["selected_target_correlation"]) for row in addresses])
            if addresses
            else None
        )
        summary[variant] = variant_summary
    return summary


def decide_run(
    config: WriteAddressabilityConfig,
    base_validation: dict[str, float],
    summary: dict[str, Any],
) -> dict[str, Any]:
    candidate = summary["inferred_address"]
    baseline = summary["global_write"]
    permutation = summary["permuted_address"]
    base_gate = (
        base_validation["sparse"] <= config.maximum_base_normalized_mse
        and base_validation["dense"] <= config.maximum_base_normalized_mse
        and base_validation["moe"] <= config.maximum_base_normalized_mse
    )
    candidate_update_gate = (
        candidate["median_update_error"] <= config.maximum_candidate_update_error
    )
    baseline_valid = baseline["median_update_error"] <= config.maximum_baseline_update_error
    leakage_ratio = candidate["median_write_leakage"] / max(
        baseline["median_write_leakage"], _EPS
    )
    locality_gate = baseline_valid and leakage_ratio <= config.maximum_leakage_ratio
    correlation = candidate.get("mechanistic_pearson")
    mechanism_gate = (
        correlation is not None and correlation >= config.minimum_mechanistic_correlation
    )
    candidate_joint = candidate["median_update_error"] + candidate["median_write_leakage"]
    permuted_joint = permutation["median_update_error"] + permutation["median_write_leakage"]
    permutation_degradation = permuted_joint / max(candidate_joint, _EPS)
    permutation_gate = permutation_degradation >= config.minimum_permutation_degradation
    passed = bool(
        base_gate
        and candidate_update_gate
        and baseline_valid
        and locality_gate
        and mechanism_gate
        and permutation_gate
    )
    return {
        "base_quality": base_gate,
        "candidate_update_generalization": candidate_update_gate,
        "global_write_baseline_valid": baseline_valid,
        "locality_advantage": locality_gate,
        "mechanistic_prediction": mechanism_gate,
        "permutation_control": permutation_gate,
        "leakage_ratio_vs_global_write": leakage_ratio,
        "permutation_joint_error_ratio": permutation_degradation,
        "pass": passed,
    }


def run_primary_seed(
    config: WriteAddressabilityConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    world, pretrained, pretraining = pretrain_models(config, seed=seed, device=device)
    sequential = run_sequential_edits(
        config,
        world,
        pretrained,
        seed=seed,
        device=device,
    )
    summary = summarize_records(sequential["records"])
    gates = decide_run(config, pretraining["base_normalized_mse"], summary)
    return {
        "seed": seed,
        "config": asdict(config),
        "superposition_load": config.superposition_load,
        "recovery_load": config.recovery_load,
        "parameter_counts": {name: parameter_count(model) for name, model in pretrained.items()},
        "pretraining": pretraining,
        "oracle_mapping": sequential["oracle_mapping"],
        "schedule": sequential["schedule"],
        "summary": summary,
        "gates": gates,
        "records": sequential["records"],
    }


def summarize_experiment(
    runs: list[dict[str, Any]],
    *,
    positive_status: str = "WRITE_ADDRESSABILITY_SUPPORTED",
    negative_status: str = "WRITE_ADDRESSABILITY_NOT_SUPPORTED",
) -> dict[str, Any]:
    if not runs:
        return {
            "status": negative_status,
            "pass": False,
            "reason": "no runs",
            "passed_seeds": 0,
            "total_seeds": 0,
        }
    passed = [bool(run["gates"]["pass"]) for run in runs]
    return {
        "status": positive_status if all(passed) else negative_status,
        "pass": all(passed),
        "passed_seeds": sum(passed),
        "total_seeds": len(passed),
        "require_all_seeds": True,
    }


def oracle_exact_zero_check(
    *,
    num_features: int = 32,
    active_features: int = 4,
    output_dim: int = 8,
    examples: int = 256,
    seed: int = 1,
) -> dict[str, float]:
    """Analytic implementation check for z=s and one-column writes."""

    generator = torch.Generator().manual_seed(seed)
    target = 3
    s = torch.zeros(examples, num_features)
    for row in range(examples):
        candidates = torch.tensor([i for i in range(num_features) if i != target])
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:active_features]]
        s[row, chosen] = torch.randn(active_features, generator=generator)
    writer = torch.randn(output_dim, num_features, generator=generator)
    delta = torch.randn(output_dim, generator=generator)
    before = s @ writer.transpose(0, 1)
    after_writer = writer.clone()
    after_writer[:, target] += delta
    after = s @ after_writer.transpose(0, 1)
    invariant_change = (after - before).abs().max()
    return {"max_invariant_change": float(invariant_change.item())}
