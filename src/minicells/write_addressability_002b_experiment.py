"""Sequential experiment and frozen gates for Core Validation 002B."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .write_addressability import SuperpositionWorld, _EPS
from .write_addressability_002b import (
    CoreValidation002BConfig,
    apply_global_ridge_write,
    apply_sparse_write,
    assembly_geometry,
    infer_sparse_write_address,
)
from .write_addressability_editing import (
    _edit_metrics,
    _gradient_edit,
    _mean,
    _median,
    _predictions,
    _retention_mse,
    make_edit_schedule,
)
from .write_addressability_models import SparseFunctionalModel, parameter_count, pretrain_models


def _assembly_name(width: int) -> str:
    return f"assembly_r{width}"


def _global_name(index: int) -> str:
    return f"global_ridge_{index}"


def _build_variants(
    pretrained: dict[str, nn.Module],
    config: CoreValidation002BConfig,
    *,
    device: torch.device,
) -> dict[str, nn.Module]:
    sparse = pretrained["sparse"]
    variants: dict[str, nn.Module] = {}
    for width in config.address_widths:
        variants[_assembly_name(width)] = copy.deepcopy(sparse)
    for index, _ in enumerate(config.global_scales):
        variants[_global_name(index)] = copy.deepcopy(sparse)
    variants["dense"] = copy.deepcopy(pretrained["dense"])
    variants["moe"] = copy.deepcopy(pretrained["moe"])
    for model in variants.values():
        model.to(device)
    for name, model in variants.items():
        if name.startswith("assembly_") or name.startswith("global_ridge_"):
            assert isinstance(model, SparseFunctionalModel)
            model.encoder.requires_grad_(False)
            model.reconstructor.requires_grad_(False)
    return variants


def oracle_latent_sanity(
    config: CoreValidation002BConfig,
    *,
    seed: int,
) -> dict[str, float]:
    """Execute exact evaluator-only z=s writes using the real writer orientation."""

    base = config.base
    world = SuperpositionWorld(base, seed=seed + 101)
    writer = world.V0.clone()
    schedule = make_edit_schedule(base, seed=seed + 701)
    generator = torch.Generator().manual_seed(seed + 809)
    updates: list[float] = []
    leakages: list[float] = []
    for task in schedule:
        _, affected, invariant, _ = world.edit_batches(task, generator=generator)
        after_writer = writer.clone()
        after_writer[:, task.target_feature] += task.delta
        delta_writer = after_writer - writer
        actual = affected.s @ delta_writer.transpose(0, 1)
        desired = affected.s[:, task.target_feature, None] * task.delta[None, :]
        signal = desired.square().mean().clamp_min(_EPS)
        updates.append(float(((actual - desired).square().mean() / signal).item()))
        invariant_change = invariant.s @ delta_writer.transpose(0, 1)
        leakages.append(float((invariant_change.square().mean() / signal).item()))
        writer = after_writer
        world.apply_edit(task)
    return {
        "median_update_error": _median(updates),
        "maximum_update_error": max(updates),
        "median_write_leakage": _median(leakages),
        "maximum_write_leakage": max(leakages),
    }


def run_sequential_edits(
    config: CoreValidation002BConfig,
    world: SuperpositionWorld,
    pretrained: dict[str, nn.Module],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    base = config.base
    world.reset_functions()
    variants = _build_variants(pretrained, config, device=device)
    schedule = make_edit_schedule(base, seed=seed + 701)
    data_generator = torch.Generator().manual_seed(seed + 809)
    retained: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    records: list[dict[str, Any]] = []

    for task in schedule:
        edit_cpu, affected_cpu, invariant_cpu, retention_cpu = world.edit_batches(
            task, generator=data_generator
        )
        edit = edit_cpu.to(device)
        affected = affected_cpu.to(device)
        invariant = invariant_cpu.to(device)
        pre = {
            name: (
                _predictions(model, edit),
                _predictions(model, affected),
                _predictions(model, invariant),
            )
            for name, model in variants.items()
        }

        inferred: dict[int, dict[str, Any]] = {}
        for width in config.address_widths:
            name = _assembly_name(width)
            model = variants[name]
            assert isinstance(model, SparseFunctionalModel)
            result = infer_sparse_write_address(
                model,
                edit.x,
                edit.y,
                width=width,
                als_steps=config.omp_als_steps,
                minimum_energy=base.address_min_energy,
            )
            result["writer_update_norm"] = apply_sparse_write(
                model, weights=result["weights"], delta=result["delta"]
            )
            inferred[width] = result

        global_updates: dict[int, float] = {}
        for index, scale in enumerate(config.global_scales):
            name = _global_name(index)
            model = variants[name]
            assert isinstance(model, SparseFunctionalModel)
            global_updates[index] = apply_global_ridge_write(
                model,
                edit.x,
                edit.y,
                ridge_lambda=config.global_ridge_lambda,
                scale=scale,
            )

        dense_loss = _gradient_edit(
            variants["dense"],
            edit,
            steps=base.dense_edit_steps,
            learning_rate=base.dense_edit_learning_rate,
        )
        moe_loss = _gradient_edit(
            variants["moe"],
            edit,
            steps=base.moe_edit_steps,
            learning_rate=base.moe_edit_learning_rate,
        )

        world.apply_edit(task)
        for name in retained:
            retained[name].append(retention_cpu.s.clone())

        repeated_target = any(
            earlier.target_feature == task.target_feature for earlier in schedule[: task.index]
        )
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
            record: dict[str, Any] = {
                "edit_index": task.index,
                "target_feature": task.target_feature,
                "repeated_target": repeated_target,
                "forced_distractor": task.forced_distractor,
                "variant": name,
                "variant_kind": "contextual",
                "address_width": None,
                "global_scale": None,
                "selected_support": None,
                "support_size": None,
                "rank1_residual_mse": None,
                "writer_update_norm": None,
                "assembly_fit_error": None,
                "assembly_off_support_energy_ratio": None,
                "assembly_context_ratio_variance": None,
                "assembly_target_correlation": None,
                **metrics,
                "retention_normalized_mse": _retention_mse(
                    model, retained[name], world, device
                ),
                "optimizer_final_mse": (
                    dense_loss if name == "dense" else moe_loss if name == "moe" else None
                ),
            }
            if name.startswith("assembly_r"):
                width = int(name.removeprefix("assembly_r"))
                result = inferred[width]
                record.update(
                    {
                        "variant_kind": "assembly",
                        "address_width": width,
                        "selected_support": result["support"],
                        "support_size": result["support_size"],
                        "rank1_residual_mse": result["rank1_residual_mse"],
                        "writer_update_norm": result["writer_update_norm"],
                        **assembly_geometry(
                            model,
                            affected,
                            invariant,
                            target_feature=task.target_feature,
                            weights=result["weights"],
                        ),
                    }
                )
            elif name.startswith("global_ridge_"):
                index = int(name.removeprefix("global_ridge_"))
                record.update(
                    {
                        "variant_kind": "global_ridge",
                        "global_scale": config.global_scales[index],
                        "writer_update_norm": global_updates[index],
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
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_variant.setdefault(str(record["variant"]), []).append(record)
    summary: dict[str, Any] = {}
    for variant, rows in by_variant.items():
        update = [float(row["update_error"]) for row in rows]
        leakage = [float(row["write_leakage"]) for row in rows]
        retention = [
            float(row["retention_normalized_mse"])
            for row in rows
            if row["retention_normalized_mse"] is not None
        ]
        repeated = [row for row in rows if bool(row["repeated_target"])]
        item: dict[str, Any] = {
            "variant_kind": rows[0]["variant_kind"],
            "address_width": rows[0]["address_width"],
            "global_scale": rows[0]["global_scale"],
            "mean_update_error": _mean(update),
            "median_update_error": _median(update),
            "mean_write_leakage": _mean(leakage),
            "median_write_leakage": _median(leakage),
            "final_retention_normalized_mse": retention[-1] if retention else None,
            "mean_repeated_target_update_error": (
                _mean([float(row["update_error"]) for row in repeated]) if repeated else None
            ),
            "mean_repeated_target_write_leakage": (
                _mean([float(row["write_leakage"]) for row in repeated]) if repeated else None
            ),
        }
        if rows[0]["variant_kind"] == "assembly":
            item.update(
                {
                    "median_assembly_fit_error": _median(
                        [float(row["assembly_fit_error"]) for row in rows]
                    ),
                    "median_assembly_off_support_energy_ratio": _median(
                        [float(row["assembly_off_support_energy_ratio"]) for row in rows]
                    ),
                    "median_assembly_context_ratio_variance": _median(
                        [float(row["assembly_context_ratio_variance"]) for row in rows]
                    ),
                    "mean_assembly_target_correlation": _mean(
                        [float(row["assembly_target_correlation"]) for row in rows]
                    ),
                    "mean_support_size": _mean([float(row["support_size"]) for row in rows]),
                }
            )
        summary[variant] = item
    return summary


def _match_global(summary: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    globals_ = [item for item in summary.values() if item["variant_kind"] == "global_ridge"]
    if not globals_:
        return {"matched": False, "u_gap": float("inf"), "leakage_ratio": float("inf")}
    chosen = min(
        globals_,
        key=lambda item: abs(float(item["median_update_error"]) - float(candidate["median_update_error"])),
    )
    gap = abs(float(chosen["median_update_error"]) - float(candidate["median_update_error"]))
    leakage_ratio = float(candidate["median_write_leakage"]) / max(
        float(chosen["median_write_leakage"]), _EPS
    )
    return {
        "matched": True,
        "global_scale": chosen["global_scale"],
        "candidate_median_u": candidate["median_update_error"],
        "global_median_u": chosen["median_update_error"],
        "global_median_l": chosen["median_write_leakage"],
        "u_gap": gap,
        "leakage_ratio": leakage_ratio,
    }


def decide_run(
    config: CoreValidation002BConfig,
    base_validation: dict[str, float],
    oracle: dict[str, float],
    summary: dict[str, Any],
) -> dict[str, Any]:
    candidates = [summary[_assembly_name(width)] for width in config.address_widths]
    width1 = summary[_assembly_name(1)]
    best = min(candidates, key=lambda item: float(item["median_update_error"]))
    best_width = int(best["address_width"])
    relative_update = float(best["median_update_error"]) / max(
        float(width1["median_update_error"]), _EPS
    )
    matched = _match_global(summary, best)

    gates = {
        "sparse_base_quality": float(base_validation["sparse"])
        <= config.maximum_sparse_base_normalized_mse,
        "oracle_latent_sanity": oracle["maximum_update_error"]
        <= config.maximum_oracle_update_error
        and oracle["maximum_write_leakage"] <= config.maximum_oracle_write_leakage,
        "sparse_assembly_update": float(best["median_update_error"])
        <= config.maximum_best_width_update_error,
        "width_improves_over_single_coordinate": best_width > 1
        and relative_update <= config.maximum_relative_update_error_vs_width1,
        "absolute_locality": float(best["median_write_leakage"])
        <= config.maximum_absolute_write_leakage,
        "matched_global_locality": bool(matched["matched"])
        and float(matched["u_gap"]) <= config.maximum_matched_u_gap
        and float(matched["leakage_ratio"])
        <= config.maximum_leakage_ratio_vs_matched_global,
        "repeated_edit_stability": best["mean_repeated_target_update_error"] is not None
        and float(best["mean_repeated_target_update_error"])
        <= config.maximum_repeated_update_error
        and float(best["mean_repeated_target_write_leakage"])
        <= config.maximum_repeated_write_leakage,
        "functional_geometry": float(best["median_assembly_fit_error"])
        <= config.maximum_assembly_fit_error,
    }
    passed = all(bool(value) for value in gates.values())
    return {
        **gates,
        "best_width": best_width,
        "best_median_update_error": best["median_update_error"],
        "best_median_write_leakage": best["median_write_leakage"],
        "width1_median_update_error": width1["median_update_error"],
        "relative_update_error_vs_width1": relative_update,
        "matched_global": matched,
        "pass": passed,
    }


def run_primary_seed(
    config: CoreValidation002BConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    world, pretrained, pretraining = pretrain_models(config.base, seed=seed, device=device)
    sequential = run_sequential_edits(
        config, world, pretrained, seed=seed, device=device
    )
    summary = summarize_records(sequential["records"])
    oracle = oracle_latent_sanity(config, seed=seed)
    gates = decide_run(config, pretraining["base_normalized_mse"], oracle, summary)
    return {
        "seed": seed,
        "config": {"base": asdict(config.base), "address_widths": list(config.address_widths)},
        "superposition_load": config.base.superposition_load,
        "recovery_load": config.base.recovery_load,
        "parameter_counts": {name: parameter_count(model) for name, model in pretrained.items()},
        "pretraining": pretraining,
        "oracle_latent_sanity": oracle,
        "schedule": sequential["schedule"],
        "summary": summary,
        "gates": gates,
        "records": sequential["records"],
    }


def summarize_experiment(
    runs: list[dict[str, Any]],
    *,
    positive_status: str,
    negative_status: str,
) -> dict[str, Any]:
    passed = [bool(run["gates"]["pass"]) for run in runs]
    return {
        "status": positive_status if runs and all(passed) else negative_status,
        "pass": bool(runs) and all(passed),
        "scientific_decision": True,
        "passed_seeds": sum(passed),
        "total_seeds": len(runs),
        "require_all_seeds": True,
    }
