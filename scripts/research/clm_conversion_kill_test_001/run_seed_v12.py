from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = ROOT / "scripts" / "research" / "clm_conversion_kill_test_001"
for path in (ROOT, MODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.research.clm_conversion_kill_test_001 import run_seed as core  # noqa: E402
from scripts.research.clm_conversion_kill_test_001.dataset import (  # noqa: E402
    ENTITIES,
    PROTOCOLS,
    REGIONS,
    calibration_prompts,
    contextual_conflict_rows,
    formation_evaluation,
    formation_validation,
)
from scripts.research.clm_conversion_kill_test_001.semantic_choice import (  # noqa: E402
    candidate_choice_metrics,
)


def _choice(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    candidates: tuple[str, ...],
    protocol: dict[str, Any],
    device: str,
    overlay: Any | None,
) -> dict[str, Any]:
    return candidate_choice_metrics(
        model,
        tokenizer,
        rows,
        candidates,
        protocol=protocol,
        device=device,
        overlay=overlay,
        sequence_module=core.seq,
    )


def _branch_phase_v12(
    *,
    model: Any,
    tokenizer: Any,
    overlay: Any,
    formation_snapshot: dict[str, torch.Tensor],
    modal: dict[str, int],
    agreement: dict[str, float],
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    local_cfg = protocol["training"]["local_mutation"]
    entity_cells = {int(modal[f"entity.{entity}"]) for entity in ENTITIES}
    if len(entity_cells) < 2:
        overlay.restore_(formation_snapshot)
        return {
            "branches": [],
            "minimum_standalone_nll_gain": 0.0,
            "minimum_standalone_choice_accuracy": 0.0,
            "minimum_merge_retention": 0.0,
            "minimum_merged_choice_accuracy": 0.0,
            "exact_rollback": core._state_equal(formation_snapshot, overlay),
            "skipped_reason": "fewer than two distinct entity primary Cells",
        }

    branch_a = core._choose_rewrite_target(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
    )
    branch_b = core._choose_rewrite_target(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
        forbidden_cells={int(branch_a["cell"])},
    )

    branches: list[dict[str, Any]] = []
    mutations = []
    standalone_gains: list[float] = []
    standalone_choices: list[float] = []
    for label, target in (("A", branch_a), ("B", branch_b)):
        overlay.restore_(formation_snapshot)
        rows = target["rows"]["evaluation"]
        before = core._evaluate(model, tokenizer, rows, protocol, device, overlay)
        before_choice = _choice(
            model, tokenizer, rows, PROTOCOLS, protocol, device, overlay
        )
        core._train_cell(
            model=model,
            tokenizer=tokenizer,
            overlay=overlay,
            rows=target["rows"]["train"],
            cell_index=int(target["cell"]),
            protocol=protocol,
            device=device,
            steps=int(local_cfg["steps"]),
            learning_rate=float(local_cfg["learning_rate"]),
            max_gradient_norm=float(local_cfg["max_gradient_norm"]),
            learn_key=False,
        )
        after = core._evaluate(model, tokenizer, rows, protocol, device, overlay)
        after_choice = _choice(
            model, tokenizer, rows, PROTOCOLS, protocol, device, overlay
        )
        standalone_gain = core._nll_gain(before, after)
        standalone_choice = float(after_choice["strict_choice_accuracy"])
        standalone_gains.append(standalone_gain)
        standalone_choices.append(standalone_choice)
        mutations.append(overlay.export_mutation(int(target["cell"])))
        branches.append(
            {
                "label": label,
                "entity": target["entity"],
                "cell": int(target["cell"]),
                "before": core._metric_dict(before),
                "before_choice": before_choice,
                "standalone": core._metric_dict(after),
                "standalone_choice": after_choice,
                "standalone_nll_gain": standalone_gain,
                "rows": target["rows"],
            }
        )

    core.disjoint_mutations(*mutations)
    overlay.restore_(formation_snapshot)
    for mutation in mutations:
        overlay.apply_mutation_(mutation)

    retentions: list[float] = []
    merged_choices: list[float] = []
    for branch in branches:
        rows = branch["rows"]["evaluation"]
        merged = core._evaluate(model, tokenizer, rows, protocol, device, overlay)
        merged_choice = _choice(
            model, tokenizer, rows, PROTOCOLS, protocol, device, overlay
        )
        branch["merged"] = core._metric_dict(merged)
        branch["merged_choice"] = merged_choice
        merged_gain = float(
            branch["before"]["mean_reference_nll"] - merged["mean_reference_nll"]
        )
        standalone_gain = float(branch["standalone_nll_gain"])
        retention = merged_gain / standalone_gain if standalone_gain > 0.0 else 0.0
        branch["merged_nll_gain"] = merged_gain
        branch["merge_retention"] = retention
        retentions.append(retention)
        merged_choices.append(float(merged_choice["strict_choice_accuracy"]))
        branch.pop("rows", None)

    overlay.restore_(formation_snapshot)
    return {
        "branches": branches,
        "minimum_standalone_nll_gain": min(standalone_gains) if standalone_gains else 0.0,
        "minimum_standalone_choice_accuracy": (
            min(standalone_choices) if standalone_choices else 0.0
        ),
        "minimum_merge_retention": min(retentions) if retentions else 0.0,
        "minimum_merged_choice_accuracy": min(merged_choices) if merged_choices else 0.0,
        "exact_rollback": core._state_equal(formation_snapshot, overlay),
    }


def run(seed: int, device: str) -> dict[str, Any]:
    protocol = core._load_json(core.PROTOCOL_PATH)
    if float(protocol["protocol_version"]) != 1.2:
        raise RuntimeError("run_seed_v12.py requires frozen protocol_version 1.2")
    if core._git_blob_sha(core.DATASET_PATH) != protocol["dataset"]["generator_git_blob_sha"]:
        raise RuntimeError("controlled dataset generator identity mismatch")
    if seed not in [int(value) for value in protocol["formal_seeds"]]:
        raise RuntimeError(f"seed {seed} is not a frozen formal seed")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    random.seed(seed)
    core._seed_everything(seed)

    import transformers

    transformers.logging.set_verbosity_error()
    model_id = protocol["base"]["model_id"]
    revision = protocol["base"]["revision"]
    core._progress(seed, f"loading frozen foundation {model_id}@{revision[:12]}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
    ).to(device)
    core.freeze_foundation_(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("foundation freeze invariant failed")

    substrate = protocol["substrate"]
    overlay = core.FunctionalCellOverlay(
        hidden_size=int(model.config.hidden_size),
        layer_indices=tuple(int(value) for value in substrate["layer_indices"]),
        max_cells=int(substrate["max_cells"]),
        initial_active_cells=int(substrate["initial_active_cells"]),
        rank=int(substrate["rank"]),
        temperature=float(substrate["temperature"]),
        top_k=int(substrate["top_k"]),
        seed=seed,
    ).to(device=device, dtype=torch.float32)
    if not overlay.zero_output_is_exact():
        raise RuntimeError("overlay must start with exact zero residual output")

    validation_rows = formation_validation()
    eval_rows = formation_evaluation()
    history = list(calibration_prompts())
    base_validation = core._evaluate(model, tokenizer, validation_rows, protocol, device, None)
    base_metrics = {
        name: core._evaluate(model, tokenizer, rows, protocol, device, None)
        for name, rows in eval_rows.items()
        if name != "routing"
    }
    base_choice = {
        "direct": _choice(
            model, tokenizer, eval_rows["direct"], PROTOCOLS, protocol, device, None
        ),
        "negation": _choice(
            model, tokenizer, eval_rows["negation"], PROTOCOLS, protocol, device, None
        ),
        "relation": _choice(
            model, tokenizer, eval_rows["relation"], REGIONS, protocol, device, None
        ),
    }
    history_teacher = core._last_logits(model, tokenizer, history, device).detach().cpu()

    compatibility_prompts = history[:4]
    repeated_a = core._last_logits(model, tokenizer, compatibility_prompts, device)
    repeated_b = core._last_logits(model, tokenizer, compatibility_prompts, device)
    converted = core._last_logits(model, tokenizer, compatibility_prompts, device, overlay)
    repeatability = float((repeated_a - repeated_b).abs().max().item())
    converted_delta = float((repeated_a - converted).abs().max().item())
    compatibility_excess = max(0.0, converted_delta - repeatability)
    compatibility = {
        "foundation_repeatability_max_abs_logit_delta": repeatability,
        "zero_overlay_max_abs_logit_delta": converted_delta,
        "excess_over_repeatability": compatibility_excess,
        "zero_output_exact": overlay.zero_output_is_exact(),
    }
    core._progress(seed, f"compatibility excess={compatibility_excess:.3e}")

    formation_log = core._train_formation(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        protocol=protocol,
        seed=seed,
        device=device,
        base_validation=base_validation,
        validation_rows=validation_rows,
        history_prompts=history,
        history_teacher=history_teacher,
    )
    formation_snapshot = overlay.snapshot()
    selected_validation = core._evaluate(
        model, tokenizer, validation_rows, protocol, device, overlay
    )
    formation_metrics = {
        name: core._evaluate(model, tokenizer, rows, protocol, device, overlay)
        for name, rows in eval_rows.items()
        if name != "routing"
    }
    formation_gains = {
        name: core._nll_gain(base_metrics[name], formation_metrics[name])
        for name in ("direct", "negation", "relation")
    }
    formation_choice = {
        "direct": _choice(
            model, tokenizer, eval_rows["direct"], PROTOCOLS, protocol, device, overlay
        ),
        "negation": _choice(
            model, tokenizer, eval_rows["negation"], PROTOCOLS, protocol, device, overlay
        ),
        "relation": _choice(
            model, tokenizer, eval_rows["relation"], REGIONS, protocol, device, overlay
        ),
    }
    formation_history_kl = core._history_kl(
        model, tokenizer, history, history_teacher, device, overlay
    )
    route_metrics, modal, agreement = core._routing_metrics(
        model,
        tokenizer,
        eval_rows["routing"],
        protocol,
        device,
        overlay,
    )
    core._progress(
        seed,
        "formation "
        f"direct={formation_gains['direct']:.3f}/choice={formation_choice['direct']['strict_choice_accuracy']:.3f} "
        f"negation={formation_gains['negation']:.3f}/choice={formation_choice['negation']['strict_choice_accuracy']:.3f} "
        f"relation={formation_gains['relation']:.3f}/choice={formation_choice['relation']['strict_choice_accuracy']:.3f}",
    )

    local_cfg = protocol["training"]["local_mutation"]
    local_target = core._choose_rewrite_target(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
    )
    local_rows = local_target["rows"]["evaluation"]
    local_base = core._evaluate(model, tokenizer, local_rows, protocol, device, overlay)
    local_choice_before = _choice(
        model, tokenizer, local_rows, PROTOCOLS, protocol, device, overlay
    )
    unrelated_rows = [
        row
        for row in eval_rows["direct"]
        if row["concept_id"] != local_target["concept_id"]
    ]
    unrelated_base = core._evaluate(model, tokenizer, unrelated_rows, protocol, device, overlay)
    unrelated_choice_before = _choice(
        model, tokenizer, unrelated_rows, PROTOCOLS, protocol, device, overlay
    )
    core._train_cell(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        rows=local_target["rows"]["train"],
        cell_index=int(local_target["cell"]),
        protocol=protocol,
        device=device,
        steps=int(local_cfg["steps"]),
        learning_rate=float(local_cfg["learning_rate"]),
        max_gradient_norm=float(local_cfg["max_gradient_norm"]),
        learn_key=False,
    )
    local_after = core._evaluate(model, tokenizer, local_rows, protocol, device, overlay)
    local_choice_after = _choice(
        model, tokenizer, local_rows, PROTOCOLS, protocol, device, overlay
    )
    unrelated_after = core._evaluate(model, tokenizer, unrelated_rows, protocol, device, overlay)
    unrelated_choice_after = _choice(
        model, tokenizer, unrelated_rows, PROTOCOLS, protocol, device, overlay
    )
    local_result = {
        "entity": local_target["entity"],
        "cell": int(local_target["cell"]),
        "route_agreement": float(local_target["agreement"]),
        "rewrite_train_target_fraction": float(local_target["train_target_fraction"]),
        "semantic_write_nll_gain": core._nll_gain(local_base, local_after),
        "semantic_choice_before": local_choice_before,
        "semantic_choice_after": local_choice_after,
        "unrelated_nll_regression": float(
            unrelated_after["mean_reference_nll"] - unrelated_base["mean_reference_nll"]
        ),
        "unrelated_choice_before": unrelated_choice_before,
        "unrelated_choice_after": unrelated_choice_after,
        "before": core._metric_dict(local_base),
        "after": core._metric_dict(local_after),
    }
    overlay.restore_(formation_snapshot)
    local_result["rollback_exact"] = core._state_equal(formation_snapshot, overlay)

    growth_cfg = protocol["training"]["growth"]
    growth_entity = str(local_target["entity"])
    growth_parent = int(local_target["cell"])
    old_protocol = str(local_target["old_protocol"])
    new_protocol = PROTOCOLS[(PROTOCOLS.index(old_protocol) + 2) % len(PROTOCOLS)]
    growth_rows = contextual_conflict_rows(growth_entity, old_protocol, new_protocol)
    alpha_base = core._evaluate(model, tokenizer, growth_rows["alpha"], protocol, device, overlay)
    beta_base = core._evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    alpha_choice_base = _choice(
        model, tokenizer, growth_rows["alpha"], PROTOCOLS, protocol, device, overlay
    )
    beta_choice_base = _choice(
        model, tokenizer, growth_rows["beta_eval"], PROTOCOLS, protocol, device, overlay
    )

    core._train_cell(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        rows=growth_rows["beta_train"],
        cell_index=growth_parent,
        protocol=protocol,
        device=device,
        steps=int(growth_cfg["steps"]),
        learning_rate=float(local_cfg["learning_rate"]),
        max_gradient_norm=float(local_cfg["max_gradient_norm"]),
        learn_key=False,
    )
    parent_alpha = core._evaluate(
        model, tokenizer, growth_rows["alpha"], protocol, device, overlay
    )
    parent_beta = core._evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    parent_alpha_choice = _choice(
        model, tokenizer, growth_rows["alpha"], PROTOCOLS, protocol, device, overlay
    )
    parent_beta_choice = _choice(
        model, tokenizer, growth_rows["beta_eval"], PROTOCOLS, protocol, device, overlay
    )
    parent_control = {
        "alpha_nll_regression": float(
            parent_alpha["mean_reference_nll"] - alpha_base["mean_reference_nll"]
        ),
        "alpha_choice": parent_alpha_choice,
        "beta_nll_gain": core._nll_gain(beta_base, parent_beta),
        "beta_choice": parent_beta_choice,
    }

    gates_cfg = protocol["gates"]
    parent_control["satisfies_registered_conflict_solution"] = bool(
        float(parent_control["alpha_nll_regression"])
        <= float(gates_cfg["maximum_growth_alpha_nll_regression"])
        and float(parent_alpha_choice["strict_choice_accuracy"])
        >= float(gates_cfg["minimum_growth_alpha_choice_accuracy"])
        and float(parent_control["beta_nll_gain"])
        >= float(gates_cfg["minimum_growth_beta_nll_gain"])
        and float(parent_beta_choice["strict_choice_accuracy"])
        >= float(gates_cfg["minimum_growth_beta_choice_accuracy"])
    )

    overlay.restore_(formation_snapshot)
    child = overlay.spawn_child(growth_parent)
    alpha_spawned = core._evaluate(
        model, tokenizer, growth_rows["alpha"], protocol, device, overlay
    )
    beta_spawned = core._evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    alpha_choice_spawned = _choice(
        model, tokenizer, growth_rows["alpha"], PROTOCOLS, protocol, device, overlay
    )
    beta_choice_spawned = _choice(
        model, tokenizer, growth_rows["beta_eval"], PROTOCOLS, protocol, device, overlay
    )
    spawn_max_nll_delta = max(
        abs(alpha_spawned["mean_reference_nll"] - alpha_base["mean_reference_nll"]),
        abs(beta_spawned["mean_reference_nll"] - beta_base["mean_reference_nll"]),
    )
    spawn_max_choice_margin_delta = max(
        abs(
            float(alpha_choice_spawned["mean_choice_margin"])
            - float(alpha_choice_base["mean_choice_margin"])
        ),
        abs(
            float(beta_choice_spawned["mean_choice_margin"])
            - float(beta_choice_base["mean_choice_margin"])
        ),
    )
    spawn_choice_accuracy_unchanged = bool(
        alpha_choice_spawned["strict_choice_accuracy"]
        == alpha_choice_base["strict_choice_accuracy"]
        and beta_choice_spawned["strict_choice_accuracy"]
        == beta_choice_base["strict_choice_accuracy"]
    )

    core._train_cell(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        rows=growth_rows["beta_train"],
        cell_index=child,
        protocol=protocol,
        device=device,
        steps=int(growth_cfg["steps"]),
        learning_rate=float(local_cfg["learning_rate"]),
        max_gradient_norm=float(local_cfg["max_gradient_norm"]),
        learn_key=True,
        route_target_weight=float(local_cfg["route_target_weight"]),
    )
    child_alpha = core._evaluate(
        model, tokenizer, growth_rows["alpha"], protocol, device, overlay
    )
    child_beta = core._evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    child_alpha_choice = _choice(
        model, tokenizer, growth_rows["alpha"], PROTOCOLS, protocol, device, overlay
    )
    child_beta_choice = _choice(
        model, tokenizer, growth_rows["beta_eval"], PROTOCOLS, protocol, device, overlay
    )
    beta_routes = core._route_primary_for_rows(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    growth_result = {
        "entity": growth_entity,
        "parent_cell": growth_parent,
        "child_cell": child,
        "spawn_max_mean_nll_delta": spawn_max_nll_delta,
        "spawn_max_choice_margin_delta": spawn_max_choice_margin_delta,
        "spawn_choice_accuracy_unchanged": spawn_choice_accuracy_unchanged,
        "parent_only_control": parent_control,
        "beta_nll_gain": core._nll_gain(beta_base, child_beta),
        "beta_choice": child_beta_choice,
        "alpha_nll_regression": float(
            child_alpha["mean_reference_nll"] - alpha_base["mean_reference_nll"]
        ),
        "alpha_choice": child_alpha_choice,
        "child_beta_route_fraction": sum(value == child for value in beta_routes)
        / max(len(beta_routes), 1),
        "alpha_before": core._metric_dict(alpha_base),
        "alpha_after": core._metric_dict(child_alpha),
        "beta_before": core._metric_dict(beta_base),
        "beta_after": core._metric_dict(child_beta),
    }
    overlay.restore_(formation_snapshot)

    branch_result = _branch_phase_v12(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        formation_snapshot=formation_snapshot,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
    )

    gates = {
        "compatibility": compatibility_excess
        <= float(gates_cfg["maximum_compatibility_excess_over_repeatability"]),
        "direct_acquisition": formation_gains["direct"]
        >= float(gates_cfg["minimum_direct_nll_gain"]),
        "direct_semantic_choice": float(formation_choice["direct"]["strict_choice_accuracy"])
        >= float(gates_cfg["minimum_direct_choice_accuracy"]),
        "negation_acquisition": formation_gains["negation"]
        >= float(gates_cfg["minimum_negation_nll_gain"]),
        "negation_semantic_choice": float(
            formation_choice["negation"]["strict_choice_accuracy"]
        )
        >= float(gates_cfg["minimum_negation_choice_accuracy"]),
        "relation_acquisition": formation_gains["relation"]
        >= float(gates_cfg["minimum_relation_nll_gain"]),
        "relation_semantic_choice": float(
            formation_choice["relation"]["strict_choice_accuracy"]
        )
        >= float(gates_cfg["minimum_relation_choice_accuracy"]),
        "history_preservation": formation_history_kl <= float(gates_cfg["maximum_history_kl"]),
        "semantic_routing": float(route_metrics["mean_route_agreement"])
        >= float(gates_cfg["minimum_mean_route_agreement"]),
        "route_diversity": int(route_metrics["distinct_primary_cells"])
        >= int(gates_cfg["minimum_distinct_primary_cells"])
        and float(route_metrics["maximum_primary_cell_fraction"])
        <= float(gates_cfg["maximum_primary_cell_fraction"]),
        "semantic_local_write": float(local_result["semantic_write_nll_gain"])
        >= float(gates_cfg["minimum_semantic_write_nll_gain"]),
        "semantic_local_write_choice": float(
            local_choice_after["strict_choice_accuracy"]
        )
        >= float(gates_cfg["minimum_semantic_write_choice_accuracy"]),
        "unrelated_locality": float(local_result["unrelated_nll_regression"])
        <= float(gates_cfg["maximum_unrelated_nll_regression"]),
        "unrelated_choice_nonregression": float(
            unrelated_choice_after["strict_choice_accuracy"]
        )
        >= float(unrelated_choice_before["strict_choice_accuracy"]),
        "growth_spawn_invariance": spawn_max_nll_delta
        <= float(gates_cfg["maximum_spawn_mean_nll_delta"])
        and spawn_max_choice_margin_delta
        <= float(gates_cfg["maximum_spawn_choice_margin_delta"])
        and spawn_choice_accuracy_unchanged,
        "growth_necessity": not bool(
            parent_control["satisfies_registered_conflict_solution"]
        ),
        "growth_beta_acquisition": float(growth_result["beta_nll_gain"])
        >= float(gates_cfg["minimum_growth_beta_nll_gain"]),
        "growth_beta_semantic_choice": float(
            child_beta_choice["strict_choice_accuracy"]
        )
        >= float(gates_cfg["minimum_growth_beta_choice_accuracy"]),
        "growth_alpha_retention": float(growth_result["alpha_nll_regression"])
        <= float(gates_cfg["maximum_growth_alpha_nll_regression"]),
        "growth_alpha_semantic_choice": float(
            child_alpha_choice["strict_choice_accuracy"]
        )
        >= float(gates_cfg["minimum_growth_alpha_choice_accuracy"]),
        "growth_child_routing": float(growth_result["child_beta_route_fraction"])
        >= float(gates_cfg["minimum_child_beta_route_fraction"]),
        "branch_standalone_acquisition": float(
            branch_result["minimum_standalone_nll_gain"]
        )
        >= float(gates_cfg["minimum_branch_standalone_nll_gain"]),
        "branch_standalone_semantic_choice": float(
            branch_result["minimum_standalone_choice_accuracy"]
        )
        >= float(gates_cfg["minimum_branch_standalone_choice_accuracy"]),
        "branch_merge": float(branch_result["minimum_merge_retention"])
        >= float(gates_cfg["minimum_branch_merge_retention"]),
        "branch_merge_semantic_choice": float(
            branch_result["minimum_merged_choice_accuracy"]
        )
        >= float(gates_cfg["minimum_branch_merged_choice_accuracy"]),
        "rollback": bool(local_result["rollback_exact"])
        and bool(branch_result["exact_rollback"]),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    protocol_sha256 = core._sha256(core.PROTOCOL_PATH)
    result = {
        "experiment": protocol["experiment"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "dataset_generator_git_blob_sha": protocol["dataset"]["generator_git_blob_sha"],
        "seed": seed,
        "status": status,
        "model_id": model_id,
        "revision": revision,
        "compatibility": compatibility,
        "base_validation": core._metric_dict(base_validation),
        "selected_validation": core._metric_dict(selected_validation),
        "base_metrics": {name: core._metric_dict(value) for name, value in base_metrics.items()},
        "base_candidate_choice": base_choice,
        "formation": {
            "training": formation_log,
            "metrics": {name: core._metric_dict(value) for name, value in formation_metrics.items()},
            "nll_gains": formation_gains,
            "candidate_choice": formation_choice,
            "history_kl": formation_history_kl,
            "routing": route_metrics,
        },
        "semantic_local_write": local_result,
        "growth": growth_result,
        "branch_merge": branch_result,
        "gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
    }
    seed_root = core.RESULTS_ROOT / f"seed-{seed}"
    core._write_json(seed_root / "result.json", result)
    core._write_json(
        seed_root / "seed_summary.json",
        {
            "experiment": protocol["experiment"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol_sha256,
            "dataset_generator_git_blob_sha": protocol["dataset"]["generator_git_blob_sha"],
            "seed": seed,
            "status": status,
            "failed_gates": result["failed_gates"],
            "formation_nll_gains": formation_gains,
            "formation_choice_accuracy": {
                name: value["strict_choice_accuracy"]
                for name, value in formation_choice.items()
            },
            "mean_route_agreement": route_metrics["mean_route_agreement"],
            "distinct_primary_cells": route_metrics["distinct_primary_cells"],
            "semantic_write_nll_gain": local_result["semantic_write_nll_gain"],
            "semantic_write_choice_accuracy": local_choice_after["strict_choice_accuracy"],
            "growth_beta_nll_gain": growth_result["beta_nll_gain"],
            "growth_beta_choice_accuracy": child_beta_choice["strict_choice_accuracy"],
            "growth_parent_only_solved": parent_control[
                "satisfies_registered_conflict_solution"
            ],
            "minimum_branch_standalone_choice_accuracy": branch_result[
                "minimum_standalone_choice_accuracy"
            ],
            "minimum_branch_merge_retention": branch_result["minimum_merge_retention"],
            "minimum_branch_merged_choice_accuracy": branch_result[
                "minimum_merged_choice_accuracy"
            ],
        },
    )
    core._progress(seed, f"formal v1.2 seed complete: {status} failed={result['failed_gates']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CLM Conversion Kill Test 001 formal seed under protocol v1.2"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.seed, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
