"""Strict Experiment 005 seed/matrix decisions and paired-window intervals."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import ARMS


def paired_bootstrap_delta(candidate_nll: list[float], baseline_nll: list[float], *,
                           seed: int, replicates: int = 5000, group_ids: list[int] | None = None) -> dict[str, Any]:
    if len(candidate_nll) != len(baseline_nll) or len(candidate_nll) < 2:
        raise ValueError("paired bootstrap requires equal-length window losses (n >= 2)")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    candidate, baseline = np.asarray(candidate_nll), np.asarray(baseline_nll)
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("paired NLL samples must be finite")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    if group_ids is None or len(group_ids) != len(delta):
        raise ValueError("paired NLL bootstrap requires an aligned story-cluster identity")
    groups = np.asarray(group_ids)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("paired bootstrap requires at least two independent stories")
    sums = np.asarray([delta[groups == group].sum() for group in unique])
    sizes = np.asarray([(groups == group).sum() for group in unique])
    indices = rng.integers(0, len(unique), size=(replicates, len(unique)))
    boot = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    mean_delta = float(delta.mean())
    return {"candidate_minus_baseline_mean_nll": mean_delta,
            "paired_bootstrap_95pct_ci": [float(np.quantile(boot, 0.025)),
                                           float(np.quantile(boot, 0.975))],
            "ppl_ratio_candidate_over_baseline": math.exp(mean_delta),
            "paired_windows": len(delta), "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "sampling_unit": "paired whole-story clusters of disjoint within-story windows",
            "story_clusters": len(unique)}


def paired_bootstrap_p95_ratio(candidate_ms: list[float], baseline_ms: list[float], *,
                               seed: int, replicates: int = 5000) -> dict[str, Any]:
    if len(candidate_ms) != len(baseline_ms) or len(candidate_ms) < 2:
        raise ValueError("paired p95 bootstrap requires equal-length timing samples (n >= 2)")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    candidate, baseline = np.asarray(candidate_ms), np.asarray(baseline_ms)
    if (not np.isfinite(candidate).all() or not np.isfinite(baseline).all() or
            np.any(candidate <= 0) or np.any(baseline <= 0)):
        raise ValueError("decode timing samples must be positive and finite")
    if candidate.ndim != 2 or candidate.shape != baseline.shape:
        raise ValueError("p95 inputs must be matching repetition-by-token matrices")
    point = float(np.quantile(candidate, 0.95) / np.quantile(baseline, 0.95))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(replicates, len(candidate)))
    candidate_p95 = np.quantile(candidate[indices], 0.95, axis=(1, 2))
    baseline_p95 = np.quantile(baseline[indices], 0.95, axis=(1, 2))
    ratios = candidate_p95 / baseline_p95
    return {"candidate_over_baseline_p95_ratio": point,
            "paired_bootstrap_95pct_ci": [float(np.quantile(ratios, 0.025)),
                                           float(np.quantile(ratios, 0.975))],
            "upper_ci": float(np.quantile(ratios, 0.975)),
            "paired_samples": len(candidate), "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "sampling_unit": "whole independent prefill/cache-reset repetitions"}


def _incomplete(reason: str, *, seed: int | None = None) -> dict[str, Any]:
    return {"status": "INCOMPLETE", "scientific_decision": False,
            "seed": seed, "reason": reason}


def seed_decision(metrics_by_arm: dict[str, dict[str, Any]], *, protocol: dict[str, Any],
                  seed: int) -> dict[str, Any]:
    missing = sorted(set(ARMS) - set(metrics_by_arm))
    if missing:
        return _incomplete("missing trained arms: " + ", ".join(missing), seed=seed)
    rows = [metrics_by_arm[arm] for arm in ARMS]
    if any(row.get("status") != "completed" for row in rows):
        return _incomplete("at least one arm is not complete", seed=seed)
    identity = rows[0].get("identity", {})
    shared_identity = ("source_commit", "config_sha256", "protocol_sha256",
                       "assets_manifest_sha256")
    for arm, row in zip(ARMS, rows):
        if row.get("seed_role") not in {"development", "confirmation"} or row.get("seed_role") != rows[0].get("seed_role"):
            return _incomplete("missing or mixed development/confirmation seed role", seed=seed)
        if row.get("arm") != arm or row.get("seed") != seed:
            return _incomplete("arm or seed identity mismatch", seed=seed)
        if row.get("source_commit") != identity.get("source_commit"):
            return _incomplete("source commit differs across paired arms", seed=seed)
        if any(row.get("identity", {}).get(key) != identity.get(key) for key in shared_identity):
            return _incomplete("config/protocol/assets identity differs across paired arms", seed=seed)
        if row.get("train_tokens") != protocol["common"]["train_tokens_per_arm"]:
            return _incomplete("arm did not finish its registered token budget", seed=seed)
        if row.get("steps") != protocol["common"]["train_steps"]:
            return _incomplete("arm did not finish the registered optimizer-step budget", seed=seed)
        evaluation = row.get("final_holdout", {})
        if evaluation.get("validation_tokens") != protocol["common"]["heldout_validation_tokens"]:
            return _incomplete("holdout token count is incomplete or mismatched", seed=seed)
        if not isinstance(evaluation.get("decode_parity"), dict) or \
                evaluation["decode_parity"].get("passed") is not True:
            return _incomplete("KV-cache decode parity did not pass", seed=seed)
    metrics = {arm: metrics_by_arm[arm]["final_holdout"] for arm in ARMS}
    params = [metrics[arm]["model_total_params"] for arm in ARMS]
    tolerance = float(protocol["gates"]["parameter_tolerance_fraction"])
    if max(params) / min(params) - 1 > tolerance:
        return {"status": "NOT_SUPPORTED", "scientific_decision": False, "seed": seed,
                "reason": "total parameter capacity differs beyond registered tolerance"}

    groups = metrics["M0"].get("story_group_ids")
    if groups is None or any(metrics[arm].get("story_group_ids") != groups for arm in ARMS):
        return _incomplete("paired story-cluster identities missing or mismatched", seed=seed)
    b1 = paired_bootstrap_delta(metrics["S0"]["sample_nll"], metrics["P0"]["sample_nll"],
                                seed=int(protocol["bootstrap_seed"]) + 100, replicates=int(protocol["bootstrap_replicates"]), group_ids=groups)
    b2_random = paired_bootstrap_delta(metrics["G0"]["sample_nll"], metrics["R0"]["sample_nll"],
                                       seed=int(protocol["bootstrap_seed"]) + 200,
                                       replicates=int(protocol["bootstrap_replicates"]), group_ids=groups)
    threshold = float(protocol["gates"]["b2_vs_r0_ppl_relative_improvement_min"])
    b1_pass = b1["paired_bootstrap_95pct_ci"][1] < 0 and b1["ppl_ratio_candidate_over_baseline"] <= 1.0 - float(
        protocol["gates"]["b1_ppl_relative_improvement_min"]
    )
    b2_pass = b2_random["paired_bootstrap_95pct_ci"][1] < 0 and b2_random["ppl_ratio_candidate_over_baseline"] <= 1.0 - threshold

    s0_rule = protocol.get("b2_strong_flat_matching_rule")
    system_rules = protocol.get("system_runtime_rule")
    s0_comparison = {
        "ppl_ratio_g0_over_s0": metrics["G0"]["perplexity"] / metrics["S0"]["perplexity"],
        "unique_weight_ratio_g0_over_s0":
            metrics["G0"]["unique_weight_params_per_token"]["mean"] /
            metrics["S0"]["unique_weight_params_per_token"]["mean"],
        "gemm_flops_ratio_g0_over_s0":
            metrics["G0"]["observed_gemm"]["observed_gemm_flops_per_token"] /
            metrics["S0"]["observed_gemm"]["observed_gemm_flops_per_token"],
        "decode_p50_ratio_g0_over_s0":
            metrics["G0"]["decode_profile"]["warm_decode_host_p50_ms"] /
            metrics["S0"]["decode_profile"]["warm_decode_host_p50_ms"],
    }
    s0_max_ratios = s0_rule.get("max_ratios") if isinstance(s0_rule, dict) else None
    s0_valid = (isinstance(s0_max_ratios, dict) and "ppl_ratio_g0_over_s0" in s0_max_ratios and
                any(key in s0_max_ratios for key in (
                    "unique_weight_ratio_g0_over_s0", "gemm_flops_ratio_g0_over_s0",
                    "decode_p50_ratio_g0_over_s0")))
    b2_s0_pass = bool(s0_valid and all(
        key in s0_comparison and s0_comparison[key] <= float(limit)
        for key, limit in s0_max_ratios.items()
    )) if s0_valid else False
    graph_rule = protocol.get("b2_graph_diagnostics_rule")
    graph_metrics = metrics_by_arm["G0"].get("graph_learning") or {}
    posthoc = metrics_by_arm["G0"].get("posthoc_diagnostics") or {}
    graph_min_changes = graph_rule.get("min_edge_replacements") if isinstance(graph_rule, dict) else None
    graph_min_perturbations = graph_rule.get("min_perturbation_runs") if isinstance(graph_rule, dict) else None
    graph_max_repeated_fraction = graph_rule.get("max_repeated_path_token_fraction") if isinstance(graph_rule, dict) else None
    graph_min_coverage = graph_rule.get("min_source_coverage_fraction") if isinstance(graph_rule, dict) else None
    graph_min_quality_change = graph_rule.get("min_ppl_relative_degradation") if isinstance(graph_rule, dict) else None
    final_layers = graph_metrics.get("final_routing_summary", {}).get("per_layer", [])
    repeated_fraction = max((float(row.get("repeated_path_token_fraction", 1.0))
                             for row in final_layers), default=1.0)
    graph_diagnostics_frozen = all(value is not None for value in (
        graph_min_changes, graph_min_perturbations, graph_max_repeated_fraction,
        graph_min_quality_change, graph_min_coverage
    )) and isinstance(graph_rule, dict) and \
        graph_rule.get("select_interventions_from") == "fixed_training_probe_edge_use"
    perturbed_ppl_changes = [(float(row["perplexity"]) /
                                 float(metrics["G0"]["perplexity"]) - 1.0)
                             for row in posthoc.get("results", [])]
    perturbation_intervals = []
    paired_perturbation_complete = all(isinstance(row.get("sample_nll"), list) and
        len(row["sample_nll"]) == len(metrics["G0"]["sample_nll"])
        for row in posthoc.get("results", [])) and bool(posthoc.get("results"))
    if paired_perturbation_complete:
        perturbation_intervals = [paired_bootstrap_delta(
            row["sample_nll"], metrics["G0"]["sample_nll"],
            seed=int(protocol["bootstrap_seed"]) + 400 + index,
            replicates=int(protocol["bootstrap_replicates"]), group_ids=groups)
            for index, row in enumerate(posthoc["results"])]
    source_coverage_complete = bool(graph_diagnostics_frozen and
        min(graph_metrics.get("effective_source_coverage_fraction_per_layer") or [0.0]) >= float(graph_min_coverage))
    graph_diagnostics_pass = bool(
        graph_diagnostics_frozen and
        int(graph_metrics.get("edge_replacements", 0)) >= int(graph_min_changes) and
        source_coverage_complete and paired_perturbation_complete and
        all(row["paired_bootstrap_95pct_ci"][0] > 0 for row in perturbation_intervals) and
        len(posthoc.get("results", [])) >= int(graph_min_perturbations) and
        repeated_fraction <= float(graph_max_repeated_fraction) and
        len(perturbed_ppl_changes) >= int(graph_min_perturbations) and
        all(value >= float(graph_min_quality_change) for value in perturbed_ppl_changes) and
        all(row.get("parameter_count_unchanged") is True
            for row in posthoc.get("results", []))
    )
    system_values = {
        "ppl_ratio": metrics["G0"]["perplexity"] / metrics["M0"]["perplexity"],
        "unique_weight_ratio": metrics["G0"]["unique_weight_params_per_token"]["mean"] /
                               metrics["M0"]["unique_weight_params_per_token"]["mean"],
        "gemm_flops_ratio": metrics["G0"]["observed_gemm"]["observed_gemm_flops_per_token"] /
                            metrics["M0"]["observed_gemm"]["observed_gemm_flops_per_token"],
        "decode_p50_ratio": metrics["G0"]["decode_profile"]["warm_decode_host_p50_ms"] /
                            metrics["M0"]["decode_profile"]["warm_decode_host_p50_ms"],
        "decode_p95_ratio": metrics["G0"]["decode_profile"]["warm_decode_host_p95_ms"] /
                            metrics["M0"]["decode_profile"]["warm_decode_host_p95_ms"],
    }
    system_gates = protocol["gates"]
    p95_limit = (system_rules.get("decode_p95_ratio_upper_ci_max")
                 if isinstance(system_rules, dict) else None)
    p95_ci = None
    if p95_limit is not None:
        min_runs = int(system_rules["min_decode_repetitions"])
        min_samples = int(system_rules["min_warm_samples"])
        for arm in ("G0", "M0"):
            profile = metrics[arm]["decode_profile"]
            if (len(profile.get("warm_decode_host_ms_by_repetition", [])) < min_runs or
                    len(profile.get("warm_decode_host_ms", [])) < min_samples):
                return _incomplete("insufficient independent decode repetitions/samples", seed=seed)
        p95_ci = paired_bootstrap_p95_ratio(
            metrics["G0"]["decode_profile"]["warm_decode_host_ms_by_repetition"],
            metrics["M0"]["decode_profile"]["warm_decode_host_ms_by_repetition"],
            seed=int(protocol["bootstrap_seed"]) + 300, replicates=int(protocol["bootstrap_replicates"])
        )
    system_pass = (
        system_values["ppl_ratio"] <= system_gates["system_ppl_ratio_max"] and
        system_values["unique_weight_ratio"] <= system_gates["system_unique_weight_ratio_max"] and
        system_values["gemm_flops_ratio"] <= system_gates["system_gemm_flops_ratio_max"] and
        system_values["decode_p50_ratio"] <= system_gates["system_decode_p50_ratio_max"] and
        p95_limit is not None and p95_ci["upper_ci"] <= float(p95_limit)
    )
    status_b2 = ("INCOMPLETE" if not (s0_valid and graph_diagnostics_frozen and source_coverage_complete and paired_perturbation_complete) else
                 "PASS" if b2_pass and b2_s0_pass and graph_diagnostics_pass else
                 "NOT_SUPPORTED")
    status_system = ("INCOMPLETE" if p95_limit is None else
                     "PASS" if system_pass else "NOT_SUPPORTED")
    return {"status": "SEED_ANALYZED", "scientific_decision": False,
            "seed": seed, "seed_role": rows[0].get("seed_role"),
            "b1_sequential_composition": {"status": "PASS" if b1_pass else "NOT_SUPPORTED",
                                          "comparison": b1},
            "b2_learned_connections": {"status": status_b2,
                                        "vs_random_fixed": b2_random,
                                        "vs_strong_flat": {"status":
                                            "PASS" if b2_s0_pass else
                                            "NOT_SUPPORTED" if s0_valid else "INCOMPLETE",
                                            "ratios": s0_comparison, "rule": s0_rule},
                                        "graph_diagnostics": {"status":
                                            "PASS" if graph_diagnostics_pass else
                                            "NOT_SUPPORTED" if graph_diagnostics_frozen and source_coverage_complete and paired_perturbation_complete else "INCOMPLETE",
                                            "effective_source_coverage_fraction_per_layer": graph_metrics.get("effective_source_coverage_fraction_per_layer"),
                                            "source_coverage_complete": source_coverage_complete,
                                            "edge_replacements": graph_metrics.get("edge_replacements"),
                                            "perturbation_runs": len(posthoc.get("results", [])),
                                            "max_repeated_path_token_fraction": repeated_fraction,
                                            "perturbed_ppl_relative_degradations": perturbed_ppl_changes,
                                            "perturbation_paired_story_bootstrap": perturbation_intervals,
                                            "absolute_sensitivity_only": [abs(value) for value in perturbed_ppl_changes],
                                            "rule": graph_rule}},
            "system_gate": {"status": status_system, "ratios": system_values,
                            "decode_p95_paired_bootstrap": p95_ci, "rule": system_rules},
            "limitations": ["development-seed analysis is not confirmation",
                            "from-scratch 61M results do not establish pretrained 7B-30B or mobile deployment"]}


def matrix_decision(seed_reports: list[dict[str, Any]], protocol: dict[str, Any],
                    confirmation_seeds: list[int]) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in seed_reports if "seed" in row}
    missing = [seed for seed in confirmation_seeds if seed not in by_seed]
    if missing or not confirmation_seeds:
        return {"status": "CONFIRMATION_INCOMPLETE", "scientific_decision": False,
                "missing_seeds": missing or list(confirmation_seeds), "seed_reports": seed_reports}
    expected = int(protocol["gates"]["confirmation_seeds_required"])
    if len(confirmation_seeds) != expected:
        return {"status": "CONFIRMATION_INCOMPLETE", "scientific_decision": False,
                "reason": f"expected {expected} registered confirmation seeds"}
    selected = [by_seed[seed] for seed in confirmation_seeds]
    if any(row.get("seed_role") != "confirmation" for row in selected):
        return {"status": "CONFIRMATION_INCOMPLETE", "scientific_decision": False,
                "reason": "development evidence cannot substitute for confirmation", "seed_reports": selected}
    gates = {}
    for name in ("b1_sequential_composition", "b2_learned_connections", "system_gate"):
        statuses = [row.get(name, {}).get("status", "INCOMPLETE") for row in selected]
        gates[name] = {"status": "SUPPORTED" if all(s == "PASS" for s in statuses)
                       else "NOT_SUPPORTED" if "NOT_SUPPORTED" in statuses
                       else "INCOMPLETE", "per_seed": statuses}
    completed = all(report.get("status") == "SEED_ANALYZED" and
                    report.get("seed_role") == "confirmation" for report in selected)
    return {"status": "CONFIRMATION_ANALYZED" if completed else "CONFIRMATION_INCOMPLETE",
            "scientific_decision": False,
            "review_status": "AWAITING_RESEARCH_REVIEW" if completed else "EVIDENCE_INCOMPLETE",
            "b1": gates["b1_sequential_composition"],
            "b2": gates["b2_learned_connections"],
            "system": gates["system_gate"],
            "seed_reports": selected}
