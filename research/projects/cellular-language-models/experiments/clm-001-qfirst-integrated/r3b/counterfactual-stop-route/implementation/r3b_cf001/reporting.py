"""Aggregate-only reports; conditional outcomes never substitute for paired ITT."""

from collections import defaultdict
import json
import numpy as np
from .statistics import paired, cluster_interval, holm


def policy_summary(rows):
    if not rows:
        return {"status": "NOT_MEASURED"}
    keys = (
        "nll",
        "cost",
        "lengths",
        "actual_stop",
        "forced_end",
        "ctx_calls",
        "self_loops",
        "two_step_loops",
        "cell0_ctx_cell0",
        "touched",
    )
    out = {k: float(np.mean([np.mean(r[k]) for r in rows.values()])) for k in keys}
    out["ctx_ge2_rate"] = float(
        np.mean([np.mean(np.asarray(r["ctx_calls"]) >= 2) for r in rows.values()])
    )
    out["stories"] = len(rows)
    out["token_paths"] = sum(len(r["nll"]) for r in rows.values())
    transitions = {}
    for r in rows.values():
        for key, n in r["transition_matrix"].items():
            transitions[key] = transitions.get(key, 0) + n
    out["transition_matrix"] = transitions
    calls = sum(sum(r["lengths"]) for r in rows.values())
    edges = sum(r["transition_count"] for r in rows.values())
    out["context_call_fraction"] = (
        sum(sum(r["ctx_calls"]) for r in rows.values()) / calls if calls else None
    )
    out["self_loop_transition_fraction"] = (
        sum(sum(r["self_loops"]) for r in rows.values()) / edges if edges else None
    )
    return out


def interventions(base, policies, c):
    comparisons = {}
    for name, rows in policies.items():
        if set(rows) != set(base):
            continue
        comparisons[name] = paired(
            base, rows, c["lambda_cost"], c["bootstrap_seed"], c["bootstrap_replicates"]
        )
        comparisons[name]["policy_summary"] = policy_summary(rows)
        a = policy_summary(rows)["transition_matrix"]
        b = policy_summary(base)["transition_matrix"]
        comparisons[name]["transition_matrix_difference"] = {
            k: a.get(k, 0) - b.get(k, 0) for k in set(a) | set(b)
        }
        strata = defaultdict(list)
        for story in sorted(rows):
            a, r = base[story], rows[story]
            for depth in set(r["first_trigger"]):
                mask = np.asarray(r["first_trigger"]) == depth
                if depth < 0:
                    continue
                delta = np.asarray(r["nll"]) - np.asarray(a["nll"])
                strata[depth].append(float(delta[mask].mean()))
        comparisons[name]["first_trigger_depth_nll_descriptive"] = {
            str(d): cluster_interval(v, c["bootstrap_seed"], c["bootstrap_replicates"])
            for d, v in strata.items()
        }
        # Conditional strata are descriptive; each contributing Story gets equal weight.
        stratified = defaultdict(list)
        replacement_counts = defaultdict(int)
        blocked_counts = defaultdict(int)
        prefix_cells = defaultdict(int)
        for story in sorted(rows):
            r = rows[story]
            a = base[story]
            delta = np.asarray(r["nll"]) - np.asarray(a["nll"])
            trigger = np.asarray(r["touched"], dtype=bool)
            for field in ("blocked_action", "replacement", "forced_end"):
                for value in set(r[field]):
                    mask = (np.asarray(r[field]) == value) & trigger
                    if mask.any():
                        stratified[field + ":" + str(value)].append(float(delta[mask].mean()))
            for token in np.where(trigger)[0]:
                replacement_counts[str(r["replacement"][token])] += 1
                blocked_counts[str(r["blocked_action"][token])] += 1
                depth = r["first_trigger"][token]
                for action in r["actions"][token][:depth]:
                    if action >= 0:
                        prefix_cells[str(action)] += 1
        comparisons[name]["conditional_strata_delta_nll"] = {
            k: cluster_interval(v, c["bootstrap_seed"], c["bootstrap_replicates"])
            for k, v in sorted(stratified.items())
        }
        comparisons[name]["first_replacement_counts"] = dict(replacement_counts)
        comparisons[name]["first_blocked_action_counts"] = dict(blocked_counts)
        comparisons[name]["prefix_action_frequency_before_first_trigger"] = dict(prefix_cells)
    correction = {}
    for metric in ("delta_utility", "delta_nll", "delta_cost"):
        for family, names in (
            ("CF2", ["FORCE_STOP_D4", "FORCE_STOP_D8", "FORCE_STOP_D12"]),
            ("CF3", ["CTX_ONCE", "NO_ADJACENT_SAME_CELL", "CTX_ONCE_AND_NO_REPEAT"]),
        ):
            if all(
                n in comparisons and "exploratory_bootstrap_p" in comparisons[n][metric]
                for n in names
            ):
                correction[family + "_" + metric] = holm(
                    {n: comparisons[n][metric]["exploratory_bootstrap_p"] for n in names}
                )
    interaction = {"status": "NOT_MEASURED"}
    names = ("CTX_ONCE_AND_NO_REPEAT", "CTX_ONCE", "NO_ADJACENT_SAME_CELL")
    if all(n in comparisons for n in names):
        effects = []
        for key in sorted(base):
            u = {
                name: np.asarray(policies[name][key]["nll"])
                + c["lambda_cost"] * np.asarray(policies[name][key]["cost"])
                for name in names
            }
            baseline = np.asarray(base[key]["nll"]) + c["lambda_cost"] * np.asarray(
                base[key]["cost"]
            )
            effects.append(float((u[names[0]] - u[names[1]] - u[names[2]] + baseline).mean()))
        interaction = cluster_interval(effects, c["bootstrap_seed"], c["bootstrap_replicates"])
    return {
        "comparisons": comparisons,
        "holm": correction,
        "interaction_delta_utility": interaction,
        "independent_training_seeds": 1,
        "formal_confirmation": False,
    }


def census_summary(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["depth_zero_based"]].append(r)
    return {
        str(d): {
            "visited_states": len(rs),
            "nonterminal_stop_legal_states": sum(
                r["stop_legal"] and not r["actual_stop"] for r in rs
            ),
            "mean_pred_gap_stop_chosen": float(
                np.mean([r["q_stop"] - r["q_chosen"] for r in rs if r["stop_legal"]])
            )
            if any(r["stop_legal"] for r in rs)
            else None,
            "forced_end_path_fraction": float(np.mean([r["forced_end_path"] for r in rs])),
            "mean_context_calls_before_state": float(np.mean([r["n_ctx_so_far"] for r in rs])),
        }
        for d, rs in sorted(groups.items())
    }


def cf1_summary(rows, c, all_story_ids):
    if not rows:
        return {"status": "NOT_MEASURED"}

    def summarize(rs):
        weights = np.asarray([r["design_weight"] for r in rs])
        stories = sorted({r["story_id"] for r in rs})
        result = {
            "states": len(rs),
            "stories": len(stories),
            "estimator": "DESIGN_WEIGHTED_HAJEK_WITH_STORY_CLUSTER_BOOTSTRAP",
        }
        for kind in ("greedy_consistent", "frozen_sampled_oracle"):
            values = {}
            for metric in (
                "misranking_lower_bound",
                "true_stop_advantage",
                "signed_bias",
                "absolute_calibration_error",
                "direction_agreement_resolved",
                "unresolved",
            ):
                x = []
                keep = []
                for r in rs:
                    v = r[kind]
                    truth = v["true_stop_advantage"]
                    unknown = v.get("status") == "UNRESOLVED"
                    keep.append(not unknown if metric == "direction_agreement_resolved" else True)
                    x.append(
                        float(v["misranking"])
                        if metric == "misranking_lower_bound"
                        else truth
                        if metric == "true_stop_advantage"
                        else r["pred_stop_advantage"] - truth
                        if metric == "signed_bias"
                        else abs(r["pred_stop_advantage"] - truth)
                        if metric == "absolute_calibration_error"
                        else float((r["pred_stop_advantage"] > 0) == (truth > 0))
                        if metric == "direction_agreement_resolved"
                        else float(unknown)
                    )
                arrays = np.asarray(x)
                effective = weights * np.asarray(keep)
                if not effective.sum():
                    values[metric] = {"estimate": None, "ci95": None, "status": "UNRESOLVED"}
                    continue
                point = float(np.average(arrays, weights=effective))
                gen = np.random.default_rng(c["bootstrap_seed"])
                numerators = []
                denominators = []
                for story in stories:
                    mask = np.asarray([r["story_id"] == story for r in rs])
                    numerators.append(float((arrays[mask] * effective[mask]).sum()))
                    denominators.append(float(effective[mask].sum()))
                inds = gen.integers(len(stories), size=(c["bootstrap_replicates"], len(stories)))
                den = np.asarray(denominators)[inds].sum(-1)
                num = np.asarray(numerators)[inds].sum(-1)
                valid = den > 0
                means = num[valid] / den[valid]
                values[metric] = {
                    "estimate": point,
                    "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
                    "story_clusters": len(stories),
                    "bootstrap_nonempty_replicates": int(valid.sum()),
                }
            if kind == "frozen_sampled_oracle":
                identified = [r for r in rs if r[kind]["status"] != "UNRESOLVED"]
                values["identified_states"] = len(identified)
                values["misranking_denominator"] = (
                    "All audited STOP-legal continuing states; unresolved probability reported separately, not classified correct"
                )
            result[kind] = values
        return result

    depth = {
        str(d): summarize([r for r in rows if r["depth"] == d])
        for d in sorted({r["depth"] for r in rows})
    }
    termination = {
        str(v): summarize([r for r in rows if r["forced_end_path"] == v])
        for v in sorted({r["forced_end_path"] for r in rows})
    }
    visited = {r["story_id"] for r in rows}
    return {
        "status": "EXPLORATORY_SINGLE_SEED",
        "overall": summarize(rows),
        "by_depth": depth,
        "by_forced_end_path": termination,
        "by_previous_action": {
            str(v): summarize([r for r in rows if r["previous_action"] == v])
            for v in sorted({r["previous_action"] for r in rows})
        },
        "by_context_count": {
            str(v): summarize([r for r in rows if min(2, r["n_ctx_so_far"]) == v])
            for v in sorted({min(2, r["n_ctx_so_far"]) for r in rows})
        },
        "missing_sampled_story_count": len(set(all_story_ids) - visited),
        "interpretation": "Weighted sampled-state estimates; not independent token samples or a causal training diagnosis",
        "mc_samples": c["mc_samples"],
        "ctx_auxiliary_truth": "NOT_MEASURED_FIXED_BUDGET",
    }


def findings(reports):
    text = [
        "# CLM-001-R3b-CF-001",
        "",
        "Single-seed Development, inference-only. CONFIRMATION_SEALED; MULTISEED_NOT_TESTED; CAUSAL_FEEDBACK_NOT_TESTED.",
        "",
        "The native Greedy baseline remains the original policy. Counterfactual improvements do not replace its reported performance.",
        "",
    ]
    for name in (
        "old-analysis-preservation.json",
        "baseline-parity.json",
        "cf1-stop-value-calibration.json",
        "cf2-forced-stop.json",
        "cf3-repeat-constraint.json",
    ):
        text += [
            "## " + name,
            "",
            "```json",
            json.dumps(reports.get(name, {"status": "NOT_MEASURED"}), indent=2, allow_nan=False),
            "```",
            "",
        ]
    text += [
        "## Interpretation",
        "",
        "Negative ΔNLL and ΔUtility support harmful over-computation on this model/distribution. Positive ΔNLL with negative ΔUtility is a cost/quality tradeoff only. Shorter paths or lower forced-end rate alone are not language-quality improvements. Mixed or incomplete evidence is INCONCLUSIVE; no unique training-loss cause is established.",
    ]
    return "\n".join(text) + "\n"
