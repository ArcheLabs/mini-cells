"""Fixed exploratory criteria; training Seeds remain the replication unit."""

import numpy as np
from minicells_r3c_moe_bridge.statistics import story_interval
from .evaluation import mean_nll


def paired_story_interval(x, y, seed, replicates):
    if set(x) != set(y):
        raise ValueError("Story pairing differs")
    keys = sorted(x)
    if any(x[k]["tokens"] != y[k]["tokens"] for k in keys):
        raise ValueError("paired Story token counts differ")
    delta = np.asarray([x[k]["mean_nll"] - y[k]["mean_nll"] for k in keys], float)
    weight = np.asarray([x[k]["tokens"] for k in keys], float)
    if not len(keys) or not np.isfinite(delta).all() or (weight <= 0).any():
        raise ValueError("finite paired Story values required")
    if np.all(weight == weight[0]):
        row = story_interval(delta, seed, replicates)
    else:
        mean = float(np.average(delta, weights=weight))
        rng = np.random.default_rng(seed)
        indices = rng.integers(len(keys), size=(replicates, len(keys)))
        sampled = (delta[indices] * weight[indices]).sum(1) / weight[indices].sum(1)
        row = {
            "mean": mean,
            "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "stories": len(keys),
            "exploratory_bootstrap_p": float(
                (1 + np.sum(np.abs(sampled - mean) >= abs(mean))) / (replicates + 1)
            ),
        }
    row["estimand"] = "TOKEN_WEIGHTED_NLL_DIFFERENCE_WITH_STORY_CLUSTER_RESAMPLING"
    return row


def conclusions(c, results):
    comparisons = c["statistics"]["comparisons"]
    per_seed = []
    quality = {}
    exposure = {}
    cost = {}
    complete = all(all(a in results.get(str(s), {}) for a in c["arms"]) for s in c["seeds"])
    for seed in c["seeds"]:
        per = results.get(str(seed), {})
        key = str(seed)
        quality[key] = {a: mean_nll(r["quality"]["development"]) for a, r in per.items()}
        exposure[key] = {a: r["exposure"] for a, r in per.items()}
        cost[key] = {
            a: {"training": r["training_cost"], "inference": r["inference_cost"]}
            for a, r in per.items()
        }
        if len(per) != len(c["arms"]):
            continue
        paired = {}
        for comp in comparisons:
            a, b = comp.split("-")
            x = per[a]["quality"]["development"]
            y = per[b]["quality"]["development"]
            row = paired_story_interval(x, y, seed, c["statistics"]["bootstrap_replicates"])
            row["token_weighted_nll_delta"] = quality[key][a] - quality[key][b]
            paired[comp] = row
        order = sorted(paired, key=lambda k: paired[k].get("exploratory_bootstrap_p", 1))
        previous = 0.0
        for i, k in enumerate(order):
            previous = max(
                previous, min(1.0, (len(order) - i) * paired[k].get("exploratory_bootstrap_p", 1))
            )
            paired[k]["holm_exploratory_p"] = previous
        per_seed.append({"seed": seed, "comparisons": paired})
    states = {h: "INCONCLUSIVE" for h in ("H1", "H2", "H3", "H4")}
    deltas = {}
    if complete:
        for k in comparisons:
            a, b = k.split("-")
            values = [quality[str(s)][a] - quality[str(s)][b] for s in c["seeds"]]
            deltas[k] = {
                "paired_seed_deltas": values,
                "mean": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }

        def supported(k):
            r = deltas[k]
            return (
                all(v < 0 for v in r["paired_seed_deltas"])
                and r["mean"] <= -c["statistics"]["minimum_practical_nll_gain"]
            )

        for h, k in [("H1", "L-F0"), ("H2", "L-M"), ("H3", "J-F")]:
            states[h] = "SUPPORTED_EXPLORATORY" if supported(k) else "NOT_SUPPORTED"
        stable = all(
            not exposure[str(s)]["J"]["persistent_zero_exposure"]
            and exposure[str(s)]["J"]["zero_gradient_step_fraction"]
            <= c["statistics"]["max_zero_gradient_step_fraction"]
            and exposure[str(s)]["J"]["zero_gradient_step_fraction"]
            <= exposure[str(s)]["F"]["zero_gradient_step_fraction"]
            + c["statistics"]["max_zero_gradient_fraction_increase"]
            for s in c["seeds"]
        )
        states["H4"] = (
            "SUPPORTED_EXPLORATORY"
            if stable and states["H3"] == "SUPPORTED_EXPLORATORY"
            else "NOT_SUPPORTED"
        )
    ranking = {
        str(s): {
            a: {
                "scored_pairs": sum(r["preference"]["scored_pairs"] for r in row["curve"]),
                "mean_candidate_gap": _average(row["curve"], "mean_candidate_gap"),
                "mean_ranking_accuracy": _average(row["curve"], "ranking_accuracy"),
                "scope": "Train pre-update sampled candidates; not independent ranking generalization",
            }
            for a, row in results.get(str(s), {}).items()
        }
        for s in c["seeds"]
    }
    return {
        "status": "EXPLORATORY_DEVELOPMENT_COMPLETE" if complete else "PARTIAL",
        "hypotheses": states,
        "nll": quality,
        "mean_nll_across_seeds": None
        if not complete
        else {a: float(np.mean([quality[str(seed)][a] for seed in c["seeds"]])) for a in c["arms"]},
        "seed_nll_sd": None
        if not complete
        else {
            a: float(np.std([quality[str(seed)][a] for seed in c["seeds"]], ddof=1))
            if len(c["seeds"]) > 1
            else None
            for a in c["arms"]
        },
        "paired_seed_differences": deltas,
        "story_bootstrap_within_seed": per_seed,
        "independent_training_seeds": len(per_seed),
        "exposure": exposure,
        "cost": cost,
        "candidate_learning": ranking,
        "confirmation": "SEALED",
        "scientific_scope": "EXPLORATORY_ONLY; no independent confirmation; bootstrap Story counts are not training repeats",
        "continued_connection_gain": "see J-P separately; H3 alone does not establish continuing-policy value",
        "noninferiority_J_P": None
        if not complete
        else all(
            v <= c["statistics"]["noninferiority_margin"]
            for v in deltas["J-P"]["paired_seed_deltas"]
        ),
    }


def _average(curve, key):
    v = [r["preference"].get(key) for r in curve if r["preference"].get(key) is not None]
    return None if not v else float(np.mean(v))


def markdown(report):
    lines = [
        "# R3d — Exploratory Findings",
        "",
        "Status: " + report["status"],
        "",
        "Confirmation: SEALED",
        "",
        "H1–H4: " + str(report["hypotheses"]),
        "",
        "| Seed | F0 | L | M | F | P | J |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed, rows in report["nll"].items():
        lines.append(
            "| "
            + seed
            + " | "
            + " | ".join(
                "—" if a not in rows else "%.6f" % rows[a] for a in ("F0", "L", "M", "F", "P", "J")
            )
            + " |"
        )
    lines += ["", "Mean NLL across training Seeds: " + str(report["mean_nll_across_seeds"]), ""]
    lines += [
        "",
        "Paired comparisons and clustered intervals: METRICS.json.",
        "",
        "Exposure/gradient coverage and route changes: ROUTING_SUMMARY.json. Training/inference work and measured timing: TRAINING_COST.json.",
        "",
        "Candidate ranking/gaps are in-training sampled-action diagnostics, not exact Q values or evidence of generalization. Compare L–F0 for closed-loop benefit, L–M for horizon value, P–F for retained policy benefit, J–F and J–P for joint/continuing benefit. These comparisons do not by themselves identify causal drift of Cell abilities.",
        "",
        "A non-negative comparison does not trigger more training. Incomplete runs remain INCONCLUSIVE; invalid integrity/budget states are INVALID_RUN. All conclusions remain Development-level.",
    ]
    return "\n".join(lines) + "\n"
