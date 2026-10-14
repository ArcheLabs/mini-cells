"""Story-paired, one-sided noninferiority; separate seed uncertainty."""

import math
import numpy as np


def paired_interval(before, after, groups, seed=1907, replicates=5000):
    a, b = np.asarray(before, float), np.asarray(after, float)
    g = np.asarray(groups)
    if a.shape != b.shape or a.ndim != 1 or len(g) != len(a) or not len(a) or not np.isfinite(a + b).all():
        raise ValueError("finite paired windows with story IDs required")
    # Preserve window/token weighting while resampling complete stories, never windows independently.
    ids = np.unique(g)
    samples = [b[g == i] - a[g == i] for i in ids]
    rng = np.random.default_rng(seed)
    draws = [
        float(np.concatenate([samples[i] for i in rng.integers(len(ids), size=len(ids))]).mean())
        for _ in range(replicates)
    ]
    return {
        "mean_delta": float((b - a).mean()),
        "ci95_two_sided": np.percentile(draws, [2.5, 97.5]).tolist(),
        "upper95_one_sided": float(np.percentile(draws, 95)),
        "stories": len(ids),
        "windows": len(a),
        "replicates": replicates,
        "statistical_seed": seed,
    }


def noninferiority(moe, candidate, margin=math.log(1.02), replicates=5000, seed=1907):
    for key in (
        "asset_sha",
        "tokenizer_sha",
        "evaluation_sha",
        "role",
        "source_sha",
        "config_sha",
        "protocol_sha",
    ):
        if moe["identity"][key] != candidate["identity"][key]:
            raise ValueError("unmatched " + key)
    for key in ("seed", "target_tokens", "training_budget_sha"):
        if moe["identity"][key] != candidate["identity"][key]:
            raise ValueError("unmatched training " + key)
    if not moe.get("baseline_stable", False):
        return {"status": "BASELINE_UNDERTRAINED", "scientific_decision": False}
    a, b = moe["quality"]["greedy"], candidate["quality"]["greedy"]
    if a["window_ids"] != b["window_ids"] or a["groups"] != b["groups"]:
        raise ValueError("window alignment differs")
    result = paired_interval(a["window_nll"], b["window_nll"], a["groups"], seed, replicates)
    result.update(
        ppl_ratio=math.exp(result["mean_delta"]),
        margin=margin,
        within_seed_noninferiority=result["upper95_one_sided"] < margin,
        scientific_decision=False,
        status="WITHIN_SEED_DIAGNOSTIC",
    )
    return result


def seed_summary(rows):
    if len(rows) < 3 or len({r["seed"] for r in rows}) != len(rows):
        return {
            "status": "INCOMPLETE",
            "scientific_decision": False,
            "reason": "at least three independent matched training seeds required",
        }
    upper = [r["upper95_one_sided"] for r in rows]
    return {
        "status": "INCONCLUSIVE"
        if not all(r["within_seed_noninferiority"] for r in rows)
        else "CONSISTENT_WITHIN_SEED_NONINFERIORITY",
        "per_seed": rows,
        "between_seed_delta_std": float(np.std([r["mean_delta"] for r in rows], ddof=1)),
        "upper_bounds": upper,
        "scientific_decision": False,
        "note": "final hypothesis decisions require frozen consistency and power review; no pooling hides seed disagreements",
    }


def internal_comparison(before, after, seed=1907, replicates=5000):
    """Independent H2/H3/H4/H5 comparisons; do not infer MoE noninferiority."""
    for key in (
        "asset_sha",
        "tokenizer_sha",
        "evaluation_sha",
        "seed",
        "role",
        "target_tokens",
        "training_budget_sha",
        "source_sha",
        "config_sha",
        "protocol_sha",
    ):
        if before["identity"][key] != after["identity"][key]:
            raise ValueError("unmatched internal comparison " + key)
    a, b = before["quality"]["greedy"], after["quality"]["greedy"]
    if a["window_ids"] != b["window_ids"] or a["groups"] != b["groups"]:
        raise ValueError("unpaired windows")
    result = paired_interval(a["window_nll"], b["window_nll"], a["groups"], seed, replicates)
    result.update(
        before_arm=before["identity"]["arm"],
        after_arm=after["identity"]["arm"],
        before_forward_gemm=a["cost"]["observed_gemm_flops_per_token"],
        after_forward_gemm=b["cost"]["observed_gemm_flops_per_token"],
        scientific_decision=False,
        status="PAIRED_DEVELOPMENT_DIAGNOSTIC",
    )
    return result


def baseline_review(history, criteria):
    if not criteria:
        return False
    count = criteria["tail_steps"]
    if len(history) < count + 1:
        return False
    values = np.asarray([row["train_probe_greedy_nll"] for row in history], float)
    tail = values[-count:]
    if not np.isfinite(values[0]) or not np.isfinite(tail).all():
        return False  # missing observations cannot support baseline stability
    slope = float(np.polyfit(np.arange(count), tail, 1)[0])
    return bool(
        values[0] - tail.mean() >= criteria["minimum_nll_drop"]
        and tail.std() / max(1e-12, abs(tail.mean())) <= criteria["tail_relative_std_max"]
        and abs(slope) <= criteria["tail_absolute_slope_max"]
    )
