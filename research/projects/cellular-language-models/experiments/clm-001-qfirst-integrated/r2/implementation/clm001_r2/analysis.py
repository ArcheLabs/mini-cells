"""Exploratory decisions from independent evidence; no confirmation or expansion approval."""

import numpy as np


def interval(values, groups, c):
    groups = np.asarray(groups)
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    keys = np.unique(groups)
    cluster = np.array([values[groups == g].mean() for g in keys])
    if not np.isfinite(cluster).all():
        raise ValueError("nonfinite diagnostic evidence")
    rng = np.random.default_rng(c["evaluation"]["seed"])
    indices = rng.integers(0, len(keys), (c["evaluation"]["bootstrap_replicates"], len(keys)))
    return {
        "mean": float(cluster.mean()),
        "ci95": np.quantile(cluster[indices].mean(1), [0.025, 0.975]).tolist(),
        "stories": len(keys),
        "scope": "conditional_identifiable_fixed_calibration_states_finite_MC_single_seed",
    }


def decide(c, results, pre_contrast, uniform_contrast):
    if len(results) != 2:
        return {
            "decision": "ENGINEERING_INCOMPLETE",
            "q_signal_status": "UNREVIEWED",
            "policy_status": "UNREVIEWED",
        }
    q = results["QF-R2"].get("q_calibration", {})
    records = [
        r for r in q.get("records", []) if not r.get("coverage_missing") and r.get("identifiable")
    ]
    if not records:
        return {
            "decision": "Q_UNIDENTIFIABLE",
            "q_signal_status": "NO_SIGNAL",
            "policy_status": "UNREVIEWED",
        }
    gains = interval(
        [r["regret"] - r["uniform_regret"] for r in records], [r["group"] for r in records], c
    )
    ranks = [r for r in records if r.get("spearman") is not None]
    rank = interval([r["spearman"] for r in ranks], [r["group"] for r in ranks], c)
    q_positive = gains["ci95"][1] < 0 and rank is not None and rank["ci95"][0] > 0
    policy_gain = (
        pre_contrast.get("utility", {}).get("ci95", [0, 0])[1] < 0
        and uniform_contrast.get("utility", {}).get("ci95", [0, 0])[1] < 0
    )
    quality = results["QF-R2"]["quality"]["greedy"]
    degenerate = quality.get("forced_end_rate") == 1.0 or quality.get("actual_stop_rate") == 1.0
    stop_error = q.get("stop_error_rate")
    collapse = degenerate and stop_error is not None and stop_error > 0
    decision = (
        "STOP_COLLAPSE"
        if collapse
        else (
            "QF_DEV_SIGNAL"
            if q_positive and policy_gain
            else ("Q_SIGNAL_NO_POLICY_GAIN" if q_positive else "NO_Q_ADVANTAGE")
        )
    )
    return {
        "decision": decision,
        "q_signal_status": "POSITIVE_DEV" if q_positive else "NO_SIGNAL",
        "policy_status": "STOP_COLLAPSE"
        if collapse
        else ("ADVANTAGE_DEV" if policy_gain else "NO_GAIN"),
        "q_regret_minus_uniform": gains,
        "q_rank_spearman": rank,
        "route_degeneracy_observed": degenerate,
        "scientific_decision": False,
        "expansion_budget_approved": False,
        "claim_scope": "EXPLORATORY_DEV_SIGNAL_ONLY",
        "warning": "Story intervals do not establish seed replication; finite-MC identifiability and state aliasing remain limitations.",
    }
