"""State-level ranking, Story-cluster summaries, fixed Calibration quantile bins."""

import numpy as np
from minicells_r3d.reports import paired_story_interval
from .analysis import distribution


def ranks(x):
    a = np.asarray(x, float)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    start = ends - counts
    return (start + (counts - 1) / 2 + 1)[inverse]


def correlation(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def ranking(scores, loss, c):
    a, b = np.triu_indices(len(loss), 1)
    sd = scores[a] - scores[b]
    vd = loss[b] - loss[a]
    st = c["statistics"]
    valid = np.abs(vd) > st["value_tie_tolerance"]
    ssign = np.where(np.abs(sd) <= st["score_tie_tolerance"], 0, np.sign(sd))
    vsign = np.where(valid, np.sign(vd), 0)
    correct = np.where(ssign == 0, 0.5, (ssign == vsign).astype(float))
    denominator = np.sqrt(np.sum(ssign != 0) * np.sum(vsign != 0))
    tau = float(np.sum(ssign * vsign) / denominator) if denominator else np.nan
    return {
        "accuracy": float(correct[valid].mean()) if valid.any() else np.nan,
        "spearman": correlation(ranks(scores), ranks(-loss)),
        "kendall_tau_b": tau,
        "pair_gaps": np.abs(vd),
        "pair_valid": valid,
        "pair_correct": correct,
        "value_ranks": ranks(loss),
        "top3": np.argsort(-scores, kind="stable")[:3],
    }


def clustered(values, story_ids, seed, replicates):
    values = np.asarray(values, float)
    groups = np.asarray(story_ids)
    per = []
    for key in dict.fromkeys(groups.tolist()):
        rows = values[groups == key]
        rows = rows[np.isfinite(rows)]
        if len(rows):
            per.append(float(rows.mean()))
    if not per:
        return {"mean": None, "ci95": None, "stories": 0, "states": len(values)}
    a = np.asarray(per)
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(len(a), size=(replicates, len(a)))].mean(1)
    return {
        "mean": float(a.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist() if len(a) > 1 else None,
        "stories": len(a),
        "states": len(values),
        "unit": "Story cluster within one training Seed; Stage/Token/pairs are not independent training replicas",
    }


def calibrate_bins(calibration, c):
    thresholds = []
    for t in range(4):
        a, b = np.triu_indices(15, 1)
        v = calibration["long"][:, t]
        gaps = np.abs(v[:, b] - v[:, a]).reshape(-1)
        thresholds.append(np.quantile(gaps, c["statistics"]["gap_quantiles"]).tolist())
    return thresholds


def summaries(arrays, selected, c, seed, cutoffs):
    values = []
    policy = []
    raw = []
    for i, row in enumerate(selected):
        for t in range(4):
            long = arrays["long"][i, t]
            short = arrays["short"][i, t]
            q = arrays["score"][i, t]
            base = arrays["base_score"][i, t]
            current = int(arrays["action"][i, t])
            original = int(arrays["base"][i, t])
            qr = ranking(q, long, c)
            br = ranking(base, long, c)
            sr = ranking(-short, long, c)
            best = float(long.min())
            vr = qr["value_ranks"]
            top3 = qr["top3"]
            truth3 = np.argsort(long, kind="stable")[:3]
            metric = {
                "stage": t,
                "story": row["story"],
                "gap": float(long[original] - best),
                "selected_regret": float(long[current] - best),
                "base_rank": float(vr[original]),
                "selected_rank": float(vr[current]),
                "range": float(long.max() - best),
                "long_short_spearman": correlation(ranks(-long), ranks(-short)),
                "long_short_pairwise": sr["accuracy"],
                "q_accuracy": qr["accuracy"],
                "q_spearman": qr["spearman"],
                "q_kendall": qr["kendall_tau_b"],
                "q_minus_base_accuracy": qr["accuracy"] - br["accuracy"],
                "q_minus_base_spearman": qr["spearman"] - br["spearman"],
                "base_accuracy": br["accuracy"],
                "base_spearman": br["spearman"],
                "base_kendall": br["kendall_tau_b"],
                "top3_contains_best": float(int(np.argmin(long)) in top3),
                "top3_contains_low_loss_top3": float(bool(set(top3) & set(truth3))),
            }
            raw.append(metric)
            bins = np.searchsorted(np.asarray(cutoffs[t])[1:-1], qr["pair_gaps"], side="right")
            for k in range(4):
                mask = qr["pair_valid"] & (bins == k)
                metric["gap_bin_" + str(k)] = (
                    float(qr["pair_correct"][mask].mean()) if mask.any() else np.nan
                )
                metric["gap_bin_pairs_" + str(k)] = int(mask.sum())
            mask = qr["pair_valid"] & (
                qr["pair_gaps"] >= max(cutoffs[t][3], c["statistics"]["value_tie_tolerance"])
            )
            metric["clear_gap_pairs"] = int(mask.sum())
            metric["all_scored_value_pairs"] = int(qr["pair_valid"].sum())
            metric["clear_gap_accuracy"] = (
                float(qr["pair_correct"][mask].mean()) if mask.any() else np.nan
            )
    st = c["statistics"]
    numeric = [k for k in raw[0] if k not in ("stage", "story")]

    def aggregate(rows):
        return {
            key: clustered(
                [r[key] for r in rows],
                [r["story"] for r in rows],
                seed + st["bootstrap_seed"],
                st["bootstrap_replicates"],
            )
            for key in numeric
        }

    for t in range(4):
        rows = [r for r in raw if r["stage"] == t]
        agg = aggregate(rows)
        values.append(
            {
                "stage": t,
                "per_candidate": [
                    {
                        "action": a,
                        "long_ce": distribution(arrays["long"][:, t, a]),
                        "short_ce": distribution(arrays["short"][:, t, a]),
                        "mean_original_score": float(arrays["base_score"][:, t, a].mean()),
                        "mean_corrected_score": float(arrays["score"][:, t, a].mean()),
                        "mean_correction": float(arrays["delta"][:, t, a].mean()),
                    }
                    for a in range(15)
                ],
                "long_ce_distribution": distribution(arrays["long"][:, t]),
                "short_ce_distribution": distribution(arrays["short"][:, t]),
                "conditional_action_range": agg["range"],
                "hindsight_candidate_opportunity_gap": agg["gap"],
                "base_action_value_rank": agg["base_rank"],
                "chosen_action_value_rank": agg["selected_rank"],
                "selected_hindsight_regret": agg["selected_regret"],
                "long_short_rank_spearman": agg["long_short_spearman"],
                "long_short_pairwise_agreement": agg["long_short_pairwise"],
            }
        )
        policy.append(
            {
                "stage": t,
                "metrics": agg,
                "gap_quantile_levels": st["gap_quantiles"],
                "gap_cutoffs_from_calibration": cutoffs[t],
                "tie_rule": "Value ties excluded; score ties get half credit; Spearman average ranks; Kendall tau-b",
                "scope": "full 15 actions at frozen observed states; target used only for scoring",
            }
        )
    return {
        "stages": values,
        "overall": aggregate(raw),
        "gap_interpretation": "HINDSIGHT CANDIDATE OPPORTUNITY GAP; label-advantaged observation, not deployable oracle or guaranteed achievable gain",
    }, {"stages": policy, "overall": aggregate(raw)}


def interpolation_public(raw, c, seed):
    baseline = raw["0.0"]["stories"]
    out = {}
    for scale, row in raw.items():
        pair = paired_story_interval(
            row["stories"],
            baseline,
            seed + c["statistics"]["bootstrap_seed"],
            c["statistics"]["bootstrap_replicates"],
        )
        out[scale] = {
            "mean_nll": row["mean_nll"],
            "tokens": row["tokens"],
            "delta_to_same_checkpoint_original_route": pair,
            "stages": row["stages"],
        }
    return {
        "scales": out,
        "scope": "same fixed Cell/common weights; fresh closed-loop rollouts from input for each lambda; no deployment lambda selected",
        "margin_definition": "original Gate second-highest minus third-highest logit; Top2 selection boundary",
        "local_change": "chosen(lambda state) versus original Gate at that same state",
        "trajectory_change": "chosen(lambda state) versus actual lambda=0 trajectory at matching input token/Stage",
    }
