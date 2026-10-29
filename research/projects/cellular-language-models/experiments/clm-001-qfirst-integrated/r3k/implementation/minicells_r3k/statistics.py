"""Fixed metrics, shared-trajectory resampling and multiplicity-adjusted exploratory contrasts."""

import numpy as np
from minicells_r3j.reports import correlation
from .prediction_freeze import SCORES


def metrics(rows, score):
    x = [r["scores"][score] for r in rows]
    y = [r["V"] for r in rows]
    resolved = [
        r
        for r in rows
        if abs(r["V"]) > r["outcome_error"] and abs(r["scores"][score]) > r["prediction_error"]
    ]
    positive = [r for r in resolved if r["V"] > 0]
    true_positive = sum(r["scores"][score] > 0 for r in positive)
    return dict(
        n=len(rows),
        rho=correlation(x, y),
        resolved=len(resolved),
        sign_accuracy=sum(np.sign(r["V"]) == np.sign(r["scores"][score]) for r in resolved)
        / len(resolved)
        if resolved
        else None,
        abstention_fraction=1 - len(resolved) / len(rows) if rows else None,
        positive_coverage=true_positive / len(positive) if positive else None,
        false_positive=sum(r["scores"][score] > 0 and r["V"] < 0 for r in resolved),
        false_negative=sum(r["scores"][score] < 0 and r["V"] > 0 for r in resolved),
    )


def analyse(rows, c):
    groups = {"R3K_NEW36": [r for r in rows if not r["reused_r3j"]], "R3K_ALL48": rows}
    for key in ("seed", "arm", "historical_accepted"):
        for value in sorted(set(r[key] for r in rows)):
            groups[f"{key}={value}"] = [r for r in rows if r[key] == value]
    for architecture in ("A", "B"):
        groups["architecture=" + architecture] = [r for r in rows if r["arm"][0] == architecture]
    for routing in ("H", "T"):
        groups["routing=" + routing] = [r for r in rows if r["arm"][-1] == routing]
    result = {}
    for group, sample in groups.items():
        result[group] = {score: metrics(sample, score) for score in SCORES}
        if len(sample) < 5:
            for row in result[group].values():
                row["rho"] = None
    sample = groups["R3K_NEW36"]
    n = len(sample)
    seed_groups = {
        seed: [r for r in sample if r["seed"] == seed]
        for seed in sorted(set(r["seed"] for r in sample))
    }
    for score in SCORES:
        row = result["R3K_NEW36"][score]
        leave = [
            correlation(
                [r["scores"][score] for j, r in enumerate(sample) if j != i],
                [r["V"] for j, r in enumerate(sample) if j != i],
            )
            for i in range(n)
        ]
        positive_seeds = sum((metrics(v, score)["rho"] or 0) > 0 for v in seed_groups.values())
        row.update(
            positive_seeds=positive_seeds,
            leave_one_out_min_rho=min((v for v in leave if v is not None), default=None),
        )
        row["status"] = (
            "PREDICTION_EVIDENCE_INCONCLUSIVE"
            if n != 36 or row["resolved"] < 20
            else (
                "PREDICTIVE_SIGNAL_EXPLORATORY"
                if (row["rho"] or 0) >= 0.30
                and row["sign_accuracy"] >= 0.70
                and positive_seeds >= 2
                and (row["leave_one_out_min_rho"] or 0) >= 0.20
                else "PREDICTIVE_SIGNAL_NOT_SUPPORTED"
            )
        )
    clusters = sorted(set((r["seed"], r["arm"]) for r in sample))
    rng = np.random.default_rng(c["bootstrap_seed"])
    comparisons = []
    for left, right in c["primary_comparisons"]:
        observed = (metrics(sample, left)["rho"] or 0) - (metrics(sample, right)["rho"] or 0)
        differences = []
        null = []
        for _ in range(c["bootstrap_replicates"]):
            selected = rng.integers(len(clusters), size=len(clusters)) if clusters else []
            boot = [r for i in selected for r in sample if (r["seed"], r["arm"]) == clusters[i]]
            differences.append(
                (metrics(boot, left)["rho"] or 0) - (metrics(boot, right)["rho"] or 0)
            )
            swap = {k: bool(rng.integers(2)) for k in clusters}
            trial = []
            for r in sample:
                x = dict(r, scores=dict(r["scores"]))
                if swap[(r["seed"], r["arm"])]:
                    x["scores"][left], x["scores"][right] = x["scores"][right], x["scores"][left]
                trial.append(x)
            null.append((metrics(trial, left)["rho"] or 0) - (metrics(trial, right)["rho"] or 0))
        p = (1 + sum(abs(v) >= abs(observed) for v in null)) / (len(null) + 1)
        comparisons.append(
            dict(
                left=left,
                right=right,
                rho_gain=observed,
                cluster_bootstrap_ci95=np.quantile(differences, [0.025, 0.975]).tolist(),
                sign_accuracy_gain=(metrics(sample, left)["sign_accuracy"] or 0)
                - (metrics(sample, right)["sign_accuracy"] or 0),
                unadjusted_exploratory_p=p,
            )
        )
    ordered = sorted(
        range(len(comparisons)), key=lambda i: comparisons[i]["unadjusted_exploratory_p"]
    )
    prior = 0
    for rank, i in enumerate(ordered):
        prior = max(
            prior, min(1, (len(ordered) - rank) * comparisons[i]["unadjusted_exploratory_p"])
        )
        comparisons[i]["holm_adjusted_exploratory_p"] = prior
    return dict(
        groups=result,
        primary_comparisons=comparisons,
        target="V100_R3K_VALUE_HOLDOUT",
        cell_mechanism_correlations={
            score: correlation([r["scores"][score] for r in sample], [r["C"] for r in sample])
            for score in SCORES
        },
        unit="Seed/Arm trajectory clusters; NOT 48 independent Seeds",
        limitation="Only three independent Seeds. Cluster bootstrap and within-trajectory score-label swaps assume exchangeability and are exploratory, not independent confirmation.",
        extreme_point_rule="Frozen leave-one-case-out minimum rho >= 0.20",
        scientific_confirmation=False,
    )
