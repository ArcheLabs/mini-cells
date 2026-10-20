"""Paired Story-cluster inference; no token-as-independent-sample inference."""

import math
import numpy as np
from minicells_runtime.store import ContinuationError


def story_summary(rows, groups, weights=None):
    x = np.asarray(rows, dtype=np.float64)
    g = np.asarray(groups)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    if (
        x.ndim != 1
        or len(x) != len(g)
        or len(w) != len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or (w <= 0).any()
    ):
        raise ContinuationError("CF_INVALID_PAIRED_VALUES")
    return {str(k): float(np.average(x[g == k], weights=w[g == k])) for k in np.unique(g)}


def cluster_interval(values, seed=20261019, replicates=2000):
    x = np.asarray(values, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ContinuationError("CF_INVALID_STORY_VALUES")
    if len(x) < 2:
        return {
            "mean": float(x.mean()),
            "story_sd": None,
            "ci95": None,
            "stories": len(x),
            "status": "INSUFFICIENT_STORIES",
        }
    gen = np.random.default_rng(seed)
    means = x[gen.integers(len(x), size=(replicates, len(x)))].mean(-1)
    # Two-sided centred cluster-bootstrap tail, exploratory only.
    p = float((1 + np.sum(np.abs(means - x.mean()) >= abs(x.mean()))) / (replicates + 1))
    return {
        "mean": float(x.mean()),
        "story_sd": float(x.std(ddof=1)),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "stories": len(x),
        "exploratory_bootstrap_p": p,
        "status": "EXPLORATORY_SINGLE_SEED",
    }


def paired(base, changed, lam, seed=20261019, replicates=2000):
    if set(base) != set(changed):
        raise ContinuationError("CF_STORY_PAIRING_MISMATCH")
    fields = {k: [] for k in ("delta_nll", "delta_cost", "delta_utility")}
    triggered = {k: [] for k in fields}
    keys = []
    touched_count = 0
    token_count = 0
    worse = []
    for key in sorted(base):
        a, b = base[key], changed[key]
        na, nb = np.asarray(a["nll"]), np.asarray(b["nll"])
        ca, cb = np.asarray(a["cost"]), np.asarray(b["cost"])
        if na.shape != nb.shape or ca.shape != cb.shape or na.shape != ca.shape:
            raise ContinuationError("CF_TOKEN_PAIRING_MISMATCH")
        mask = np.asarray(b["touched"], dtype=bool)
        if mask.shape != na.shape:
            raise ContinuationError("CF_TRIGGER_PAIRING_MISMATCH")
        deltas = {
            "delta_nll": nb - na,
            "delta_cost": cb - ca,
            "delta_utility": nb - na + lam * (cb - ca),
        }
        for name, array in deltas.items():
            if not np.isfinite(array).all():
                raise ContinuationError("CF_NONFINITE_DELTA")
            fields[name].append(float(array.mean()))
            if mask.any():
                triggered[name].append(float(array[mask].mean()))
        touched_count += int(mask.sum())
        token_count += len(mask)
        worse.append(float((nb > na).mean()))
        keys.append(key)
    report = {name: cluster_interval(v, seed, replicates) for name, v in fields.items()}
    report["triggered_subset"] = {
        name: cluster_interval(v, seed, replicates) if v else {"status": "NO_TRIGGERED_TOKENS"}
        for name, v in triggered.items()
    }
    report.update(
        paired_stories=len(keys),
        touched_token_fraction=touched_count / token_count,
        triggered_subset_interpretation="CONDITIONED_DESCRIPTIVE_NOT_ITT",
        mean_token_nll_worsening_fraction=float(np.mean(worse)),
        independent_training_seeds=1,
    )
    return report


def holm(pvalues, alpha=0.05):
    names = sorted(pvalues, key=pvalues.get)
    out = {}
    previous = 0
    for i, name in enumerate(names):
        value = pvalues[name]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ContinuationError("CF_INVALID_PVALUE")
        adjusted = min(1, max(previous, (len(names) - i) * value))
        previous = adjusted
        out[name] = {
            "adjusted_exploratory_p": adjusted,
            "below_alpha": adjusted < alpha,
            "formal_confirmation": False,
        }
    return out
