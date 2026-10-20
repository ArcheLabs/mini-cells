"""Label-free frozen projection groups, and independently evaluated Gate utility."""

import numpy as np
from .common import canonical


def freeze_groups(states, state_ids, *, role, seed):
    x = np.asarray(states, float)
    if (
        role != "calibration"
        or x.ndim != 2
        or len(x) != len(state_ids)
        or not len(x)
        or len(set(state_ids)) != len(state_ids)
        or not np.isfinite(x).all()
    ):
        raise ValueError("unique finite Calibration states required")
    projection = np.random.default_rng(seed).normal(size=x.shape[1])
    projection /= np.linalg.norm(projection)
    rule = {
        "projection": projection.tolist(),
        "threshold": float(np.median(x @ projection)),
        "selection_role": role,
        "discovery_ids": list(state_ids),
        "seed": seed,
        "uses_targets": False,
    }
    rule["sha256"] = canonical(rule)
    return rule


def apply_groups(states, rule, state_ids, *, role):
    content = {k: v for k, v in rule.items() if k != "sha256"}
    if canonical(content) != rule["sha256"]:
        raise ValueError("group rule changed")
    if role not in ("calibration", "development"):
        raise ValueError("Confirmation sealed")
    if role == "development" and set(state_ids) & set(rule["discovery_ids"]):
        raise ValueError("discovery/validation overlap")
    x = np.asarray(states, float)
    w = np.asarray(rule["projection"])
    if x.ndim != 2 or x.shape[1] != len(w) or len(x) != len(state_ids) or not np.isfinite(x).all():
        raise ValueError("invalid state features")
    return (x @ w >= rule["threshold"]).astype(int)


def gate_utility(
    indices,
    selection_values,
    validation_values,
    *,
    selection_stream,
    validation_stream,
    identifiable_margin,
):
    # A selected pair is compared with the independent best pair. Oracle never feeds inference.
    if selection_stream == validation_stream:
        raise ValueError("independent value estimates required")
    ids = np.asarray(indices, int)
    a = np.asarray(selection_values, float)
    b = np.asarray(validation_values, float)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or ids.shape != (len(a), 2)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or identifiable_margin < 0
    ):
        raise ValueError("invalid Gate utility inputs")
    if (ids < 0).any() or (ids >= a.shape[1]).any() or (ids[:, 0] == ids[:, 1]).any():
        raise ValueError("distinct legal pair required")
    ranks = np.argsort(a, axis=1, kind="stable")
    best = ranks[:, :2]
    rows = np.arange(len(a))[:, None]
    selected = b[rows, ids].mean(1)
    oracle = b[rows, best].mean(1)
    mask = np.sort(a, axis=1)[:, 2] - np.sort(a, axis=1)[:, 1] > identifiable_margin
    return {
        "identifiable_states": int(mask.sum()),
        "unidentifiable_states": int((~mask).sum()),
        "ranking_agreement": float((ids[:, :, None] == best[:, None, :]).any(-1)[mask].mean())
        if mask.any()
        else None,
        "regret": float((selected - oracle)[mask].mean()) if mask.any() else None,
        "selected_advantage": float((b.mean(1) - selected)[mask].mean()) if mask.any() else None,
        "oracle_inference": False,
        "status": "EXPLORATORY_ONLY",
    }
