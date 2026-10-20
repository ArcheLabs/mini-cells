"""Seed is the replication unit; Story is the paired observation unit inside a Seed."""

import itertools
import numpy as np


def story_interval(x, seed, replicates=2000):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("finite Story values required")
    if len(a) < 2:
        return {
            "mean": float(a.mean()),
            "ci95": None,
            "stories": len(a),
            "status": "INSUFFICIENT_STORIES",
        }
    rng = np.random.default_rng(seed)
    samples = a[rng.integers(len(a), size=(replicates, len(a)))].mean(1)
    p = float((1 + np.sum(np.abs(samples - a.mean()) >= abs(a.mean()))) / (replicates + 1))
    return {
        "mean": float(a.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "stories": len(a),
        "exploratory_bootstrap_p": p,
    }


def paired_seed_comparison(arms, seed, replicates=2000):
    if set(arms) != {"A", "B", "C", "D", "E"}:
        raise ValueError("five complete arms required")
    out = {}
    for control in ("A", "B", "D", "E"):
        if set(arms["C"]) != set(arms[control]):
            raise ValueError("Story pairing differs")
        keys = sorted(arms["C"])
        delta = [arms["C"][k]["mean_nll"] - arms[control][k]["mean_nll"] for k in keys]
        out["C-" + control] = story_interval(delta, seed, replicates)
    family = sorted(
        (k for k in out if k != "C-E"), key=lambda k: out[k].get("exploratory_bootstrap_p", 1)
    )
    previous = 0.0
    for i, k in enumerate(family):
        if "exploratory_bootstrap_p" not in out[k]:
            continue
        previous = max(previous, min(1.0, (len(family) - i) * out[k]["exploratory_bootstrap_p"]))
        out[k]["holm_exploratory_p"] = previous
    return {
        "comparisons": out,
        "seed": seed,
        "formal_confirmation": False,
        "status": "EXPLORATORY_ONLY",
    }


def seed_variation(reports):
    seeds = [r["seed"] for r in reports]
    if len(seeds) != len(set(seeds)):
        raise ValueError("training Seed duplicated")
    return {
        "independent_training_seeds": len(seeds),
        "per_seed": reports,
        "seed_sd": {
            k: float(np.std([r["comparisons"][k]["mean"] for r in reports], ddof=1))
            if len(seeds) > 1
            else None
            for k in ("C-A", "C-B", "C-D", "C-E")
        },
        "scientific_decision": False,
    }


def crossfit_oracle(selection, validation, *, selection_stream, validation_stream):
    a, b = np.asarray(selection, float), np.asarray(validation, float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired independent counterfactual values required")
    if selection_stream == validation_stream:
        raise ValueError("independent selection/validation evidence required")
    fixed = int(a.mean(0).argmin())
    chosen = a.argmin(1)
    fixed_value = float(b[:, fixed].mean())
    oracle = float(b[np.arange(len(b)), chosen].mean())
    return {
        "best_fixed_cell": fixed,
        "best_fixed_cell_validation_cost": fixed_value,
        "statewise_crossfit_oracle_cost": oracle,
        "oracle_potential_gap": fixed_value - oracle,
        "scope": "UPPER_BOUND_DIAGNOSTIC_NOT_ACHIEVED_ROUTE_BENEFIT",
        "inference_uses_oracle": False,
    }


def conditional_groups(
    selection,
    groups_a,
    validation,
    groups_b,
    *,
    selection_role,
    validation_role,
    selection_ids,
    validation_ids,
):
    if (
        selection_role != "calibration"
        or validation_role != "development"
        or set(selection_ids) & set(validation_ids)
    ):
        raise ValueError("independent Calibration/Development state sets required")
    a, b = np.asarray(selection, float), np.asarray(validation, float)
    ga, gb = np.asarray(groups_a), np.asarray(groups_b)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or len(a) != len(ga)
        or len(b) != len(gb)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("invalid conditional values")
    rules = {str(g): int(a[ga == g].mean(0).argmin()) for g in np.unique(ga)}
    if any(str(g) not in rules for g in np.unique(gb)):
        raise ValueError("unseen validation group")
    means = {str(g): b[gb == g].mean(0).tolist() for g in np.unique(gb)}
    return {
        "calibration_selected_cells": rules,
        "development_group_cell_means": means,
        "status": "THRESHOLDS_NOT_FROZEN_EXPLORATORY_ONLY",
        "groups_use_targets": False,
        "inference_uses_targets": False,
    }


def align_cells(reference, current):
    a, b = np.asarray(reference, float), np.asarray(current, float)
    if a.shape != b.shape or a.ndim != 2 or len(a) > 6:
        raise ValueError("up to six matched functional signatures required")
    permutation = min(
        itertools.permutations(range(len(a))), key=lambda p: float(((a - b[list(p)]) ** 2).sum())
    )
    return list(permutation)


def temporal_summary(exposure, ability, routing):
    a, b, c = map(lambda x: np.asarray(x, float), (exposure, ability, routing))
    if a.shape != b.shape or a.shape != c.shape or a.ndim != 3 or len(a) < 3:
        raise ValueError("matched time/group/Cell series required")

    def correlation(x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None

    return {
        "exposure_then_ability": correlation(a[:-2], b[1:-1]),
        "ability_then_routing": correlation(b[1:-1], c[2:]),
        "claim": "CORRELATION_ONLY_CAUSAL_SELF_REINFORCEMENT_NOT_TESTED",
    }
