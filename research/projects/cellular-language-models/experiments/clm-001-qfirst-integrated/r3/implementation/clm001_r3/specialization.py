"""Predeclared crossover contrasts and descriptive longitudinal routes; no causal overclaim."""

from copy import copy
import numpy as np
from .audit import capture, measure, clustered


def rebind(bank, model):
    result = []
    for item in bank:
        row = dict(item)
        row["state"] = copy(item["state"])
        row["state"].registry_sha = model.registry.fingerprint
        result.append(row)
    return result


def crossover(records, c):
    candidates = []
    families = len(c["alignment"]["cell_pairs"]) * 2
    # Bonferroni confidence bands: conservative across the predeclared pairs and two groups.
    for pair in c["alignment"]["cell_pairs"]:
        stats = []
        for group in (0, 1):
            rows = [
                r
                for r in records
                if (r["cell_i"], r["cell_j"]) == tuple(pair) and r["group_feature"] == group
            ]
            values = [-r["fixed_suffix_gap"] for r in rows]
            groups = [r["group"] for r in rows]
            stat = clustered(values, groups, c["alignment"]["replicates"], c["seed"])
            if values:
                keys = np.unique(groups)
                arr = np.array(
                    [np.mean([v for v, g in zip(values, groups) if g == key]) for key in keys]
                )
                rng = np.random.default_rng(c["seed"])
                idx = rng.integers(len(keys), size=(c["alignment"]["replicates"], len(keys)))
                stat["familywise_ci"] = np.quantile(
                    arr[idx].mean(1), [0.025 / families, 1 - 0.025 / families]
                ).tolist()
            else:
                stat["familywise_ci"] = None
            stats.append(stat)
        a, b = [s["familywise_ci"] for s in stats]
        crossed = (
            a is not None and b is not None and ((a[0] > 0 and b[1] < 0) or (a[1] < 0 and b[0] > 0))
        )
        candidates.append(
            {"cells": pair, "groups": stats, "conditional_crossover_candidate": crossed}
        )
    return {
        "contrasts": candidates,
        "candidate_count": sum(r["conditional_crossover_candidate"] for r in candidates),
        "multiplicity": "Bonferroni over predeclared pair-by-group family",
        "group_definition": "initial_input_token_id_mod_2",
        "scope": "exploratory_crossover_candidates_not_validated_specialization",
    }


def route_change(previous, current):
    if previous is None:
        return {"status": "NO_PREVIOUS_SNAPSHOT"}
    if len(previous) != len(current):
        return {"status": "COVERAGE_CHANGED"}
    if not current:
        return {"status": "NO_STATES"}
    p = np.array([r["cell_probabilities"] for r in previous])
    q = np.array([r["cell_probabilities"] for r in current])
    m = np.maximum((p + q) / 2, 1e-30)
    js = 0.5 * np.sum(p * np.log(np.maximum(p, 1e-30) / m), axis=1) + 0.5 * np.sum(
        q * np.log(np.maximum(q, 1e-30) / m), axis=1
    )
    return {
        "action_flip_rate": float((p.argmax(1) != q.argmax(1)).mean()),
        "mean_js": float(js.mean()),
        "states": len(current),
    }


def snapshot(model, data, c, initial_bank, step, previous=None, validation_role="development"):
    live, missing = capture(model, data, c)
    frozen = measure(model, rebind(initial_bank, model), c, resampled=False)
    current = measure(model, live, c, resampled=False)
    independent, independent_missing = capture(
        model, data, c, role=validation_role, independent=True
    )
    validation = measure(model, independent, c, resampled=False)
    first = crossover(current["records"], c)
    second = crossover(validation["records"], c)
    stable = [
        a["cells"]
        for a, b in zip(first["contrasts"], second["contrasts"])
        if a["conditional_crossover_candidate"]
        and b["conditional_crossover_candidate"]
        and np.sign(a["groups"][0]["mean"]) == np.sign(b["groups"][0]["mean"])
    ]
    old = previous or {}
    return {
        "step": step,
        "source_model_sha": frozen["records"][0]["snapshot_sha"] if frozen["records"] else None,
        "frozen_state_probes": frozen,
        "live_recomputed_probes": current,
        "independent_development_story_validation": validation,
        "crossover": first,
        "crossover_validation": second,
        "stable_crossover_pairs": stable,
        "coverage_missing": len(missing),
        "independent_coverage_missing": len(independent_missing),
        "fixed_route_change": route_change(
            old.get("frozen_state_probes", {}).get("state_records"), frozen["state_records"]
        ),
        "live_route_change": route_change(
            old.get("live_recomputed_probes", {}).get("state_records"), current["state_records"]
        ),
        "causal_feedback_status": "CAUSAL_FEEDBACK_NOT_TESTED",
        "scientific_decision": False,
    }


def longitudinal(snapshots):
    steps = [s["step"] for s in snapshots]
    trajectories = []
    for snapshot in snapshots:
        for mode in ("frozen_state_probes", "live_recomputed_probes"):
            rows = snapshot[mode]["records"]
            keys = sorted(set((r["cell_i"], r["cell_j"], r["group_feature"]) for r in rows))
            for a, b, group in keys:
                selected = [
                    r
                    for r in rows
                    if (r["cell_i"], r["cell_j"], r["group_feature"]) == (a, b, group)
                ]
                trajectories.append(
                    {
                        "step": snapshot["step"],
                        "mode": mode,
                        "cells": [a, b],
                        "group": group,
                        "A_ij": float(np.mean([-r["fixed_suffix_gap"] for r in selected])),
                        "R_ij": float(np.mean([r["route_log_preference"] for r in selected])),
                        "states": len(selected),
                    }
                )
    return {
        "snapshots": steps,
        "advantage_route_trajectories": trajectories,
        "fixed_and_live_separated": True,
        "stable_crossover_pairs_by_step": {
            str(s["step"]): s["stable_crossover_pairs"] for s in snapshots
        },
        "fixed_route_change_by_step": {str(s["step"]): s["fixed_route_change"] for s in snapshots},
        "live_route_change_by_step": {str(s["step"]): s["live_route_change"] for s in snapshots},
        "advantage_leads_route": "DESCRIPTIVE_TRAJECTORIES_ONLY",
        "route_leads_advantage": "DESCRIPTIVE_TRAJECTORIES_ONLY",
        "causal_feedback_status": "CAUSAL_FEEDBACK_NOT_TESTED",
        "seed_replication": "NOT_TESTED",
        "scientific_decision": False,
    }
