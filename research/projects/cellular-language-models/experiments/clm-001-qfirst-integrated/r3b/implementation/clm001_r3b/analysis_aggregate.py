"""Manifest-driven aggregation. Missing observations remain missing, never zero."""

from minicells_runtime.store import canonical, ContinuationError
from .analysis import primary, summarize_router, mean, descriptive_lag


def aggregate_tasks(c, meta, tasks, results, ledger):
    planned = {canonical(t): t for t in tasks}
    found = {}
    for result in results:
        key = canonical(result["identity"])
        if key not in planned or result["task_status"] != "COMPLETE":
            raise ContinuationError("UNVERIFIED_ANALYSIS_TASK")
        if key in found and canonical(found[key]) != canonical(result):
            raise ContinuationError("ANALYSIS_TASK_RESULT_CONFLICT")
        found[key] = result
    missing = [
        {
            "task_key": k,
            "arm": t["arm"],
            "step": t["step"],
            "phase": t["phase"],
            "event_revision": t["event_revision"],
            "stage": t["stage"],
            "status": "NOT_EVALUATED",
        }
        for k, t in planned.items()
        if k not in found
    ]
    qualities, metrics, functionals, routers, trajectories, gradients = {}, [], [], [], [], []
    for row in found.values():
        t, result = row["identity"], row["result"]
        base = {
            k: t[k]
            for k in (
                "arm",
                "step",
                "phase",
                "event_revision",
                "checkpoint_sha256",
                "analysis_source_sha",
                "analysis_protocol_sha256",
                "role",
                "state_scope",
            )
        }
        gradients.append(
            {
                **base,
                "gradient_audit": row.get("gradient_audit", {}),
                "source": "ORIGINAL_TRAINER_JOURNAL",
            }
        )
        if t["stage"] == "quality":
            key = (t["step"], t["phase"], t["role"])
            arm_values = qualities.setdefault(key, {}).setdefault(t["arm"], {})
            for policy, value in result["quality"].items():
                if policy in arm_values and canonical(value) != canonical(arm_values[policy]):
                    raise ContinuationError("ANALYSIS_POLICY_CONFLICT")
                arm_values[policy] = value
            metrics.append(
                {
                    **base,
                    "quality": {
                        p: {
                            **{k: v for k, v in q.items() if k not in ("nll", "cost", "groups")},
                            "independent_stories": len(set(q["groups"])),
                            "windows": len(q["nll"]),
                        }
                        for p, q in result["quality"].items()
                    },
                    "samples": row["sample_count"],
                }
            )
        elif t["stage"] == "functional":
            m = result["functional"]
            functionals.append(
                {
                    **base,
                    "summary": m["summary"],
                    "crossover": m["crossover"],
                    "scored_states": len(m["state_records"]),
                    "paired_regret": mean([r["paired_regret"] for r in m["state_records"]]),
                    "rank_correlation": mean(
                        [r.get("rank_correlation") for r in m["state_records"]]
                    ),
                }
            )
            for pair in c["alignment"]["cell_pairs"]:
                for group in (0, 1):
                    records = [
                        r
                        for r in m["records"]
                        if (r["cell_i"], r["cell_j"], r["group_feature"]) == (*pair, group)
                    ]
                    trajectories.append(
                        {
                            **base,
                            "cells": pair,
                            "group": group,
                            "advantage": mean([-r["fixed_suffix_gap"] for r in records]),
                            "route_log_preference": mean(
                                [r["route_log_preference"] for r in records]
                            ),
                            "states": len(records),
                            "group_exposure": row.get("training_exposure_by_group", {}).get(
                                str(group), {}
                            ),
                        }
                    )
        else:
            cal = result["action_type_calibration"]
            routers.append(
                {
                    **base,
                    "router": summarize_router(result["router"]),
                    "action_pairs": len(cal),
                    "identifiable_pairs": sum(r["identifiable"] for r in cal),
                    "direction_agreement": mean(
                        [r["direction_agreement"] for r in cal if r["identifiable"]]
                    ),
                }
            )
    contrasts = []
    points = sorted({(t["step"], t["phase"]) for t in tasks if t["stage"] == "quality"})
    for step, phase in points:
        semantics = {
            tuple(r["identity"]["event_semantics"])
            for r in found.values()
            if r["identity"]["stage"] == "quality"
            and r["identity"]["step"] == step
            and r["identity"]["phase"] == phase
            and r["identity"]["role"] == "development"
        }
        q = qualities.get((step, phase, "development"), {})
        valid = {a: v for a, v in q.items() if all(p in v for p in ("greedy", "uniform"))}
        # primary() requires all three arms and story-level pairing; no mean-only substitute.
        comparison = primary(c, valid) if len(semantics) <= 1 else {"status": "INCOMPARABLE"}
        contrasts.append({"step": step, "phase": phase, **comparison})
    training_complete = meta["complete"]
    analysis_complete = (
        bool(tasks)
        and not missing
        and all(r["coverage_status"] != "INSUFFICIENT_COVERAGE" for r in found.values())
    )
    status = ("TRAINING_COMPLETE_" if training_complete else "TRAINING_PARTIAL_") + (
        "ANALYSIS_COMPLETE" if analysis_complete else "ANALYSIS_PARTIAL"
    )
    common = {
        "scientific_decision": False,
        "independent_seeds": 1,
        "CAUSAL_FEEDBACK_NOT_TESTED": True,
        "MULTISEED_NOT_TESTED": True,
    }
    return {
        "experiment-manifest.json": {
            **common,
            "status": status,
            "training_status": meta["status"],
            "analysis_status": "COMPLETE" if analysis_complete else "PARTIAL",
            "completed_tasks": len(found),
            "planned_tasks": len(tasks),
            "missing": missing,
        },
        "metrics-summary.json": {**common, "rows": metrics},
        "specialization-summary.json": {
            **common,
            "rows": functionals,
            "advantage_route_trajectories": trajectories,
            "descriptive_lead_lag": descriptive_lag(trajectories),
        },
        "gradient-audit-summary.json": {**common, "rows": gradients},
        "routing-longitudinal-summary.json": {**common, "rows": routers},
        "statistical-contrast.json": {**common, "rows": contrasts},
        "diagnostic-resource-ledger.json": {
            **common,
            "analysis_seconds": sum(r["elapsed_seconds"] for r in found.values()),
            "tasks": [
                {
                    "task_key": canonical(r["identity"]),
                    "stage": r["identity"]["stage"],
                    "role": r["identity"]["role"],
                    "elapsed_seconds": r["elapsed_seconds"],
                    "timings": r["timings"],
                }
                for r in found.values()
            ],
        },
        "analysis-index.json": {**common, "tasks": ledger},
    }
