"""Single-seed paired story inference and descriptive longitudinal mechanisms."""

from pathlib import Path
import numpy as np
from minicells_runtime.store import canonical, read, digest, ContinuationError
from clm001_r3.report import verified, policy_signal
from clm001_r2.report import contrast
from .common import ARMS


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize_router(rows):
    result = {
        "states": len(rows),
        "stop_legal_states": sum(r["stop_legal"] for r in rows),
        "stop_minus_best_cell": mean([r["stop_minus_best_cell"] for r in rows]),
        "ctx_minus_best_cell": mean([r["ctx_minus_best_cell"] for r in rows]),
        "cell_entropy": mean([r["cell_entropy"] for r in rows]),
        "near_tie_fraction": mean([r["near_tie"] for r in rows]),
    }
    result["by_depth_and_context"] = []
    for depth, ctx in sorted({(r["depth"], r["ctx_seen"]) for r in rows}):
        chosen = [r for r in rows if (r["depth"], r["ctx_seen"]) == (depth, ctx)]
        result["by_depth_and_context"].append(
            {
                "depth": depth,
                "ctx_seen": ctx,
                "states": len(chosen),
                "stop_margin": mean([r["stop_minus_best_cell"] for r in chosen]),
                "ctx_margin": mean([r["ctx_minus_best_cell"] for r in chosen]),
            }
        )
    return result


def primary(c, policies):
    if not all(a in policies for a in ARMS):
        return {"status": "PARTIAL", "independent_seeds": 1, "scientific_decision": False}
    signal = {arm: policy_signal(value) for arm, value in policies.items()}
    did = contrast(signal["QC-R3b"], signal["Q-R3b"], c)
    return {
        "status": "EXPLORATORY_DEV_SIGNAL" if did["utility"]["ci95"][1] < 0 else "INCONCLUSIVE",
        "primary": "(QC_greedy-QC_own_uniform)-(Q_greedy-Q_own_uniform)",
        "DID": did,
        "QC_vs_Q_greedy": contrast(policies["QC-R3b"]["greedy"], policies["Q-R3b"]["greedy"], c),
        "QC_vs_own_uniform": contrast(
            policies["QC-R3b"]["greedy"], policies["QC-R3b"]["uniform"], c
        ),
        "QC_vs_QS_negative_control": contrast(signal["QC-R3b"], signal["QS-R3b"], c),
        "independent_seeds": 1,
        "Q_QS_are_not_independent_replications": True,
        "scientific_decision": False,
        "scope": "SINGLE_SEED_STORY_CLUSTER_BOOTSTRAP_NOT_CONFIRMATION",
    }


def aggregate(root):
    root = Path(root)
    c = read(root / "runtime-config.json")
    meta = read(root / "continuation.json")
    covered = {
        a: {r["checkpoint_sha256"] for r in p["checkpoints"]} for a, p in meta["arms"].items()
    }
    rows, snapshots = [], {}
    missing = []
    for arm in ARMS:
        found = []
        for path in sorted((root / "arms" / arm / "diagnostics").glob("snapshot-*.json")):
            if path.name.endswith(".sha256.json"):
                continue
            value = verified(path)
            if (
                value["checkpoint_sha256"] not in covered.get(arm, set())
                or value["runtime_config_sha256"] != meta["runtime_config_sha256"]
                or value["evaluation_protocol_sha256"] != canonical(c["r3b"]["diagnostics"])
            ):
                raise ContinuationError("SNAPSHOT_IDENTITY_MISMATCH")
            if (
                value.get("identity") != read(root / "arms" / arm / "identity.json")
                or value["probe_sha256"] != meta["arms"][arm]["probe_bank_sha256"]
                or value["development_probe_sha256"]
                != digest(root / "arms" / arm / "development-probe-bank.pt")
            ):
                raise ContinuationError("SNAPSHOT_PROBE_OR_ARM_IDENTITY_MISMATCH")
            if value["quality_role"] != "development":
                raise ContinuationError("PILOT_IS_NOT_RESEARCH_DIAGNOSTICS")
            found.append(value)
        snapshots[arm] = found
        for step in c["r3b"]["diagnostics"]["steps"]:
            if not any(s["step"] == step and s["phase"] == "post_event" for s in found):
                missing.append(
                    {"arm": arm, "step": step, "phase": "post_event", "reason": "NOT_RECORDED"}
                )
        for value in found:
            quality = {
                p: {k: v for k, v in q.items() if k not in ("nll", "cost", "groups")}
                for p, q in value["quality"].items()
            }
            calibration = value["action_type_calibration"]
            rows.append(
                {
                    "arm": arm,
                    "step": value["step"],
                    "phase": value["phase"],
                    "event_revision": value["event_revision"],
                    "checkpoint_sha256": value["checkpoint_sha256"],
                    "probe_sha256": value["probe_sha256"],
                    "quality": quality,
                    "router": summarize_router(value["router"]),
                    "action_type_agreement": {
                        t: {
                            "pairs": sum(r["type"] == t for r in calibration),
                            "identifiable_fraction": mean(
                                [r["identifiable"] for r in calibration if r["type"] == t]
                            ),
                            "direction_agreement": mean(
                                [
                                    r["direction_agreement"]
                                    for r in calibration
                                    if r["type"] == t and r["identifiable"]
                                ]
                            ),
                        }
                        for t in ("STOP", "CTX")
                    },
                    "gradient_audit": value["gradient_audit"],
                    "diagnostic_seconds": value["elapsed_seconds"],
                }
            )
    contrasts = []
    for step in c["r3b"]["diagnostics"]["steps"]:
        for phase in ("post_event", "pre_event"):
            chosen = {
                arm: max(
                    (s for s in values if s["step"] == step and s["phase"] == phase),
                    key=lambda r: r["event_revision"],
                    default=None,
                )
                for arm, values in snapshots.items()
            }
            contrasts.append(
                {
                    "step": step,
                    "phase": phase,
                    **primary(c, {a: v["quality"] for a, v in chosen.items() if v}),
                }
            )
    functional, trajectories = [], []
    for arm, values in snapshots.items():
        for snap in values:
            for role, measured in snap["functional"].items():
                functional.append(
                    {
                        "arm": arm,
                        "step": snap["step"],
                        "phase": snap["phase"],
                        "role": role,
                        "summary": measured["summary"],
                        "crossover": measured["crossover"],
                        "router_oracle_agreement": mean(
                            [r["router_choice_agreement"] for r in measured["state_records"]]
                        ),
                        "paired_regret": mean(
                            [r["paired_regret"] for r in measured["state_records"]]
                        ),
                        "rank_correlation": mean(
                            [r.get("rank_correlation") for r in measured["state_records"]]
                        ),
                        "scored_states": len(measured["state_records"]),
                    }
                )
                for pair in c["alignment"]["cell_pairs"]:
                    for group in (0, 1):
                        selected = [
                            r
                            for r in measured["records"]
                            if (r["cell_i"], r["cell_j"], r["group_feature"]) == (*pair, group)
                        ]
                        trajectories.append(
                            {
                                "arm": arm,
                                "role": role,
                                "step": snap["step"],
                                "phase": snap["phase"],
                                "cells": pair,
                                "group": group,
                                "advantage": mean([-r["fixed_suffix_gap"] for r in selected]),
                                "route_log_preference": mean(
                                    [r["route_log_preference"] for r in selected]
                                ),
                                "states": len(selected),
                                "group_exposure": snap.get("training_exposure_by_group", {}).get(
                                    str(group), {}
                                ),
                            }
                        )
    lagged = []
    keys = {(r["arm"], r["role"], tuple(r["cells"]), r["group"]) for r in trajectories}
    for arm, role, pair, group in sorted(keys):
        series = sorted(
            (
                r
                for r in trajectories
                if (r["arm"], r["role"], tuple(r["cells"]), r["group"]) == (arm, role, pair, group)
                and r["phase"] == "post_event"
            ),
            key=lambda r: r["step"],
        )
        for before, after in zip(series, series[1:]):
            if None in (
                before["advantage"],
                after["advantage"],
                before["route_log_preference"],
                after["route_log_preference"],
            ):
                continue
            lagged.append(
                {
                    "arm": arm,
                    "role": role,
                    "cells": list(pair),
                    "group": group,
                    "from_step": before["step"],
                    "to_step": after["step"],
                    "preceding_advantage": before["advantage"],
                    "following_route_change": after["route_log_preference"]
                    - before["route_log_preference"],
                    "preceding_exposure": before["group_exposure"],
                    "following_advantage_change": after["advantage"] - before["advantage"],
                    "causal_interpretation": "NOT_TESTED",
                }
            )
    shared = {
        "experiment_id": meta["experiment_id"],
        "run_id": meta["run_id"],
        "source_sha": meta["source_sha"],
        "scientific_decision": False,
        "formal_confirmation": False,
        "independent_seeds": 1,
        "CAUSAL_FEEDBACK": "CAUSAL_FEEDBACK_NOT_TESTED",
        "MULTISEED": "MULTISEED_NOT_TESTED",
    }
    complete = not missing
    evidence = root.parent / (root.name + "-evidence")
    store = (
        read(evidence / "storage-status.json")
        if (evidence / "storage-status.json").exists()
        else {"status": "NOT_ATTEMPTED"}
    )
    latest = [
        v for v in contrasts if v["step"] == c["training"]["steps"] and v["phase"] == "post_event"
    ][0]
    return {
        "experiment-manifest.json": {
            **shared,
            "arm_progress": {a: p["step"] for a, p in meta["arms"].items()},
            "complete": meta["complete"],
            "diagnostic_status": "DIAGNOSTIC_COMPLETE" if complete else "INCOMPLETE",
            "missing": missing,
            "storage_status": store["status"],
            "H1": "H1_DEV_SIGNAL"
            if any(
                r["arm"] == "QC-R3b"
                and r["role"] == "calibration"
                and r["summary"]["status"] == "LOCAL_ALIGNMENT_PASS"
                for r in functional
            )
            else "INCONCLUSIVE",
            "system_endpoint": latest["status"],
            "H2": "EXPLORATORY_ONLY",
        },
        "metrics-summary.json": {
            **shared,
            "rows": [
                {k: r[k] for k in ("arm", "step", "phase", "event_revision", "quality")}
                for r in rows
            ],
        },
        "routing-longitudinal-summary.json": {
            **shared,
            "rows": [
                {k: r[k] for k in ("arm", "step", "phase", "router", "action_type_agreement")}
                for r in rows
            ],
        },
        "specialization-summary.json": {
            **shared,
            "functional": functional,
            "advantage_route_trajectories": trajectories,
            "descriptive_lead_lag": lagged,
            "scope": "CONDITIONAL_CROSSOVER_CANDIDATES_NOT_CAUSAL_SPECIALIZATION",
        },
        "gradient-audit-summary.json": {
            **shared,
            "rows": [{k: r[k] for k in ("arm", "step", "phase", "gradient_audit")} for r in rows],
        },
        "statistical-contrast.json": {
            **shared,
            "contrasts": contrasts,
            "time_series": "EXPLORATORY_REPEATED_LOOKS_NOT_MULTIPLE_CONFIRMATORY_TESTS",
        },
        "resource-ledger.json": {
            **shared,
            "snapshot_seconds": sum(r["diagnostic_seconds"] for r in rows),
            "training": {
                a: read(
                    root / "arms" / a / "worker-progress" / (Path(p["checkpoint"]).name + ".json")
                )
                if (
                    root / "arms" / a / "worker-progress" / (Path(p["checkpoint"]).name + ".json")
                ).exists()
                else {"status": "NOT_AVAILABLE"}
                for a, p in meta["arms"].items()
            },
            "storage": read(evidence / "resource-ledger.json")
            if (evidence / "resource-ledger.json").exists()
            else {"status": "NOT_MEASURED"},
        },
    }
