"""Three-arm partial aggregation, DID primary endpoint and separate H1/H2 decisions."""

from pathlib import Path
import json
from clm001_r2.report import seal, verified, contrast
from .common import ARMS, canonical_sha256, file_sha256

__all__ = ["seal", "verified", "aggregate", "primary"]


def policy_signal(policies):
    g = policies["greedy"]
    u = policies["uniform"]
    if g["groups"] != u["groups"] or len(g["nll"]) != len(u["nll"]):
        raise ValueError("unpaired own-policy data")
    return {
        **g,
        "nll": [a - b for a, b in zip(g["nll"], u["nll"])],
        "cost": [a - b for a, b in zip(g["cost"], u["cost"])],
    }


def primary(c, results, stage):
    needed = ("QC-R3", "Q-R3", "QS-R3")
    if not all(arm in results for arm in needed):
        return {"status": "PARTIAL"}
    policies = {
        arm: (
            results[arm]["pre_admission"]["policies"] if stage == "pre" else results[arm]["quality"]
        )
        for arm in needed
    }
    signal = {arm: policy_signal(p) for arm, p in policies.items()}
    contrasts = {arm: contrast(signal["QC-R3"], signal[arm], c) for arm in ("Q-R3", "QS-R3")}
    utility_supported = all(row["utility"]["ci95"][1] < 0 for row in contrasts.values())
    return {
        "status": "EXPLORATORY_DEV_SIGNAL" if utility_supported else "INCONCLUSIVE",
        "endpoint": "DID_of_greedy_minus_own_uniform_cost_aware_utility",
        "contrasts": contrasts,
        "QC_vs_controls_greedy": {
            arm: contrast(policies["QC-R3"]["greedy"], policies[arm]["greedy"], c)
            for arm in ("Q-R3", "QS-R3")
        },
        "scope": "single_seed_story_cluster_intervals_not_seed_replication",
        "scientific_decision": False,
    }


def aggregate(c, output, source):
    output = Path(output)
    results = {}
    arms = {}
    failures = []
    common = {
        "experiment_id": c["experiment_id"],
        "source_sha": source,
        "config_hash": canonical_sha256(c),
        "scope": "EXPLORATORY_DEVELOPMENT_AGGREGATE",
        "scientific_decision": False,
        "formal_confirmation": False,
    }
    for arm in ARMS:
        folder = output / "arms" / arm
        last = 0
        if (folder / "latest.json").exists():
            latest = json.loads((folder / "latest.json").read_text())
            if (
                Path(latest["file"]).name != latest["file"]
                or file_sha256(folder / latest["file"]) != latest["sha256"]
            ):
                raise ValueError("checkpoint evidence corrupt")
            if latest["identity"]["source_commit"] != source or latest["identity"][
                "config_sha256"
            ] != canonical_sha256(c):
                raise ValueError("checkpoint foreign identity")
            last = int(latest["file"].split("-")[2].split(".")[0])
        if (folder / "metrics.json").exists():
            result = verified(folder / "metrics.json")
            expected = {
                "source_commit": source,
                "config_sha256": canonical_sha256(c),
                "experiment_id": c["experiment_id"],
                "arm": arm,
            }
            if any(result["identity"].get(k) != v for k, v in expected.items()):
                raise ValueError("foreign metrics")
            results[arm] = result
        if (folder / "failure.json").exists():
            failures.append({"arm": arm, **verified(folder / "failure.json")})
        result = results.get(arm, {})
        summary = {
            **common,
            "arm": arm,
            "last_committed_step": last,
            "complete": bool(result),
            "training_status": "COMPLETE"
            if result
            else ("FAILED" if any(row["arm"] == arm for row in failures) else "INCOMPLETE"),
            "quality": {
                policy: {k: v for k, v in value.items() if k not in ("nll", "cost", "groups")}
                for policy, value in result.get("quality", {}).items()
            },
            "shadow_status": result.get("shadow_status", "NOT_RUN"),
        }
        seal(output / ("arm-" + arm + "-metrics-summary.json"), summary)
        arms[arm] = summary
    if (output / "workflow-failure.json").exists():
        failures.append({"arm": "ORCHESTRATOR", **verified(output / "workflow-failure.json")})
    complete = len(results) == len(ARMS)
    status = "COMPLETE" if complete else ("PARTIAL" if results else "NOT_TRAINED")
    pre = primary(c, results, "pre")
    final = primary(c, results, "final")
    qc = results.get("QC-R3", {}).get("pre_admission", {}).get("policies", {}).get("greedy", {})
    failures_of_credit = []
    if qc.get("max_cell_share") == 1.0:
        failures_of_credit.append("ALL_ONE_CELL_COLLAPSE")
    if qc.get("forced_end_rate") == 1.0:
        failures_of_credit.append("STOP_NOT_ACQUIRED")
    if qc.get("context_call_fraction") == 0.0:
        failures_of_credit.append("CTX_NOT_ACQUIRED")
    report = {
        **common,
        "complete": complete,
        "training_status": status,
        "engineering_status": "WORKERS_COMPLETE" if complete else status,
        "arms": arms,
        "credit_failure_flags": failures_of_credit,
        "H1": "DEVELOPMENT_SIGNAL_ONLY"
        if pre.get("status") == "EXPLORATORY_DEV_SIGNAL" and not failures_of_credit
        else "INCONCLUSIVE",
        "H2": "CAUSAL_FEEDBACK_NOT_TESTED",
        "multiseed": "NOT_TESTED",
        "formal_confirmation": False,
        "gpu_pilot": verified(output / "gpu-pilot-summary.json")
        if (output / "gpu-pilot-summary.json").exists()
        else {"status": "NOT_RUN"},
    }
    specialization = {arm: r.get("specialization", {}) for arm, r in results.items()}
    summaries = {
        "combined-report.json": report,
        "pre-admission-contrast.json": pre,
        "final-system-contrast.json": final,
        "resource-ledger.json": {
            "arms": {
                arm: {
                    **r["resource_ledger"],
                    "inference_profile": verified(output / "arms" / arm / "cost-profile.json")
                    if (output / "arms" / arm / "cost-profile.json").exists()
                    else {"status": "NOT_PROFILED"},
                }
                for arm, r in results.items()
            },
            "compute_matched_training_extension": "NOT_RUN",
            "equal_step_same_token_comparison": True,
        },
        "q-calibration-summary.json": {
            "arms": {
                arm: {k: v for k, v in r.get("q_calibration", {}).items() if k != "records"}
                for arm, r in results.items()
            }
        },
        "competition-gradient-audit.json": {
            "arms": {arm: r.get("competition_gradient_audit", {}) for arm, r in results.items()}
        },
        "specialization-summary.json": {"arms": specialization},
        "feedback-longitudinal-summary.json": {
            "arms": specialization,
            "causal_feedback": "CAUSAL_FEEDBACK_NOT_TESTED",
        },
        "stop-ctx-path-summary.json": {
            "arms": {arm: value["quality"] for arm, value in arms.items()}
        },
        "shadow-candidate-summary.json": {
            "arms": {arm: r.get("shadow_summary", []) for arm, r in results.items()}
        },
        "failure-index.json": {"failures": failures},
    }
    for name, value in summaries.items():
        seal(output / name, {**common, "complete": complete, **value})
    return report
