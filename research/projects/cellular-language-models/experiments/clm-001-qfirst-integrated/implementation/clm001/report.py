"""Matched development-only decision tree; confidence intervals and zero-result outcomes."""

import numpy as np
from pd001.analysis import paired_interval
from .common import canonical_sha256, source_sha

FIELDS = (
    "params_resident",
    "params_active",
    "train_tokens",
    "dev_nll_greedy",
    "dev_nll_sampled",
    "q_rank_spearman",
    "q_regret",
    "stop_accuracy_vs_oracle",
    "actual_stop_rate",
    "forced_end_rate",
    "mean_executed_actions",
    "shadow_proposed",
    "shadow_promoted",
    "shadow_rolled_back",
    "training_wall_seconds",
    "exploration_wall_seconds",
)


def summarize(c, results, *, oracle_pass=False, shadow_journal=None):
    complete = all(arm in results for arm in c["arms"])
    status = "INCOMPLETE"
    contrasts = {}
    checks = {}
    flat = {}
    for arm in c["arms"]:
        item = results.get(arm, {})
        row = {key: None for key in FIELDS}
        quality = item.get("quality", {})
        q = item.get("q_calibration", {}) or {}
        if item:
            inv = item.get("parameters", {})
            greedy = quality.get("greedy", {})
            sampled = quality.get("sampled", {})
            row.update(
                params_resident=inv.get("total_parameters"),
                params_active=item.get("cost_profile", {}).get("unique_touched_parameters_window"),
                train_tokens=item.get("live_tokens"),
                dev_nll_greedy=greedy.get("mean_nll"),
                dev_nll_sampled=sampled.get("mean_nll"),
                q_rank_spearman=q.get("spearman"),
                q_regret=q.get("regret"),
                stop_accuracy_vs_oracle=1 - q["stop_error_rate"]
                if q.get("stop_error_rate") is not None
                else None,
                actual_stop_rate=greedy.get("actual_stop_rate"),
                forced_end_rate=greedy.get("forced_end_rate"),
                mean_executed_actions=float(np.mean(greedy["cost"])) * c["model"]["budget"]
                if greedy.get("cost")
                else None,
                training_wall_seconds=item.get("timings", {}).get("elapsed_seconds"),
                exploration_wall_seconds=item.get("exploration_wall_seconds"),
            )
            counts = item.get("shadow_counts", {})
            row.update(
                {
                    k: counts.get(k)
                    for k in ("shadow_proposed", "shadow_promoted", "shadow_rolled_back")
                }
            )
        flat[arm] = row
    if not oracle_pass:
        status = "ORACLE_UNTRUSTED"
    elif complete:
        q = results["QF-FULL"].get("q_calibration", {})
        records = q.get("records", [])
        if not q.get("identifiable_states", 0):
            status = "Q_UNIDENTIFIABLE"
        elif q.get("spearman") is None:
            status = "Q_GENERALIZATION_FAILURE"
        else:
            status = "DEVELOPMENT_COMPLETE_UNREVIEWED"
        frozen = (
            c["gates"]["thresholds_frozen"]
            and c["gates"]["gpu_pilot_pass"]
            and c["gates"]["portable_pass"]
            and c["gates"]["development_approved"]
            and not c.get("synthetic")
            and all(v is not None for v in c["thresholds"].values())
            and c["policy"]["lambda_cost"] is not None
        )
        qa = results["QF-FULL"].get("quality", {})
        ja = results["J0-FULL"].get("quality", {})
        if qa and ja and c["policy"]["lambda_cost"] is not None:
            baseline = ja["greedy"]
            candidate = qa["greedy"]
            lam = c["policy"]["lambda_cost"]
            if baseline["paired_identity"] != candidate["paired_identity"]:
                raise ValueError("internal contrast is not matched")

            def utility(x):
                return np.asarray(x["nll"]) + lam * np.asarray(x["cost"])

            for label, b in [
                ("J0-FULL", baseline),
                ("self_uniform", qa["uniform"]),
                ("self_constant", qa["constant"]),
            ]:
                contrasts[label] = paired_interval(
                    utility(b),
                    utility(candidate),
                    candidate["groups"],
                    seed=c["evaluation"]["seed"],
                    replicates=c["evaluation"]["bootstrap_replicates"],
                )
            valid = [r for r in records if r["spearman"] is not None]
            if valid:
                rank = paired_interval(
                    [0.0] * len(valid),
                    [r["spearman"] for r in valid],
                    list(range(len(valid))),
                    seed=c["evaluation"]["seed"],
                    replicates=c["evaluation"]["bootstrap_replicates"],
                )
                regret = paired_interval(
                    [r["regret"] for r in valid],
                    [r["uniform_regret"] for r in valid],
                    list(range(len(valid))),
                    seed=c["evaluation"]["seed"],
                    replicates=c["evaluation"]["bootstrap_replicates"],
                )
                contrasts["q_rank"] = rank
                contrasts["regret_improvement"] = regret
                if frozen:
                    t = c["thresholds"]
                    checks = {
                        "rank": rank["ci95_two_sided"][0] >= t["rank_spearman_lower95"],
                        "regret": regret["ci95_two_sided"][0] >= t["regret_improvement_lower95"],
                        "internal_utility": all(
                            contrasts[k]["ci95_two_sided"][1] <= t["internal_utility_delta_upper95"]
                            for k in ("J0-FULL", "self_uniform", "self_constant")
                        ),
                        "stop": q["stop_error_rate"] <= t["stop_error_rate_max"],
                    }
                    if not checks["stop"]:
                        status = "STOP_COLLAPSE"
                    elif not checks["rank"] or not checks["regret"]:
                        status = "Q_GENERALIZATION_FAILURE"
                    elif not checks["internal_utility"]:
                        status = "Q_SIGNAL_NO_POLICY_GAIN"
                    elif all(checks.values()):
                        status = "INTEGRATED_DEV_SIGNAL"
    journals = shadow_journal or []
    if (
        status == "DEVELOPMENT_COMPLETE_UNREVIEWED"
        and complete
        and all(flat[a]["shadow_promoted"] == 0 for a in c["arms"])
    ):
        status = "NO_SHADOW_PROMOTION"
    identities = [v.get("identity", {}) for v in results.values()]
    run_id = identities[0] if identities else {}
    return {
        "experiment_id": c["experiment_id"],
        "claim_scope": "EXPLORATORY_ONLY",
        "status": status,
        "source_sha": run_id.get("source_commit", source_sha()),
        "config_sha256": canonical_sha256(c),
        "asset_manifest_sha256": run_id.get("assets_manifest_sha256"),
        "tokenizer_sha256": run_id.get("tokenizer_sha256"),
        "seed": c["seed"],
        "arms": flat,
        "arm_details": results,
        "paired_internal_contrasts": contrasts,
        "numerical_checks": checks,
        "oracle_gate": {"cpu_pass": oracle_pass, "gpu_pilot_pass": c["gates"]["gpu_pilot_pass"]},
        "action_conditional_utility": {
            a: results.get(a, {}).get("q_calibration") for a in c["arms"]
        },
        "stop_nll_compute_pareto": {a: results.get(a, {}).get("quality") for a in c["arms"]},
        "shadow_journal_sha256": canonical_sha256(journals),
        "scientific_decision": False,
        "formal_confirmation": False,
        "notes": [
            "No untouched confirmation was read",
            "No specialization claim follows from Cell IDs or entropy",
            "INTEGRATED_DEV_SIGNAL, if present, is exploratory and not confirmation",
        ],
    }
