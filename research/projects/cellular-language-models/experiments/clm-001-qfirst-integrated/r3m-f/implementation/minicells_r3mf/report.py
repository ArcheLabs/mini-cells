"""Complete-matrix descriptive reporting; capability and causal sensitivity stay distinct."""

import math
from pathlib import Path
import numpy as np
from .protocol import atomic, digest, read

ARTIFACTS = (
    "EXPERIMENT_MANIFEST.json",
    "MODEL_LINEAGE.json",
    "DATA_AUDIT.json",
    "TINY_OVERFIT_REPORT.json",
    "COPY_CAPABILITY.json",
    "ARITHMETIC_CAPABILITY.json",
    "TRAINING_CURVES.json",
    "STORY_RETENTION.json",
    "CELL_EXPOSURE.json",
    "GRADIENT_CONFLICT.json",
    "CELL_INTERVENTION.json",
    "SPECIALIZATION_TRAJECTORY.json",
    "LEARNABILITY_DECISION.json",
    "COMPUTE_COST.json",
    "RUN_MANIFEST.json",
    "SCIENTIFIC-FINDINGS.md",
)


def tiny_decision(result, c):
    initial = result["evaluations"]["0"]["metrics"]["tiny"]
    final = result["evaluations"][str(c["continuation_steps"])]["metrics"]["tiny"]
    grads = [g for r in result["curve"] for g in r["cell_gradient_norms"]]
    valid = bool(
        grads
        and all(math.isfinite(g) for g in grads)
        and any(g > 0 for g in grads)
        and result["parameter_updated"]
        and result["resume"]["exact_full_state"]
    )
    passed = bool(valid and final["exact_match"] >= c["tiny"]["minimum_exact_match"])
    return dict(
        status="TINY_OVERFIT_PASS_REVIEW_REQUIRED" if passed else "TINY_OVERFIT_FAILED",
        passed=passed,
        facts=final["facts"],
        threshold=c["tiny"]["minimum_exact_match"],
        initial=initial,
        final=final,
        gradient_and_parameter_checks=valid,
        restore=result["resume"],
        optimizer_updates=result["main_updates"],
        pilot_authorized=False,
        scientific_generalization_claim=False,
    )


def _primary(row, cell=1):
    return next(
        x
        for x in row["interventions"]["rows"]
        if x["cell"] == cell and x["control"] == "USAGE_MATCHED"
    )


def pilot_decision(c, results, baseline):
    if len(results) != 9:
        return dict(status="INCOMPLETE_NO_SCIENTIFIC_DECISION", scientific_decision=False)
    lookup = {(r["identity"]["seed"], r["identity"]["arm"]): r for r in results}
    terminal = str(c["continuation_steps"])
    directions = []
    copy_good = []
    special = []
    trajectories = []
    for seed in c["seeds"]:
        r = lookup[seed, "F2-A1"]
        start = r["evaluations"]["0"]["metrics"]
        end = r["evaluations"][terminal]["metrics"]
        dev = end["a1"]
        initial = start["a1"]
        held = dev["templates"][str(c["arithmetic"]["heldout_template"])]["exact_match"]
        learned = bool(
            dev["exact_match"] >= c["decision"]["a1_unseen_em_min"]
            and dev["exact_match"] > baseline
            and end["a1_train"]["exact_match"] > start["a1_train"]["exact_match"]
            and dev["answer_nll"] < initial["answer_nll"]
            and dev["exact_match"] > initial["exact_match"]
            and held > baseline
        )
        directions.append(
            dict(
                seed=seed,
                learned=learned,
                unseen_em=dev["exact_match"],
                baseline=baseline,
                heldout_template_em=held,
            )
        )
        cp = lookup[seed, "F1-COPY"]["evaluations"]
        copy_good.append(
            cp[terminal]["metrics"]["copy"]["exact_match"] >= c["decision"]["copy_unseen_em_min"]
            and cp[terminal]["metrics"]["copy"]["exact_match"]
            > cp["0"]["metrics"]["copy"]["exact_match"]
        )
        change = {}
        for arm in c["arms"]:
            ev = lookup[seed, arm]["evaluations"]
            change[arm] = _primary(ev[terminal])["specificity"] - _primary(ev["0"])["specificity"]
            for cell in range(6):
                values = [_primary(ev[str(t)], cell)["specificity"] for t in c["barriers"]]
                em = [ev[str(t)]["metrics"]["a1"]["exact_match"] for t in c["barriers"]]
                corr = (
                    float(np.corrcoef(values, em)[0, 1])
                    if np.std(values) > 0 and np.std(em) > 0
                    else None
                )
                trajectories.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        cell=cell,
                        times=c["barriers"],
                        specificity=values,
                        a1_em=em,
                        pearson=corr,
                        interpretation="ASSOCIATION_NOT_DIRECTIONAL_CAUSATION",
                    )
                )
        delta = {a: change["F2-A1"] - change[a] for a in ("F0-STORY", "F1-COPY")}
        positive = (
            all(v >= c["decision"]["specificity_double_difference_min"] for v in delta.values())
            and _primary(r["evaluations"][terminal])["raw_delta"]["a1"] > 0
        )
        special.append(dict(seed=seed, cell=1, contrasts=delta, positive=positive))
    arithmetic = sum(x["learned"] for x in directions) >= 2
    specific = sum(x["positive"] for x in special) >= 2
    copy_pass = sum(copy_good) >= 2
    if arithmetic:
        status = (
            "FUNCTIONAL_SPECIALIZATION_SIGNAL"
            if specific
            else "LEARNABLE_WITHOUT_DETECTED_SPECIALIZATION"
        )
    elif specific:
        status = "PARTIAL_SPECIALIZATION_WITHOUT_COMPETENCE"
    else:
        status = (
            "COPY_LEARNABLE_ARITHMETIC_NOT_ESTABLISHED"
            if copy_pass
            else "BASIC_LEARNABILITY_BOTTLENECK"
        )
    return dict(
        status=status,
        arithmetic_status="ARITHMETIC_LEARNABILITY_OBSERVED"
        if arithmetic
        else "ARITHMETIC_LEARNING_NOT_ESTABLISHED",
        copy_learnable=copy_pass,
        per_seed=directions,
        specialization=special,
        trajectories=trajectories,
        scientific_decision=False,
        scope="EXPLORATORY_SIGNAL_NOT_CONFIRMATION",
        a2_a3_authorized=False,
        connection_reorganization_supported=False,
    )


def write(c, output, ident, manifest, failures=()):
    output = Path(output)
    dest = output / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    roots = output / "private-checkpoints" / ident["run_id"] / "jobs"
    results = []
    missing = []
    for seed in c["seeds"]:
        for arm in c["arms"]:
            path = roots / str(seed) / arm / "EVALUATION.json"
            if not path.exists():
                missing.append([seed, arm])
                continue
            r = read(path)
            if r["identity"] != dict(ident, seed=seed, arm=arm):
                raise ValueError("REPORT_IDENTITY_MISMATCH")
            results.append(r)
    tiny = c["execution_phase"] == "tiny"
    decision = (
        tiny_decision(results[0], c)
        if tiny and results and not failures and not missing
        else dict(status="PARTIAL" if missing or failures else "NOT_RUN", pilot_authorized=False)
    )
    if not tiny:
        decision = pilot_decision(
            c, results, manifest["task_audit"]["frequency_baseline"]["dev_fact_accuracy"]
        )
    common = dict(
        identity=ident,
        confirmation="SEALED",
        r3m_registration="BLOCKED",
        scope="EXPLORATORY_DEVELOPMENT",
    )
    payload = {
        name: dict(common, status="NOT_RUN_TINY_ONLY" if tiny else "RECORDED")
        for name in ARTIFACTS
        if name.endswith(".json")
    }
    payload["EXPERIMENT_MANIFEST.json"].update(
        config=c,
        phase=c["execution_phase"],
        expected_jobs=len(c["seeds"]) * len(c["arms"]),
        completed_jobs=len(results),
        missing=missing,
        failures=list(failures),
    )
    payload["MODEL_LINEAGE.json"].update(
        parent=c["baseline"],
        architecture=c["model"],
        connection_updates=0,
        embedding_head="ORIGINAL_FROZEN_RULES",
        parameters_added=0,
        anchors=[
            dict(seed=seed, arm=arm, evidence=read(roots / str(seed) / arm / "ANCHOR.json"))
            for seed in c["seeds"]
            for arm in c["arms"]
            if (roots / str(seed) / arm / "ANCHOR.json").is_file()
        ],
    )
    payload["DATA_AUDIT.json"].update(manifest=manifest)
    payload["TINY_OVERFIT_REPORT.json"].update(
        decision if tiny else dict(status="PRIOR_REVIEW_REQUIRED")
    )
    payload["LEARNABILITY_DECISION.json"].update(decision)

    def key(r):
        return str(r["identity"]["seed"]) + "/" + r["identity"]["arm"]

    payload["TRAINING_CURVES.json"]["jobs"] = {key(r): r["curve"] for r in results}
    for name, field in (("COPY_CAPABILITY.json", "copy"), ("ARITHMETIC_CAPABILITY.json", "a1")):
        payload[name]["jobs"] = {
            key(r): {
                t: dict(dev=e["metrics"][field], train=e["metrics"].get("a1_train"))
                for t, e in r["evaluations"].items()
            }
            for r in results
            if not tiny
        }
    for name, field in (("STORY_RETENTION.json", "story_nll"), ("CELL_EXPOSURE.json", "exposure")):
        payload[name]["jobs"] = {
            key(r): {t: e["metrics"].get(field) for t, e in r["evaluations"].items()}
            for r in results
        }
    for name, field in (
        ("GRADIENT_CONFLICT.json", "gradient"),
        ("CELL_INTERVENTION.json", "interventions"),
    ):
        payload[name]["jobs"] = {
            key(r): {t: e.get(field) for t, e in r["evaluations"].items()}
            for r in results
            if not tiny
        }
    if not tiny:
        payload["STORY_RETENTION.json"]["jobs"] = {
            key(r): {
                t: dict(
                    nll=e["metrics"]["story_nll"],
                    delta_from_anchor=e["metrics"]["story_nll"]
                    - r["evaluations"]["0"]["metrics"]["story_nll"],
                )
                for t, e in r["evaluations"].items()
            }
            for r in results
        }
    payload["SPECIALIZATION_TRAJECTORY.json"].update(
        trajectories=decision.get("trajectories", []),
        cell_id_indexing="ZERO_BASED",
        cell_1_selected_from_prior_evidence=True,
    )
    payload["COMPUTE_COST.json"].update(
        jobs={
            key(r): {
                k: r[k]
                for k in (
                    "continuation_cost",
                    "parameter_counts",
                    "observed_parameter_work",
                    "main_updates",
                    "tokens",
                    "cell_calls",
                )
            }
            for r in results
        },
        analysis_seconds=sum(
            e["analysis_seconds"] for r in results for e in r["evaluations"].values()
        ),
        full_pilot_updates=8100,
        full_pilot_tokens=4147200,
        tiny_excluded_from_pilot=True,
        observed_gpu_latency="NOT_INFERRED_FROM_CPU",
        hard_runtime_termination=False,
    )
    for name, value in payload.items():
        if name != "RUN_MANIFEST.json":
            atomic(dest / name, value)
    findings = (
        "# R3m-F findings\n\nStatus: " + decision["status"] + ".\n\n"
        "Tiny Overfit is training-fact memorization, not unseen-fact competence. Pilot is not automatically authorized by a pass.\n\n"
        "Complete capability, all six Cells, all seeds and time points must be reviewed together. Replacement harm is local contribution; routing frequency and gradient differences alone are not specialization.\n\n"
        "A2/A3 and Connection reorganization remain unauthorized. Original R3m registration stays BLOCKED.\n"
    )
    if not tiny:
        for item in decision.get("per_seed", []):
            findings += (
                "\nSeed "
                + str(item["seed"])
                + ": unseen fact EM "
                + str(item["unseen_em"])
                + ", frequency baseline "
                + str(item["baseline"])
                + ", held-out-template EM "
                + str(item["heldout_template_em"])
                + ".\n"
            )
    if not tiny:
        findings += "\nDecision details are in LEARNABILITY_DECISION.json; all predictions, controls and fact denominators accompany it.\n"
    questions = [
        (
            "Eight-fact memorization",
            decision["status"] if tiny else "Separate reviewed Tiny evidence required",
        ),
        ("Unseen A1 competence", decision.get("arithmetic_status", "NOT_RUN_TINY_ONLY")),
        ("Copy competence", str(decision.get("copy_learnable", "NOT_RUN_TINY_ONLY"))),
        (
            "Beyond numeric-output learning",
            "Review A1 unseen/held-out template performance with F1 numeric-control mismatch disclosed"
            if not tiny
            else "NOT_RUN_TINY_ONLY",
        ),
        (
            "Intervenable Cell specialization",
            decision["status"] if not tiny else "NOT_RUN_TINY_ONLY",
        ),
        (
            "Prior Cell 1 sensitivity replication",
            str(decision.get("specialization", "NOT_RUN_TINY_ONLY")),
        ),
        (
            "Capability-specialization co-evolution",
            "All five timepoints/all Cells; descriptive correlations only"
            if not tiny
            else "NOT_RUN_TINY_ONLY",
        ),
        ("Proceed to A2/A3", "Separate review and frozen protocol required; not authorized"),
        ("Connection reorganization value", "Not tested; no support claim"),
    ]
    findings += (
        "\n## Required questions\n\n"
        + "\n".join(
            str(i + 1) + ". " + q + ": " + answer for i, (q, answer) in enumerate(questions)
        )
        + "\n"
    )
    (dest / "SCIENTIFIC-FINDINGS.md").write_text(findings)
    run = dict(
        common,
        status="PARTIAL" if failures or missing else decision["status"],
        trial_status=decision["status"],
        run_id=ident["run_id"],
        source_sha=ident["source_sha"],
        config_sha256=ident["config_sha256"],
        seeds=c["seeds"],
        exit_code=1 if failures or missing else 0,
        publication_status="LOCAL_RETAINED",
        remote_commit_sha=None,
        artifact_files={
            p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in dest.iterdir()
            if p.is_file() and p.name != "RUN_MANIFEST.json"
        },
        scientific_decision=False,
        full_pilot_authorized=False,
    )
    atomic(dest / "RUN_MANIFEST.json", run)
    return run
