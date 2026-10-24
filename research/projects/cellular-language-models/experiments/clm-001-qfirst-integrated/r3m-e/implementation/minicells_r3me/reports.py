"""Complete-matrix, exploratory-only reports; missing jobs remain inconclusive."""

from pathlib import Path

from .common import atomic, digest, read

ARTIFACTS = (
    "EXPERIMENT_MANIFEST.json",
    "INPUT_IDENTITY.json",
    "MODEL_LINEAGE.json",
    "ARITHMETIC_DATA_AUDIT.json",
    "ARM_MATRIX.json",
    "TRAINING_CURVES.json",
    "STORY_RETENTION.json",
    "ARITHMETIC_CAPABILITY.json",
    "CELL_EXPOSURE.json",
    "GRADIENT_CONFLICT.json",
    "FUNCTIONAL_DIFFERENTIATION.json",
    "CONNECTION_INTERVENTION.json",
    "FOUR_GRID_ATTRIBUTION.json",
    "LONG_HORIZON_VALUE.json",
    "COMPUTE_COST.json",
    "SCIENTIFIC-FINDINGS.md",
    "RUN_MANIFEST.json",
)


def write(c, output, ident, input_identity, failures=()):
    output = Path(output)
    dest = output / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in c["seeds"]:
        for arm in c["arms"]:
            path = (
                output
                / "private-checkpoints"
                / ident["run_id"]
                / "jobs"
                / str(seed)
                / arm
                / "EVALUATION.json"
            )
            if path.exists():
                row = read(path)
                if row.get("seed") != seed or row.get("arm") != arm:
                    raise RuntimeError("R3ME_EVALUATION_IDENTITY_MISMATCH")
                jobs.append(row)
    complete = len(jobs) == len(c["seeds"]) * len(c["arms"]) and not failures
    by = {(r["seed"], r["arm"]): r for r in jobs}
    comparisons = []
    for seed in c["seeds"]:
        stable = by.get((seed, "E1-MIX-STABLE"), {}).get("evaluations", {})
        reorg = by.get((seed, "E2-MIX-REORG"), {}).get("evaluations", {})
        sham = by.get((seed, "E3-MIX-SHAM"), {}).get("evaluations", {})
        for barrier in (0, 100, 110, 200, 400, 600):
            key = str(barrier)
            if key not in stable or key not in reorg:
                continue
            s, r = stable[key], reorg[key]
            sham_row = sham.get(key)
            comparisons.append(
                {
                    "seed": seed,
                    "relative_step": barrier,
                    "global_step": c["anchor_step"] + barrier,
                    "arithmetic_answer_gain": s["answer_nll"] - r["answer_nll"],
                    "story_harm": r["story_nll"] - s["story_nll"],
                    "exact_match_change": r["arithmetic_exact_match"] - s["arithmetic_exact_match"],
                    "sham_answer_nll": sham_row[key]["answer_nll"] if sham_row else None,
                }
            )
    final_rows = [r for r in comparisons if r["relative_step"] == 600]
    gains = [r["arithmetic_answer_gain"] for r in final_rows]
    middle_rows = [r for r in comparisons if r["relative_step"] == 400]
    middle_gains = [r["arithmetic_answer_gain"] for r in middle_rows]
    transactions = [
        {"seed": r["seed"], "arm": r["arm"], "transactions": r.get("transactions", [])}
        for r in jobs
    ]
    evals = [
        {"seed": r["seed"], "arm": r["arm"], "evaluations": r.get("evaluations", {})}
        for r in jobs
    ]
    cost = [
        {"seed": r["seed"], "arm": r["arm"], "cost": r.get("cost", {})}
        for r in jobs
    ]
    intervention_ok = any(
        tx.get("accepted") and tx.get("trace", {}).get("effective")
        for r in jobs
        for tx in r.get("transactions", [])
    )
    learnability = []
    for seed in c["seeds"]:
        rows_for_seed = [by.get((seed, arm), {}).get("evaluations", {}) for arm in ("E1-MIX-STABLE", "E2-MIX-REORG")]
        improved = False
        for values in rows_for_seed:
            if "0" in values and "600" in values:
                improved |= (
                    values["0"]["answer_nll"] - values["600"]["answer_nll"]
                    >= c["exploratory_decisions"]["arithmetic_answer_nll_min_gain"]
                    and values["600"]["arithmetic_exact_match"] > values["0"]["arithmetic_exact_match"]
                )
        learnability.append({"seed": seed, "learnable_signal": improved})
    matching = [
        row
        for job in jobs
        if job["arm"] in ("E1-MIX-STABLE", "E2-MIX-REORG")
        for row in job.get("specialization", [])
        if row.get("control") == "USAGE_MATCHED"
    ]
    differentiation_status = (
        "INCONCLUSIVE"
        if not complete or not matching
        else "CAUSAL_INTERVENTION_REPORT_AVAILABLE_REQUIRES_REVIEW"
    )
    positive = sum(value > 0 for value in gains)
    story_harm_mean = (
        sum(row["story_harm"] for row in final_rows) / len(final_rows) if final_rows else None
    )
    exact_match_ok = bool(final_rows) and all(row["exact_match_change"] >= 0 for row in final_rows)
    sham_differences = [
        row["sham_answer_nll"]
        - by[(row["seed"], "E2-MIX-REORG")]["evaluations"]["600"]["answer_nll"]
        for row in final_rows
        if row["sham_answer_nll"] is not None
    ]
    sham_separates = bool(sham_differences) and sum(sham_differences) / len(sham_differences) > 0
    retained = bool(
        complete
        and gains
        and sum(gains) / len(gains) >= c["exploratory_decisions"]["arithmetic_answer_nll_min_gain"]
        and positive >= c["exploratory_decisions"]["minimum_seed_directions_positive"]
        and story_harm_mean <= c["exploratory_decisions"]["story_nll_harm_max"]
        and exact_match_ok
        and sham_separates
        and intervention_ok
    )
    if not complete:
        reorg_status = "INCONCLUSIVE"
    elif not intervention_ok:
        reorg_status = "INTERVENTION_INSUFFICIENT"
    elif middle_gains and sum(middle_gains) / len(middle_gains) > 0 and not retained:
        reorg_status = "TRANSIENT_GAIN_ONLY"
    elif retained:
        reorg_status = "PERSISTENT_GAIN_EXPLORATORY"
    else:
        reorg_status = "NO_DETECTABLE_REORGANIZATION_GAIN"
    decision = {
        "status": "INCONCLUSIVE" if not complete else "EXPLORATORY_SIGNAL" if retained else reorg_status,
        "arithmetic_learnability": (
            "INCONCLUSIVE" if not complete else
            "ARITHMETIC_LEARNABILITY_INSUFFICIENT" if not any(x["learnable_signal"] for x in learnability)
            else "LEARNABILITY_SIGNAL_OBSERVED"
        ),
        "learnability_by_seed": learnability,
        "functional_differentiation": differentiation_status,
        "reorganization": reorg_status,
        "phase_iii": "INCOMPLETE" if not complete else "PERSISTENT" if retained else "NOT_ESTABLISHED",
        "seed_level_final_gains": gains,
        "mean_final_gain": sum(gains) / len(gains) if gains else None,
        "mean_step_700_gain": sum(middle_gains) / len(middle_gains) if middle_gains else None,
        "positive_seed_directions": positive,
        "mean_story_harm": story_harm_mean,
        "all_seed_exact_match_nonharm": exact_match_ok,
        "e2_separates_from_sham": sham_separates,
        "seed_level_sham_minus_e2_answer_nll": sham_differences,
        "effective_connection_change_observed": intervention_ok,
        "confirmation": "NOT_AUTHORIZED",
        "scientific_decision": False,
    }
    tables = {
        "EXPERIMENT_MANIFEST.json": {"identity": ident, "config": c, "status": "COMPLETE_EXPLORATORY" if complete else "PARTIAL"},
        "INPUT_IDENTITY.json": input_identity,
        "MODEL_LINEAGE.json": {"parent": c["model"], "anchors": input_identity.get("anchors", []), "history_limitations": c["known_limitations"]},
        "ARITHMETIC_DATA_AUDIT.json": input_identity.get("input_manifest", {}).get("arithmetic", {}),
        "ARM_MATRIX.json": {"expected": [{"seed": s, "arm": a} for s in c["seeds"] for a in c["arms"]], "completed": [{"seed": r["seed"], "arm": r["arm"], "status": r["status"]} for r in jobs], "complete": complete},
        "TRAINING_CURVES.json": [{"seed": r["seed"], "arm": r["arm"], "curve": r.get("curve", [])} for r in jobs],
        "STORY_RETENTION.json": {"role": "REUSED_HISTORICAL_DEVELOPMENT", "comparisons": comparisons},
        "ARITHMETIC_CAPABILITY.json": {"role": "ARITH_DEV", "comparisons": comparisons, "evaluation_rows": evals},
        "CELL_EXPOSURE.json": [
            {
                "seed": r["seed"],
                "arm": r["arm"],
                "training_exposure_by_task": r.get("training_exposure", {}),
                "evaluation_exposure_by_task": r.get("cell_exposure", {}),
            }
            for r in jobs
        ],
        "GRADIENT_CONFLICT.json": [{"seed": r["seed"], "arm": r["arm"], "gradient_conflict": r.get("gradient_conflict")} for r in jobs],
        "FUNCTIONAL_DIFFERENTIATION.json": [{"seed": r["seed"], "arm": r["arm"], "interventions": r.get("specialization", {})} for r in jobs],
        "CONNECTION_INTERVENTION.json": transactions,
        "FOUR_GRID_ATTRIBUTION.json": [{"seed": r["seed"], "arm": r["arm"], "four_grid": r.get("four_grid", {})} for r in jobs],
        "LONG_HORIZON_VALUE.json": {"paired_seed_comparisons": comparisons, "decision": decision},
        "COMPUTE_COST.json": {"jobs": cost, "logical_updates": c["training"]["logical_updates"], "logical_tokens": c["training"]["logical_tokens"], "confirmation": "SEALED"},
    }
    for name, value in tables.items():
        atomic(dest / name, value)
    candidate_state = (
        "an effective Connection/path change was observed"
        if intervention_ok
        else "no effective Connection/path change was observed; this is intervention-insufficient, not evidence of no benefit"
    )
    next_step = (
        "Independent review may consider the next autonomous-organization question."
        if decision["reorganization"] == "PERSISTENT_GAIN_EXPLORATORY"
        else "Do not expand this experiment automatically; review task learnability, functional interventions, and candidate effectiveness first."
    )
    findings = [
        "# R3m-E Scientific Findings (exploratory)",
        "",
        f"Status: {decision['status']}",
        "",
        f"1. Arithmetic learning on held-out Arithmetic Dev facts: {decision['arithmetic_learnability']}; see per-seed Answer NLL, exact match, and carry strata in ARITHMETIC_CAPABILITY.json.",
        f"2. Cell functional differentiation: {decision['functional_differentiation']}; the full task-specific exposure, signed gradient conflict, and usage-matched replacement effects are retained for review.",
        "3. Stable Connection learning is measured directly by E1-MIX-STABLE against its Step 300 anchor; E0 is not treated as an equal-data Arithmetic baseline.",
        f"4. One-time reorganization result: {decision['reorganization']}; E2 is paired against both E1 and the E3 sham.",
        f"5. Persistence through global Step 900 / Phase III: {decision['phase_iii']}; Step 700 and Step 900 gains are reported side by side.",
        f"6. Intervention interpretation: {candidate_state}.",
        f"7. Next research decision: {next_step}",
        "",
        "Development Story data are reused historical data. Original R3m Main candidates and Confirmation were not accessed.",
        "All results are exploratory; no formal scientific decision is authorized.",
    ]
    (dest / "SCIENTIFIC-FINDINGS.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    artifacts = []
    for name in ARTIFACTS:
        path = dest / name
        if path.is_file() and name != "RUN_MANIFEST.json":
            artifacts.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {
        "identity": ident,
        "status": "COMPLETE_EXPLORATORY" if complete else "PARTIAL",
        "completed_jobs": len(jobs),
        "expected_jobs": len(c["seeds"]) * len(c["arms"]),
        "failures": list(failures),
        "artifacts": artifacts,
        "confirmation": "NOT_AUTHORIZED",
        "scientific_decision": False,
    }
    atomic(dest / "RUN_MANIFEST.json", manifest)
    return manifest
