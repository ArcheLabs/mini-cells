"""Additive correction: preserve old artifacts and distinguish attempted from applied work."""

from minicells_clm002.native import treatment

from .common import atomic, digest, read


def reconcile(c, reports, output):
    manifest = read(reports / "RUN_MANIFEST.json")
    ident = manifest["identity"]
    if (
        ident["source_sha"] != c["historical"]["source_sha"]
        or ident["run_id"] != c["historical"]["run_id"]
    ):
        raise RuntimeError("FAST002_HISTORICAL_RUN_PROVENANCE_MISMATCH")
    expected = {f"{seed}-{arm}.json" for seed in c["seeds"] for arm in c["arms"]}
    paths = list((reports / "jobs").glob("*.json"))
    if {p.name for p in paths} != expected:
        raise RuntimeError("FAST002_HISTORICAL_MATRIX_INCOMPLETE")
    totals = {
        "logical_updates": 0,
        "main_target_tokens": 0,
        "cell_optimizer_calls": 0,
        "candidate_optimizer_calls": 0,
        "proposals": 0,
        "scale_checks": 0,
        "admitted_proposals": 0,
        "committed_connection_updates": 0,
        "rejected_candidate_optimizer_calls": 0,
        "synchronized_training_wall_seconds": 0.0,
    }
    corrected = []
    for path in sorted(paths):
        job = read(path)
        if (
            job["identity"] != dict(ident, seed=job["seed"], arm=job["arm"], phase=job["phase"])
            or job["step"] != 500
            or len(job["curve"]) != 500
            or [r["step"] for r in job["curve"]] != list(range(1, 501))
            or job["main_target_tokens"] != 256000
            or not job["gpu_used"]
        ):
            raise RuntimeError("FAST002_JOB_PROVENANCE_OR_BUDGET_MISMATCH:" + path.name)
        proposals = [r["proposal"] for r in job["curve"] if r["proposal"] is not None]
        cells, connection = treatment(job["arm"])
        proposed = sum(p["policy_optimizer_updates"] for p in proposals)
        admitted = sum(p["chosen_scale"] is not None for p in proposals)
        if len(proposals) != (10 if connection else 0):
            raise RuntimeError("FAST002_PROPOSAL_CADENCE_MISMATCH")
        for p in proposals:
            chosen = p["chosen_scale"]
            passed = [a for a in p["attempts"] if a["passed"]]
            if (
                (chosen is None) != (len(passed) == 0)
                or chosen is not None
                and passed[-1]["scale"] != chosen
            ):
                raise RuntimeError("FAST002_PROPOSAL_ADMISSION_RECORD_INCONSISTENT")
        counter = job["budget"]["optimizer_calls_by_group"]
        if (
            counter.get("cell", 0) != (500 if cells else 0)
            or counter.get("connection", 0) != proposed
        ):
            raise RuntimeError("FAST002_OPTIMIZER_COUNTER_CURVE_MISMATCH")
        if job["budget"]["uncertain_optimizer_calls"] or job["budget"]["pending_attempts"]:
            raise RuntimeError("FAST002_UNCERTAIN_HISTORICAL_WORK")
        row = {
            "job": path.stem,
            "original_sha256": digest(path),
            "gpu_used": job["gpu_used"],
            "device": job["environment"]["gpu"],
            "logical_updates": job["step"],
            "main_target_tokens": job["main_target_tokens"],
            "cell_optimizer_calls": counter.get("cell", 0),
            "candidate_optimizer_calls": proposed,
            "proposals": len(proposals),
            "scale_checks": sum(len(p["attempts"]) for p in proposals),
            "admitted_proposals": admitted,
            "committed_connection_updates": admitted,
            "rejected_candidate_optimizer_calls": sum(
                p["policy_optimizer_updates"] for p in proposals if p["chosen_scale"] is None
            ),
            "synchronized_training_wall_seconds": job["training_seconds"],
            "gpu_kernel_seconds": None,
            "gpu_kernel_time_status": "NOT_MEASURED_NOT_INFERRED_FROM_WALL_TIME",
            "legacy_budget_label": job["budget"]["accounting_status"],
            "accounting_status": "GPU_LANGUAGE_TARGETS_SUMMARY_AND_CURVE_RECONCILED",
            "evidence_limit": "RAW_COST_JOURNALS_NOT_IN_PUBLIC_PACKAGE; COUNTERS_CROSS_CHECKED_WITH_CURVE_NOT_INDEPENDENT_OPTIMIZER_ATTESTATION",
        }
        corrected.append(row)
        for key in totals:
            totals[key] += row[key]
    report = {
        "status": "FAST001_ADDITIVE_AUDIT_COMPLETE",
        "historical_result_commit": c["historical"]["results_commit"],
        "historical_identity": ident,
        "totals": totals,
        "jobs": corrected,
        "original_conclusion": read(reports / "FINAL_ANALYSIS.json")["status"],
        "historical_artifacts_unchanged": True,
        "stage_a": read(reports / "STAGE_A_DECISION.json"),
        "joint_effect_interpretation": "NOT_TESTED_CONNECTION_TREATMENT_NOT_DELIVERED"
        if totals["committed_connection_updates"] == 0
        else "REQUIRES_TREATMENT_COVERAGE_REVIEW",
        "old_statistics_preserved": True,
        "scientific_optimizer_updates_executed_by_audit": 0,
    }
    atomic(output / "HISTORICAL_AUDIT.json", report)
    return report
