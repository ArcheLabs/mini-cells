"""Original upper bounds plus genuinely nonempty hard-path updates."""

from minicells_r3f.evaluation import admission as original_admission


def decide(metrics, c, numeric):
    result = original_admission(metrics, c, numeric)
    effective = (
        metrics["centered_score_delta_max_abs"] > numeric
        and metrics["action_changed_count"] > 0
        and metrics["path_changed_count"] > 0
    )
    result["effective_update"] = effective
    result["trust_region_passed"] = (
        result["kl_pass"]
        and result["action_change_pass"]
        and result["path_change_pass"]
        and effective
    )
    result["reject_reasons"] = [r for r in result["reject_reasons"] if r != "NO_EFFECTIVE_UPDATE"]
    if not effective:
        result["reject_reasons"].append("NO_ACTUAL_NONEMPTY_PATH_UPDATE")
    result["passed"] = result["trust_region_passed"]
    result["path_coverage_at_least_one_percent"] = metrics["path_change"] >= 0.01
    return result


def select(attempts):
    chosen = next((r for r in attempts if r["scale"] > 0 and r["passed"]), None)
    if chosen:
        return chosen["scale"], None
    stable = [
        r
        for r in attempts
        if r["scale"] > 0 and r["kl_pass"] and r["action_change_pass"] and r["path_change_pass"]
    ]
    return None, "ONLY_NUMERIC_NOOP_SCALES" if stable else "NO_TRUST_REGION_SCALE"
