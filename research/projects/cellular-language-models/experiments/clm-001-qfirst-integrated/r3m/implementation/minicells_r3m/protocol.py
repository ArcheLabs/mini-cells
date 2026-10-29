"""Budget, phase boundaries and independent scientific decisions; no outcome-driven tuning."""

import math


def arithmetic_fraction(relative_step, environment):
    if environment not in ("N", "C") or not 0 <= relative_step < 600:
        raise ValueError("CURRICULUM_IDENTITY_INVALID")
    return (
        0.0
        if environment == "N"
        else (0.1 if relative_step < 100 else 0.3 if relative_step < 200 else 0.5)
    )


def task_for(relative_step, environment):
    # Fixed step-level ratio within each ten-step block, not adaptive to quality.
    return (
        "arithmetic"
        if relative_step % 10 < round(10 * arithmetic_fraction(relative_step, environment))
        else "story"
    )


def schedule(anchor_lr, anchor_epsilon, relative_step, c):
    if not (math.isfinite(anchor_lr) and anchor_lr > 0 and 0 <= relative_step <= 600):
        raise ValueError("SCHEDULE_INVALID")
    f = min(relative_step, c["continuation_steps"] - 1) / max(1, c["continuation_steps"] - 1)
    return (
        anchor_lr * (1 + (c["lr_terminal_fraction"] - 1) * f),
        anchor_epsilon + (c["epsilon_terminal"] - anchor_epsilon) * f,
    )


def main_gate(c, pilot, pilot_sha):
    release = c["main_release"]
    if (
        not release["reviewed"]
        or not release["frozen"]
        or release["pilot_manifest_sha256"] != pilot_sha
        or pilot.get("status") != "REORGANIZATION_OPPORTUNITY_OBSERVED"
    ):
        raise RuntimeError("MAIN_REQUIRES_REVIEWED_POSITIVE_PILOT_AND_FROZEN_PROTOCOL")
    if not c.get("main_initialization") or set(c["main_initialization"]) != set(
        map(str, c["main_seeds"])
    ):
        raise RuntimeError("MAIN_PHASE_I_INITIALIZATION_NOT_FROZEN")
    if any(v is None for v in c.get("main_thresholds", {}).values()) or not c.get(
        "main_thresholds"
    ):
        raise RuntimeError("MAIN_SCIENTIFIC_THRESHOLDS_NOT_FROZEN")
    if any(
        c["pressure"].get(k) is None
        for k in ("threshold", "loss_mean", "loss_std", "gradient_mean", "gradient_std")
    ):
        raise RuntimeError("MAIN_PRESSURE_CALIBRATION_NOT_FROZEN")
    if any(c["pressure"][k] <= 0 for k in ("loss_std", "gradient_std")):
        raise RuntimeError("MAIN_PRESSURE_SCALE_INVALID")


def pilot_decision(rows, thresholds, expected_seeds):
    if {r["seed"] for r in rows} != set(expected_seeds) or len(rows) != len(expected_seeds):
        return {"status": "INCONCLUSIVE", "Q4": "NOT_TESTED", "reason": "INCOMPLETE_SEED_MATRIX"}
    if not all(r["task_learnable"] for r in rows):
        return {"status": "TASK_LEARNABILITY_INSUFFICIENT", "Q4": "NOT_TESTED"}
    effective = sum(r["effective"] and r["action_change"] >= thresholds["action_min"] for r in rows)
    if effective < thresholds["minimum_seeds"]:
        return {"status": "INTERVENTION_INSUFFICIENT", "Q4": "NOT_TESTED"}
    positive = sum(r["answer_gain_600"] > 0 for r in rows)

    def mean(key):
        return sum(r[key] for r in rows) / len(rows)

    retained = all(
        r["answer_gain_400"] > 0
        and r["answer_gain_600"] >= thresholds["retention_min"] * r["answer_gain_400"]
        for r in rows
    )
    passed = (
        positive >= thresholds["minimum_seeds"]
        and mean("answer_gain_600") >= thresholds["answer_gain"]
        and mean("story_harm_600") <= thresholds["story_harm_max"]
        and mean("sham_advantage_600") > 0
        and retained
        and all(
            r["em_change_600"] >= -thresholds["em_harm_max"] and r["attribution_valid"]
            for r in rows
        )
    )
    return {
        "status": "REORGANIZATION_OPPORTUNITY_OBSERVED"
        if passed
        else "NO_DETECTABLE_REORGANIZATION_OPPORTUNITY",
        "Q3": "PERSISTENT_REORGANIZATION_GAIN" if passed else "NO_PERSISTENT_REORGANIZATION_GAIN",
        "Q4": "NOT_TESTED",
        "effective_seeds": effective,
        "positive_seeds": positive,
        "independent_confirmation": False,
    }
