"""Seed-level paired endpoints; examples and checkpoints are never independent Seeds."""

import math
from statistics import mean, stdev
from minicells_jct001.analysis import holm


def mean_interval(values):
    if len(values) != 5 or not all(math.isfinite(v) for v in values):
        raise ValueError("CLM002_FIVE_COMPLETE_CONFIRMATION_SEEDS_REQUIRED")
    centre = mean(values)
    radius = 2.7764451051977987 * stdev(values) / math.sqrt(5)
    return dict(
        mean=centre,
        low=centre - radius,
        high=centre + radius,
        method="TWO_SIDED_PAIRED_SEED_STUDENT_T_DF4_95_PERCENT_CANDIDATE",
        assumptions="INDEPENDENT_SEED_EFFECTS_AND_APPROX_NORMALITY; REGISTRATION_REQUIRED",
        n=5,
    )


def factorial(seed_rows):
    result = []
    if len(seed_rows) != 5 or len({r["seed"] for r in seed_rows}) != 5:
        raise ValueError("CLM002_DISTINCT_COMPLETE_SEED_MATRIX_REQUIRED")
    for row in seed_rows:
        if set(row["losses"]) != set(
            k + "-" + d for k in ("B00", "B10", "B01", "B11") for d in ("S", "H")
        ):
            raise ValueError("CLM002_INCOMPLETE_FACTORIAL_MATRIX")
        if not all(math.isfinite(v) for v in row["losses"].values()):
            raise ValueError("CLM002_NONFINITE_FACTORIAL_ENDPOINT")
        effects = {
            d: row["losses"]["B11-" + d]
            - row["losses"]["B10-" + d]
            - row["losses"]["B01-" + d]
            + row["losses"]["B00-" + d]
            for d in ("S", "H")
        }
        result.append(
            dict(
                seed=row["seed"],
                I_H=effects["H"],
                I_S=effects["S"],
                delta_I=effects["H"] - effects["S"],
            )
        )
    return dict(
        per_seed=result,
        I_H=mean_interval([r["I_H"] for r in result]),
        I_S=mean_interval([r["I_S"] for r in result]),
        delta_I=mean_interval([r["delta_I"] for r in result]),
        primary="I_H",
        secondary="delta_I",
        statistical_unit="TRAINING_SEED",
        analysis_status="DRAFT_NOT_SCIENTIFICALLY_FROZEN",
        secondary_intervals="UNADJUSTED_DESCRIPTIVE; HOLM_DECISION_REQUIRED_FOR_FAMILYWISE_CLAIM",
        secondary_multiplicity="HOLM",
        scientific_decision=False,
    )


def stage_a(rows, threshold=0.01, paired_threshold=0.01):
    if len(rows) != 5 or len({r["seed"] for r in rows}) != 5:
        return dict(
            status="STAGE_A_INCOMPLETE",
            reason="REQUIRES_FIVE_DISTINCT_CONFIRMATION_SEEDS",
            stage_b_authorized=False,
        )
    complete = all(
        r.get("complete") and r.get("registered_confirmation") and r.get("integrity_pass")
        for r in rows
    )
    if not complete:
        return dict(
            status="STAGE_A_INCOMPLETE",
            reason="INTEGRITY_OR_REGISTRATION_INCOMPLETE",
            stage_b_authorized=False,
        )
    accepted = []
    for r in rows:
        windows = r["checkpoints"]
        steps = [w.get("step") for w in windows]
        if any(not isinstance(step, int) for step in steps) or len(set(steps)) != len(steps):
            return dict(
                status="STAGE_A_INCOMPLETE",
                reason="CHECKPOINT_IDENTITIES_MISSING_OR_DUPLICATED",
                stage_b_authorized=False,
            )
        valid = [
            w
            for w in windows
            if w["H"]["bidirectional"]
            and w["H"]["S"] >= threshold
            and w["H"]["S"] - w["S"]["S"] >= paired_threshold
            and not w.get("single_route_collapse", False)
        ]
        accepted.append(len(valid) >= 2 and max(steps, default=0) in {w["step"] for w in valid})
    supported = sum(accepted) >= 4
    return dict(
        status="STAGE_A_DIFFERENTIATION_SUPPORTED"
        if supported
        else "STAGE_A_DIFFERENTIATION_NOT_SUPPORTED",
        accepted_seed_count=sum(accepted),
        stage_b_authorized=False,
        release_note="REQUIRES_ARCHIVED_INDEPENDENT_STAGE_A_REVIEW_AND_SEPARATE_RELEASE",
    )


def joint_decision(
    effects,
    *,
    registered,
    stage_a_archived,
    integrity,
    quality_safe,
    differentiation,
    cost_complete,
    cost_regression,
    minimum=0.01,
    heterogeneity_multiplicity_pass=False,
):
    if not all((registered, stage_a_archived, integrity, quality_safe, cost_complete)):
        return "INCOMPLETE_OR_PROTOCOL_VIOLATION"
    extra = effects["I_H"]["high"] <= -minimum and effects["I_H"]["mean"] < 0
    if not extra:
        return (
            "FUNCTIONAL_DIFFERENTIATION_WITHOUT_JOINT_BENEFIT"
            if differentiation
            else "JOINT_SYNERGY_NOT_SUPPORTED"
        )
    if not differentiation:
        return "JOINT_SYNERGY_NOT_SUPPORTED"
    if cost_regression:
        return "JOINT_QUALITY_GAIN_WITH_COST_REGRESSION"
    return (
        "JOINT_SYNERGY_SUPPORTED"
        if effects["delta_I"]["high"] < 0 and heterogeneity_multiplicity_pass
        else "JOINT_BENEFIT_WITHOUT_HETEROGENEITY_INTERACTION"
    )


def correct_secondary(pvalues):
    return holm(pvalues)
