"""Frozen FAST seed-level decisions; no best-checkpoint selection."""

import math
from statistics import mean, stdev

from minicells_clm002.statistics import mean_interval
from minicells_jct001.analysis import holm


def stage_a(c, rows):
    if len(rows) != 5 or {r["seed"] for r in rows} != set(c["seeds"]):
        return {"status": "STAGE_A_INCOMPLETE", "stage_b_allowed": False}
    accepted = []
    for row in rows:
        good = True
        for step in c["stage_a"]["persistence_steps"]:
            s, h = row["S"][str(step)], row["H"][str(step)]
            good = (
                good
                and h["specialization"]["bidirectional"]
                and h["specialization"]["S"] >= c["stage_a"]["specialization_min"]
                and h["specialization"]["S"] - s["specialization"]["S"]
                >= c["stage_a"]["paired_min"]
                and not (s["single_route_collapse"] or h["single_route_collapse"])
            )
        accepted.append({"seed": row["seed"], "passed": bool(good)})
    passed = sum(r["passed"] for r in accepted) >= c["stage_a"]["required_seeds"]
    return {
        "status": "STAGE_A_DIFFERENTIATION_SUPPORTED"
        if passed
        else "STAGE_A_DIFFERENTIATION_NOT_SUPPORTED",
        "stage_b_allowed": passed,
        "per_seed": accepted,
    }


def t_one_sided_negative(values):
    # Exact Student-t df=4 CDF via the elementary closed-form integral.
    centre = mean(values)
    error = stdev(values) / math.sqrt(5)
    if error == 0:
        return 0.0 if centre < 0 else 1.0 if centre > 0 else 0.5
    t = centre / error
    u = t / math.sqrt(4 + t * t)
    return min(1.0, max(0.0, 0.5 + 0.75 * (u - u**3 / 3)))


def factorial(c, rows):
    if len(rows) != 5 or {row["seed"] for row in rows} != set(c["seeds"]):
        return {"status": "INCOMPLETE_OR_PROTOCOL_VIOLATION", "scientific_support": False}
    effects, quality, differentiated = [], True, 0
    for row in rows:
        values = row["arms"]
        if set(values) != {k + "-" + d for k in ("B00", "B10", "B01", "B11") for d in ("S", "H")}:
            return {"status": "INCOMPLETE_OR_PROTOCOL_VIOLATION", "scientific_support": False}
        e = {
            d: values["B11-" + d]["L"]
            - values["B10-" + d]["L"]
            - values["B01-" + d]["L"]
            + values["B00-" + d]["L"]
            for d in ("S", "H")
        }
        effects.append(
            {"seed": row["seed"], "I_H": e["H"], "I_S": e["S"], "delta_I": e["H"] - e["S"]}
        )
        for condition in ("S", "H"):
            for task in ("Story", "Arithmetic"):
                base = values["B00-" + condition]["tasks"][task]["nll"]
                quality = quality and all(
                    values[arm + "-" + condition]["tasks"][task]["nll"] - base
                    <= c["analysis"]["quality_cap"]
                    for arm in ("B10", "B01", "B11")
                )
        h, s = values["B11-H"], values["B11-S"]
        differentiated += int(
            h["specialization"]["S"] >= c["stage_a"]["specialization_min"]
            and h["specialization"]["S"] - s["specialization"]["S"] >= c["stage_a"]["paired_min"]
            and h["specialization"]["bidirectional"]
            and not (h["single_route_collapse"] or s["single_route_collapse"])
        )
    primary = mean_interval([r["I_H"] for r in effects])
    secondary_values = [r["delta_I"] for r in effects]
    secondary = mean_interval(secondary_values)
    p = t_one_sided_negative(secondary_values)
    secondary_pass = mean(secondary_values) < 0 and holm([p])[0] < 0.05
    primary_pass = primary["high"] <= -c["analysis"]["minimum_effect"]
    support = (
        primary_pass
        and quality
        and differentiated >= c["analysis"]["joint_specialization_required_seeds"]
    )
    return {
        "status": "JOINT_SYNERGY_SUPPORTED"
        if support and secondary_pass
        else "JOINT_BENEFIT_WITHOUT_HETEROGENEITY_INTERACTION"
        if support
        else "JOINT_SYNERGY_NOT_SUPPORTED",
        "scientific_support": support,
        "per_seed": effects,
        "I_H": primary,
        "delta_I": secondary,
        "secondary_holm_p": holm([p])[0],
        "quality_safe": quality,
        "joint_specialized_seed_count": differentiated,
        "interpretation_scope": "CONTROLLED_SYNTHETIC_TASKS_CURRENT_MODEL_AND_BUDGET_ONLY",
    }
