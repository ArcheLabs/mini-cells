from __future__ import annotations

import math

LOCAL_RELATIVE_MSE_MAX = 0.10
LOCAL_COSINE_MIN = 0.95
HANDOFF_SAFETY_RATIO_MAX = 1.20
HANDOFF_PROGRESS_RATIO_MAX = 1.01
FINAL_HANDOFF_RATIO_MAX = 1.03
QUALITY_SAFE_K_MAX = 5
SAMPLE_VARIATION_MIN = 0.05
NORMALIZED_ADVANTAGE_MIN = 0.002
RECEPTOR_RATIO_MAX = 0.05


def local_imitation_pass(relative_mse: float, cosine: float) -> bool:
    return relative_mse <= LOCAL_RELATIVE_MSE_MAX and cosine >= LOCAL_COSINE_MIN


def handoff_stage_pass(*, before_ppl: float, after_ppl: float, teacher_ppl: float) -> bool:
    safety_ratio = after_ppl / teacher_ppl
    progress_ratio = after_ppl / before_ppl
    return (
        safety_ratio <= HANDOFF_SAFETY_RATIO_MAX
        and progress_ratio <= HANDOFF_PROGRESS_RATIO_MAX
    )


def make_closed_loop_decision(
    workers: list[dict[str, object]],
    arms: list[dict[str, object]],
    *,
    teacher_nll: float,
) -> dict[str, object]:
    if any(row["status"] == "CLMV2_SCAFFOLD_EQUIVALENCE_FAILURE" for row in workers):
        diagnosis = "CLMV2_SCAFFOLD_EQUIVALENCE_FAILURE"
        successes = 0
        evidence: list[dict[str, object]] = []
    elif sum(
        row["status"] == "CLMV2_CLOSED_LOOP_LOCAL_APPROXIMATION_FAILURE"
        for row in workers
    ) >= 2:
        diagnosis = "CLMV2_CLOSED_LOOP_LOCAL_APPROXIMATION_FAILURE"
        successes = 0
        evidence = []
    elif sum(
        row["status"] == "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE" for row in workers
    ) >= 2:
        diagnosis = "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE"
        successes = 0
        evidence = []
    else:
        grouped: dict[int, dict[str, dict[str, object]]] = {}
        for row in arms:
            grouped.setdefault(int(row["replicate"]), {})[str(row["arm"])] = row

        successes = 0
        handoffs = 0
        safe_capacity = 0
        varied = 0
        causal = 0
        evidence = []
        teacher_ppl = math.exp(teacher_nll)
        for worker in workers:
            replicate = int(worker["replicate"])
            if worker["status"] != "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL":
                continue
            handoffs += 1
            replicate_arms = grouped.get(replicate, {})
            if set(replicate_arms) != {"dense", "dynamic", "static", "shuffled"}:
                continue
            dynamic = replicate_arms["dynamic"]
            top_k = int(worker["quality_safe_k"])
            quality = float(dynamic["ppl"]) / teacher_ppl
            static_advantage = (
                float(replicate_arms["static"]["nll"]) - float(dynamic["nll"])
            ) / teacher_nll
            shuffled_advantage = (
                float(replicate_arms["shuffled"]["nll"]) - float(dynamic["nll"])
            ) / teacher_nll
            is_safe = top_k <= QUALITY_SAFE_K_MAX and quality <= FINAL_HANDOFF_RATIO_MAX
            is_varied = float(dynamic["sample_variation"]) >= SAMPLE_VARIATION_MIN
            is_causal = (
                static_advantage >= NORMALIZED_ADVANTAGE_MIN
                and shuffled_advantage >= NORMALIZED_ADVANTAGE_MIN
            )
            receptor_ok = float(dynamic["receptor_ratio"]) <= RECEPTOR_RATIO_MAX
            passed = is_safe and is_varied and is_causal and receptor_ok
            safe_capacity += int(is_safe)
            varied += int(is_varied)
            causal += int(is_causal)
            successes += int(passed)
            evidence.append({
                "replicate": replicate,
                "quality_safe_k": top_k,
                "quality_ratio_to_teacher": quality,
                "sample_variation": float(dynamic["sample_variation"]),
                "static_advantage": static_advantage,
                "shuffled_advantage": shuffled_advantage,
                "receptor_ratio": float(dynamic["receptor_ratio"]),
                "passed": passed,
            })

        if successes >= 2:
            diagnosis = "CLMV2_PROGRAM_CONDITIONALITY_SIGNAL"
        elif safe_capacity >= 2:
            diagnosis = "CLMV2_CONDITIONAL_CAPACITY_WITHOUT_CAUSAL_ROUTING"
        elif handoffs >= 2:
            diagnosis = "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL"
        else:
            diagnosis = "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE"

    result = {
        "format": "minicells.clm-v2-validation-001b.v1",
        "experiment": "CLM v2 Validation 001b — Closed-Loop Scaffold Handoff",
        "status": "PASS" if diagnosis in (
            "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL",
            "CLMV2_CONDITIONAL_CAPACITY_WITHOUT_CAUSAL_ROUTING",
            "CLMV2_PROGRAM_CONDITIONALITY_SIGNAL",
        ) else "FAIL",
        "diagnosis": diagnosis,
        "successful_replicates": successes,
        "strong_program_sparsity": sum(
            int(row.get("quality_safe_k", 12)) <= 4 for row in workers
        ) >= 2,
        "thresholds": {
            "local_relative_mse_max": LOCAL_RELATIVE_MSE_MAX,
            "local_cosine_min": LOCAL_COSINE_MIN,
            "handoff_safety_ratio_max": HANDOFF_SAFETY_RATIO_MAX,
            "handoff_progress_ratio_max": HANDOFF_PROGRESS_RATIO_MAX,
            "final_handoff_ratio_max": FINAL_HANDOFF_RATIO_MAX,
            "quality_safe_k_max": QUALITY_SAFE_K_MAX,
            "sample_variation_min": SAMPLE_VARIATION_MIN,
            "normalized_advantage_min": NORMALIZED_ADVANTAGE_MIN,
            "receptor_ratio_max": RECEPTOR_RATIO_MAX,
        },
        "evidence": evidence,
    }
    return result
