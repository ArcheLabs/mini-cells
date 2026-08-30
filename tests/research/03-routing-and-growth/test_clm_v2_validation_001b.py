from __future__ import annotations

import math

from minicells.clm_v2_closed_loop_validation import (
    HANDOFF_PROGRESS_RATIO_MAX,
    HANDOFF_SAFETY_RATIO_MAX,
    LOCAL_COSINE_MIN,
    LOCAL_RELATIVE_MSE_MAX,
    handoff_stage_pass,
    local_imitation_pass,
    make_closed_loop_decision,
)


def test_local_imitation_gate_is_local_not_direct_rollout() -> None:
    assert local_imitation_pass(LOCAL_RELATIVE_MSE_MAX, LOCAL_COSINE_MIN)
    assert not local_imitation_pass(LOCAL_RELATIVE_MSE_MAX + 1e-3, 0.99)
    assert not local_imitation_pass(0.01, LOCAL_COSINE_MIN - 1e-3)


def test_handoff_gate_requires_safety_and_recovery() -> None:
    teacher = 20.0
    before = 22.0
    after = 21.5
    assert handoff_stage_pass(before_ppl=before, after_ppl=after, teacher_ppl=teacher)

    unsafe = teacher * (HANDOFF_SAFETY_RATIO_MAX + 0.01)
    assert not handoff_stage_pass(before_ppl=unsafe * 1.01, after_ppl=unsafe, teacher_ppl=teacher)

    regressed = before * (HANDOFF_PROGRESS_RATIO_MAX + 0.01)
    assert not handoff_stage_pass(before_ppl=before, after_ppl=regressed, teacher_ppl=teacher)


def _worker(replicate: int, status: str, k: int = 5) -> dict[str, object]:
    return {"replicate": replicate, "status": status, "quality_safe_k": k}


def _arms(replicate: int, teacher_nll: float) -> list[dict[str, object]]:
    dynamic_nll = teacher_nll * 1.005
    return [
        {"replicate": replicate, "arm": "dense", "nll": teacher_nll,
         "ppl": math.exp(teacher_nll), "sample_variation": 0.0,
         "receptor_ratio": 0.0},
        {"replicate": replicate, "arm": "dynamic", "nll": dynamic_nll,
         "ppl": math.exp(teacher_nll) * 1.01, "sample_variation": 0.10,
         "receptor_ratio": 0.02},
        {"replicate": replicate, "arm": "static", "nll": dynamic_nll + teacher_nll * 0.003,
         "ppl": math.exp(dynamic_nll + teacher_nll * 0.003), "sample_variation": 0.0,
         "receptor_ratio": 0.02},
        {"replicate": replicate, "arm": "shuffled", "nll": dynamic_nll + teacher_nll * 0.004,
         "ppl": math.exp(dynamic_nll + teacher_nll * 0.004), "sample_variation": 0.10,
         "receptor_ratio": 0.02},
    ]


def test_decision_recognizes_closed_loop_handoff_without_final_sparsity() -> None:
    teacher_nll = 3.0
    workers = [
        _worker(0, "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL", 6),
        _worker(1, "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL", 6),
        _worker(2, "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE", 6),
    ]
    decision = make_closed_loop_decision(workers, [], teacher_nll=teacher_nll)
    assert decision["diagnosis"] == "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL"
    assert decision["status"] == "PASS"


def test_decision_recognizes_program_conditionality() -> None:
    teacher_nll = 3.0
    workers = [
        _worker(0, "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL", 5),
        _worker(1, "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL", 5),
        _worker(2, "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE", 6),
    ]
    arms = [*_arms(0, teacher_nll), *_arms(1, teacher_nll)]
    decision = make_closed_loop_decision(workers, arms, teacher_nll=teacher_nll)
    assert decision["diagnosis"] == "CLMV2_PROGRAM_CONDITIONALITY_SIGNAL"
    assert decision["successful_replicates"] == 2


def test_local_approximation_failure_has_priority() -> None:
    workers = [
        _worker(0, "CLMV2_CLOSED_LOOP_LOCAL_APPROXIMATION_FAILURE", 6),
        _worker(1, "CLMV2_CLOSED_LOOP_LOCAL_APPROXIMATION_FAILURE", 6),
        _worker(2, "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL", 6),
    ]
    decision = make_closed_loop_decision(workers, [], teacher_nll=3.0)
    assert decision["diagnosis"] == "CLMV2_CLOSED_LOOP_LOCAL_APPROXIMATION_FAILURE"
    assert decision["status"] == "FAIL"
