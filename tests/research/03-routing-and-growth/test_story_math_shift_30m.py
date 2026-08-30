from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.story_math_shift_30m import (
    EVAL_TOKENS,
    GROWTH_DECISION_TOKENS,
    MATH_FRACTION,
    PROBATION_TOKENS,
    SHIFT_BASE_LR,
    SHIFT_STEPS,
    SHIFT_TOKENS,
    STORY_FRACTION,
    TOKENS_PER_STEP,
    pareto_crossover,
    promotion_decision,
    schedule_manifest,
    shift_domain,
    shift_lr,
)


ROOT = Path(__file__).resolve().parents[1]


def test_shift_schedule_is_exactly_90_10_and_deterministic() -> None:
    first = [shift_domain(step) for step in range(1000)]
    second = [shift_domain(step) for step in range(1000)]
    assert first == second
    assert first.count("math") == 900
    assert first.count("story") == 100
    manifest = schedule_manifest(SHIFT_TOKENS)
    assert manifest["math_fraction"] == MATH_FRACTION
    assert manifest["story_fraction"] == STORY_FRACTION
    assert int(manifest["steps"]) == SHIFT_STEPS


def test_shift_budget_and_growth_windows_are_aligned() -> None:
    assert SHIFT_TOKENS == 50_000_000
    assert TOKENS_PER_STEP == 1000
    assert PROBATION_TOKENS == 2_000_000
    assert GROWTH_DECISION_TOKENS == (10_000_000, 25_000_000)
    assert all(value % TOKENS_PER_STEP == 0 for value in EVAL_TOKENS)
    assert all(value + PROBATION_TOKENS <= SHIFT_TOKENS for value in GROWTH_DECISION_TOKENS)


def test_shift_lr_is_bounded_and_keeps_late_learning_rate() -> None:
    values = [shift_lr(step) for step in (1, 500, SHIFT_STEPS // 2, SHIFT_STEPS)]
    assert all(0.0 < value <= SHIFT_BASE_LR for value in values)
    assert shift_lr(500) == SHIFT_BASE_LR
    assert shift_lr(SHIFT_STEPS) > 0.0
    assert shift_lr(SHIFT_STEPS) <= SHIFT_BASE_LR * 0.101


def _evaluation(story: float, math: float, *, accuracy: float) -> dict[str, object]:
    return {
        "story_ppl": float(torch.exp(torch.tensor(story))),
        "story_batch_nlls": [story - 0.01, story, story + 0.01, story],
        "math_nll": math,
        "math_batch_nlls": [math - 0.01, math, math + 0.01, math],
        "math_exact_answer_accuracy": accuracy,
    }


def test_budgeted_promotion_requires_future_utility_and_retention() -> None:
    control = _evaluation(1.0, 1.0, accuracy=0.40)
    better = _evaluation(0.99, 0.98, accuracy=0.43)
    decision = promotion_decision(control, better, seed=1)
    assert decision["promote"] is True

    forgetful = _evaluation(1.05, 0.95, accuracy=0.48)
    rejected = promotion_decision(control, forgetful, seed=1)
    assert rejected["promote"] is False


def test_pareto_crossover_is_first_joint_dominance_point() -> None:
    llm = [
        {"shift_tokens": 0, "story_ppl": 5.0, "math_exact_answer_accuracy": 0.10},
        {"shift_tokens": 10, "story_ppl": 6.0, "math_exact_answer_accuracy": 0.40},
        {"shift_tokens": 20, "story_ppl": 7.0, "math_exact_answer_accuracy": 0.60},
    ]
    clm = [
        {"shift_tokens": 0, "story_ppl": 5.1, "math_exact_answer_accuracy": 0.10},
        {"shift_tokens": 10, "story_ppl": 5.8, "math_exact_answer_accuracy": 0.35},
        {"shift_tokens": 20, "story_ppl": 6.2, "math_exact_answer_accuracy": 0.61},
    ]
    result = pareto_crossover(llm, clm)
    assert result is not None
    assert result["shift_tokens"] == 20


def test_unattended_protocol_is_frozen_to_eight_hours() -> None:
    protocol = json.loads(
        (ROOT / "research/stages/03-routing-and-growth/sources/experiment-025-protocol.json").read_text(encoding="utf-8")
    )
    hardware = protocol["hardware_budget"]
    assert hardware["global_wall_hours"] == 8.0
    assert hardware["finalization_reserve_minutes"] == 30
    assert hardware["worker_round_wall_hours"] == 2.5
    assert hardware["automatic_resume"] is True
    assert hardware["one_shot_kaggle_run_all"] is True
    assert protocol["runtime_estimate"]["clm_main_plus_counterfactual_physical_tokens"] == 54_000_000


def test_kaggle_notebook_uses_one_shot_budget_and_auto_publish() -> None:
    notebook = json.loads(
        (ROOT / "research/notebooks/03-routing-and-growth/experiment-025-story-math-growth.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "--total-wall-hours', '8'" in source
    assert "--round-wall-hours', '2.5'" in source
    assert "publish_experiment_025_story_math_growth.py" in source
    assert "--push" in source


def test_experiment_025_scripts_compile() -> None:
    for relative in (
        "scripts/research/run_experiment_025_story_math_worker.py",
        "scripts/research/run_experiment_025_story_math_growth.py",
        "scripts/research/report_experiment_025_story_math_growth.py",
    ):
        path = ROOT / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
