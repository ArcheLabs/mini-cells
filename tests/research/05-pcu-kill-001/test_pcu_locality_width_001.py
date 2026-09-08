"""Static/unit guards for PCU-LOCALITY-WIDTH-001."""

from __future__ import annotations

import inspect

import pytest

from minicells.pcu_kill_001 import locality_width


def test_locality_width_holds_non_width_variables_fixed() -> None:
    assert locality_width.ENGINEERING_SEED == 26090501
    assert locality_width.TARGET_LAYER == 7
    assert locality_width.BASELINE_K == 8
    assert locality_width.PRIMARY_WIDTHS == (16, 32)
    assert locality_width.FALLBACK_WIDTH == 64
    assert locality_width.LEARNING_RATE == 1e-3
    assert locality_width.MAX_OPTIMIZER_STEPS == 128
    assert locality_width.MAX_TRAINING_TOKENS == 500_000
    assert locality_width.BATCH_SIZE == 8
    assert locality_width.DIRECT_CAPABILITY_FLOOR == 0.80


def test_k64_runs_only_when_primary_widths_fail_floor() -> None:
    assert locality_width.should_run_fallback({16: {"direct_accuracy": 0.2}, 32: {"direct_accuracy": 0.7}}) is True
    assert locality_width.should_run_fallback({16: {"direct_accuracy": 0.8}, 32: {"direct_accuracy": 0.1}}) is False
    assert locality_width.should_run_fallback({16: {"direct_accuracy": 0.1}, 32: {"direct_accuracy": 0.9}}) is False
    with pytest.raises(ValueError, match="complete K=16 and K=32"):
        locality_width.should_run_fallback({16: {"direct_accuracy": 0.1}})


def test_nested_width_guard_requires_strict_prefix_extensions() -> None:
    baseline = {"selected": tuple(f"L7:E0:C{i}" for i in range(8))}
    selected16 = tuple(list(baseline["selected"]) + [f"L7:E1:C{i}" for i in range(8)])
    selected32 = tuple(list(selected16) + [f"L7:E2:C{i}" for i in range(16)])
    results = {
        16: {"allocation": {"selected": list(selected16)}},
        32: {"allocation": {"selected": list(selected32)}},
    }
    locality_width._assert_nested_widths(results, baseline)
    bad = dict(results)
    corrupted = list(selected32)
    corrupted[0], corrupted[1] = corrupted[1], corrupted[0]
    bad[32] = {"allocation": {"selected": corrupted}}
    with pytest.raises(RuntimeError, match="LOCALITY_ALLOCATION_DRIFT"):
        locality_width._assert_nested_widths(bad, baseline)


def test_diagnostic_reuses_l7_k8_and_changes_only_selected_width() -> None:
    source = inspect.getsource(locality_width.run_locality_width_diagnostic)
    worker = inspect.getsource(locality_width._run_one_width)
    assert '"causal_variable": "selected_cell_width_k_only"' in source
    assert '"target_layer": TARGET_LAYER' in source
    assert '"loss": "answer-token-causal-cross-entropy"' in source
    assert '"optimizer": "AdamW"' in source
    assert '"routing": "inherited_parent_router"' in source
    assert '"evaluation": "A_eval_greedy_exact"' in source
    assert "allocation.selected[: int(width)]" in worker
    assert "selected[:BASELINE_K]" in worker
    assert "LOCALITY_ALLOCATION_DRIFT" in worker


def test_width_diagnostic_is_engineering_only() -> None:
    source = inspect.getsource(locality_width)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source


def test_decision_taxonomy_distinguishes_rescue_improvement_and_no_improvement() -> None:
    baseline = {
        "direct_accuracy": 0.046875,
        "effective_count": 37.0,
        "topk_mass": {"8": 0.36},
    }
    rescued = {
        16: {"direct_accuracy": 0.2, "allocation": {"gradient_mass_at_k": 0.5, "effective_count": 37.0}, "training": {"final_loss": 1.5}},
        32: {"direct_accuracy": 0.85, "allocation": {"gradient_mass_at_k": 0.8, "effective_count": 37.0}, "training": {"final_loss": 0.8}},
    }
    status, values = locality_width._classify(rescued, baseline)
    assert status == "LOCALITY_WIDTH_RESCUES_LOCAL_CELL_MUTATION"
    assert values["rescued"] is True

    improved = {
        16: {"direct_accuracy": 0.1, "allocation": {"gradient_mass_at_k": 0.5, "effective_count": 37.0}, "training": {"final_loss": 1.5}},
        32: {"direct_accuracy": 0.2, "allocation": {"gradient_mass_at_k": 0.8, "effective_count": 37.0}, "training": {"final_loss": 1.2}},
    }
    status, values = locality_width._classify(improved, baseline)
    assert status == "LOCALITY_WIDTH_IMPROVES_BUT_DOES_NOT_RESCUE"
    assert values["rescued"] is False
    assert values["improved"] is True

    flat = {
        16: {"direct_accuracy": 0.01, "allocation": {"gradient_mass_at_k": 0.5, "effective_count": 37.0}, "training": {"final_loss": 1.5}},
        32: {"direct_accuracy": 0.04, "allocation": {"gradient_mass_at_k": 0.8, "effective_count": 37.0}, "training": {"final_loss": 1.3}},
    }
    status, values = locality_width._classify(flat, baseline)
    assert status == "LOCALITY_WIDTH_DID_NOT_IMPROVE"
    assert values["improved"] is False
