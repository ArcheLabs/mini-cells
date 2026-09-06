"""Static/unit guards for PCU-CROSS-LAYER-READOUT-001."""

from __future__ import annotations

import inspect

from minicells.pcu_kill_001 import cross_layer_readout


def test_cross_layer_design_is_minimal_and_fixed() -> None:
    assert cross_layer_readout.ENGINEERING_SEED == 26090501
    assert cross_layer_readout.ASSOCIATION_LAYER == 7
    assert cross_layer_readout.ASSOCIATION_K == 64
    assert cross_layer_readout.READOUT_LAYER == 23
    assert cross_layer_readout.READOUT_K == 16
    assert cross_layer_readout.READOUT_OBJECTIVE == "answer-token-causal-cross-entropy"
    assert cross_layer_readout.LEARNING_RATE == 1e-3
    assert cross_layer_readout.MAX_OPTIMIZER_STEPS == 128
    assert cross_layer_readout.BATCH_SIZE == 8
    assert cross_layer_readout.SYNERGY_FLOOR == 0.30


def test_l7_is_replayed_then_frozen_before_l23_allocation() -> None:
    source = inspect.getsource(cross_layer_readout.run_cross_layer_readout_diagnostic)
    freeze = inspect.getsource(cross_layer_readout._freeze_l7_runtime)
    assert "_train_hybrid_branch(" in source
    assert "_freeze_l7_runtime(model, runtime7)" in source
    assert source.index("_freeze_l7_runtime(model, runtime7)") < source.index(
        "full_model_task_conditioned_allocation("
    )
    assert "runtime.requires_grad_(False)" in freeze
    assert "model.requires_grad_(False)" in freeze


def test_l23_is_allocated_once_under_frozen_l7_state() -> None:
    source = inspect.getsource(cross_layer_readout.run_cross_layer_readout_diagnostic)
    assert source.count("full_model_task_conditioned_allocation(") == 1
    assert "calibration_rows=CALIBRATION_ROWS" in source
    assert "calibration_batch_size=CALIBRATION_BATCH_SIZE" in source
    assert "selected_l23 = tuple(allocation.selected[:READOUT_K])" in source


def test_same_l23_cells_are_reused_in_control_and_cross_layer_arm() -> None:
    source = inspect.getsource(cross_layer_readout.run_cross_layer_readout_diagnostic)
    control = inspect.getsource(cross_layer_readout._train_l23_only_control)
    assert "selected_l23," in source
    assert "selected_l23=selected_l23" in source
    assert "tuple(l23_only[\"selected_l23\"]) != selected_l23" in source
    assert "selected_l23," in control
    assert "L23_ONLY_CONTROL_ALLOCATION_DRIFT" in control
    assert "full_model_task_conditioned_allocation" not in control


def test_only_l23_deltas_are_trainable_after_l7_freeze() -> None:
    source = inspect.getsource(cross_layer_readout.run_cross_layer_readout_diagnostic)
    assert "_assert_only_selected_deltas_trainable(model, runtime23)" in source
    control = inspect.getsource(cross_layer_readout._train_l23_only_control)
    assert "_assert_only_selected_deltas_trainable(model, runtime23)" in control


def test_cross_layer_success_requires_both_native_readout_and_association() -> None:
    classify = cross_layer_readout._classify
    assert classify(
        l7_direct=0.03,
        l23_only_direct=0.20,
        cross_direct=0.85,
        cross_ranking=0.85,
    ) == "SPARSE_CROSS_LAYER_READOUT_RESCUE_SUPPORTED"
    assert classify(
        l7_direct=0.03,
        l23_only_direct=0.85,
        cross_direct=0.90,
        cross_ranking=0.90,
    ) == "L23_ONLY_READOUT_SUFFICIENT_CROSS_LAYER_NOT_REQUIRED"
    assert classify(
        l7_direct=0.03,
        l23_only_direct=0.20,
        cross_direct=0.85,
        cross_ranking=0.50,
    ) == "CROSS_LAYER_GENERATION_RESCUE_ASSOCIATION_REGRESSED"
    assert classify(
        l7_direct=0.03,
        l23_only_direct=0.20,
        cross_direct=0.60,
        cross_ranking=0.85,
    ) == "CROSS_LAYER_READOUT_IMPROVES_BUT_DOES_NOT_RESCUE"
    assert classify(
        l7_direct=0.03,
        l23_only_direct=0.20,
        cross_direct=0.10,
        cross_ranking=0.85,
    ) == "MINIMAL_L23_READOUT_DID_NOT_HELP"


def test_experiment_is_engineering_only() -> None:
    source = inspect.getsource(cross_layer_readout)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source
