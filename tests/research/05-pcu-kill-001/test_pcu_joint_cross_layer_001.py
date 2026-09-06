"""Static/unit guards for PCU-JOINT-CROSS-LAYER-001."""

from __future__ import annotations

import inspect

from minicells.pcu_kill_001 import joint_cross_layer


def test_joint_design_is_exact_depth3_topology() -> None:
    assert joint_cross_layer.ENGINEERING_SEED == 26090501
    assert joint_cross_layer.TRANSPORT_LAYER == 15
    assert joint_cross_layer.READOUT_LAYER == 23
    assert joint_cross_layer.TRANSPORT_K == 16
    assert joint_cross_layer.READOUT_K == 16
    assert joint_cross_layer.PRIMARY_STEPS == 128
    assert joint_cross_layer.SECONDARY_STEPS == 256
    assert joint_cross_layer.SEQUENTIAL_DIRECT == 0.140625
    assert joint_cross_layer.SEQUENTIAL_RANKING == 0.7890625


def test_published_depth3_cells_are_reused_without_reallocation() -> None:
    source = inspect.getsource(joint_cross_layer.run_joint_arm)
    assert "load_published_depth3(" in source
    assert "depth3.selected_l15" in source
    assert "depth3.selected_l23" in source
    assert "_selected_map(selected, layer)" in source
    assert "full_model_task_conditioned_allocation" not in source
    assert "allocate_topk" not in source


def test_l7_is_replayed_and_frozen_before_joint_layers_exist() -> None:
    source = inspect.getsource(joint_cross_layer.run_joint_arm)
    assert "_train_hybrid_branch(" in source
    assert "_freeze_l7_runtime(model, runtime7)" in source
    assert source.index("_freeze_l7_runtime(model, runtime7)") < source.index(
        "for layer, selected in ("
    )


def test_both_l15_and_l23_are_trainable_in_one_optimizer() -> None:
    source = inspect.getsource(joint_cross_layer._train_joint)
    assert "_assert_joint_trainable_only(model, (runtime15, runtime23))" in source
    assert "torch.optim.AdamW(parameters" in source
    assert "loss.backward()" in source
    assert "optimizer.step()" in source
    guard = inspect.getsource(joint_cross_layer._assert_joint_trainable_only)
    assert "selected_delta_parameters(runtime)" in guard
    assert "observed != allowed" in guard


def test_primary_arm_is_per_parameter_update_matched() -> None:
    source = inspect.getsource(joint_cross_layer.aggregate_joint)
    assert '"primary_arm": "joint128_per_parameter_update_matched"' in source
    assert '"secondary_arm": "joint256_extra_joint_updates_diagnostic"' in source
    classify = joint_cross_layer.classify_joint
    primary = {"metrics": {"direct_accuracy": 0.85, "ranking_eval_accuracy": 0.85}}
    secondary = {"metrics": {"direct_accuracy": 0.20, "ranking_eval_accuracy": 0.85}}
    assert classify(primary, secondary) == "JOINT_COORDINATION_RESCUES_NATIVE_GENERATION"


def test_secondary_rescue_does_not_prove_coordination_alone() -> None:
    classify = joint_cross_layer.classify_joint
    primary = {"metrics": {"direct_accuracy": 0.10, "ranking_eval_accuracy": 0.85}}
    secondary = {"metrics": {"direct_accuracy": 0.85, "ranking_eval_accuracy": 0.85}}
    assert classify(primary, secondary) == "EXTRA_JOINT_UPDATES_RESCUE_COORDINATION_ALONE_UNPROVEN"


def test_joint_improvement_requires_association_retention_for_clean_claim() -> None:
    classify = joint_cross_layer.classify_joint
    primary = {"metrics": {"direct_accuracy": 0.20, "ranking_eval_accuracy": 0.70}}
    secondary = {"metrics": {"direct_accuracy": 0.10, "ranking_eval_accuracy": 0.85}}
    assert classify(primary, secondary) == "JOINT_GENERATION_IMPROVES_ASSOCIATION_REGRESSED"


def test_experiment_is_engineering_only() -> None:
    source = inspect.getsource(joint_cross_layer)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source
