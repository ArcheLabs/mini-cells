"""Static/unit guards for the engineering-only PCU layer-placement diagnostic."""

from __future__ import annotations

import inspect

from minicells.pcu_kill_001 import layer_placement


def test_layer_target_rule_selects_depth_fraction_probes() -> None:
    targets = layer_placement.choose_layer_targets(tuple(range(24)))
    assert targets == {"early": 7, "mid": 15, "late_baseline": 23}


def test_layer_target_rule_uses_nearest_available_sparse_layers() -> None:
    targets = layer_placement.choose_layer_targets((1, 5, 9, 13, 17, 21, 23))
    assert targets["late_baseline"] == 23
    assert targets["early"] == 5  # target 7 is tied; deterministic tie-break picks lower layer
    assert targets["mid"] == 13
    assert len(set(targets.values())) == 3


def test_diagnostic_holds_training_variables_fixed() -> None:
    assert layer_placement.ENGINEERING_SEED == 26090501
    assert layer_placement.K == 8
    assert layer_placement.LEARNING_RATE == 1e-3
    assert layer_placement.MAX_OPTIMIZER_STEPS == 128
    assert layer_placement.MAX_TRAINING_TOKENS == 500_000
    assert layer_placement.BATCH_SIZE == 8
    assert layer_placement.CALIBRATION_ROWS == 64
    assert layer_placement.CALIBRATION_BATCH_SIZE == 8
    assert layer_placement.DIRECT_CAPABILITY_FLOOR == 0.80


def test_diagnostic_is_a_only_and_reuses_published_late_baseline() -> None:
    source = inspect.getsource(layer_placement.run_layer_placement_diagnostic)
    assert '"task": "A_only_U_to_V"' in source
    assert '"source": "published_PCU_KILL_001_E0"' in source
    assert '"early": (early, devices[0])' in source
    assert '"mid": (mid, devices[1])' in source
    assert "late_baseline" in source


def test_full_model_training_still_uses_answer_token_ce_and_adamw() -> None:
    loss_source = inspect.getsource(layer_placement._full_task_loss)
    train_source = inspect.getsource(layer_placement._train_full_model_branch)
    allocation_source = inspect.getsource(layer_placement.full_model_task_conditioned_allocation)
    assert "answer_token_cross_entropy" in loss_source
    assert "torch.optim.AdamW" in train_source
    assert "supervised_total" in allocation_source
    assert "calibration_batch_size" in allocation_source
    assert "allocate_topk" in allocation_source


def test_diagnostic_has_no_formal_execution_path() -> None:
    source = inspect.getsource(layer_placement)
    assert "RESERVED_UNTOUCHED" not in source
    assert "mark_formal_seed" not in source
    assert "run_formal" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source
