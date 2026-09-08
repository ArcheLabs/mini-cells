"""Static/unit guards for PCU-HYBRID-OBJECTIVE-001."""

from __future__ import annotations

import inspect

from minicells.pcu_kill_001 import hybrid_objective


def test_hybrid_experiment_holds_all_non_objective_variables_fixed() -> None:
    assert hybrid_objective.ENGINEERING_SEED == 26090501
    assert hybrid_objective.TARGET_LAYER == 7
    assert hybrid_objective.TARGET_K == 64
    assert hybrid_objective.LEARNING_RATE == 1e-3
    assert hybrid_objective.MAX_OPTIMIZER_STEPS == 128
    assert hybrid_objective.MAX_TRAINING_TOKENS == 500_000
    assert hybrid_objective.BATCH_SIZE == 8
    assert hybrid_objective.CANDIDATE_POOL_SIZE == 16
    assert hybrid_objective.RANKING_TEMPERATURE == 1.0
    assert hybrid_objective.CE_WEIGHT == 0.25
    assert hybrid_objective.ASSOCIATION_FLOOR == 0.80
    assert hybrid_objective.DIRECT_CAPABILITY_FLOOR == 0.80


def test_hybrid_uses_original_ce_not_ranking_score_proxy() -> None:
    source = inspect.getsource(hybrid_objective._train_hybrid_branch)
    ranking = inspect.getsource(hybrid_objective._hybrid_ranking_loss_for_sample)
    assert "build_task_sequences" in source
    assert "answer_token_cross_entropy" in source
    assert "(float(CE_WEIGHT) * ce_loss).backward()" in source
    assert "(rank_loss / float(config.batch_size)).backward()" in source
    assert "_candidate_scores_tensor" in ranking
    assert "CE regularizer is intentionally not derived from these scores" in ranking


def test_hybrid_reuses_exact_k64_and_never_reallocates() -> None:
    source = inspect.getsource(hybrid_objective.run_hybrid_objective_diagnostic)
    module = inspect.getsource(hybrid_objective)
    assert '"selected_cells": list(baseline.selected_cells)' in source
    assert "baseline.selected_cells" in source
    assert "HYBRID_OBJECTIVE_ALLOCATION_DRIFT" in source
    assert "allocate_topk" not in module
    assert "full_model_task_conditioned_allocation" not in module


def test_hybrid_design_changes_only_ce_readout_regularizer() -> None:
    source = inspect.getsource(hybrid_objective.run_hybrid_objective_diagnostic)
    assert '"causal_variable": "ce_readout_regularizer_weight_only"' in source
    assert '"from": "ranking_only"' in source
    assert '"to": "ranking_plus_original_answer_token_ce"' in source
    assert '"ce_weight": float(CE_WEIGHT)' in source
    assert '"ce_encoding": "original_task_sequence_encoding"' in source


def test_hybrid_decision_requires_both_gates_for_rescue() -> None:
    classify = hybrid_objective._classify
    assert classify(ranking_accuracy=0.90, direct_accuracy=0.90) == (
        "HYBRID_OBJECTIVE_RESCUES_ASSOCIATION_AND_GENERATION"
    )
    assert classify(ranking_accuracy=0.90, direct_accuracy=0.50) == (
        "HYBRID_OBJECTIVE_PRESERVES_ASSOCIATION_GENERATION_UNRESOLVED"
    )
    assert classify(ranking_accuracy=0.50, direct_accuracy=0.90) == (
        "HYBRID_OBJECTIVE_RESCUES_GENERATION_ASSOCIATION_REGRESSED"
    )
    assert classify(ranking_accuracy=0.50, direct_accuracy=0.50) == (
        "HYBRID_OBJECTIVE_DID_NOT_JOINTLY_RESCUE"
    )


def test_hybrid_is_engineering_only() -> None:
    source = inspect.getsource(hybrid_objective)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source
