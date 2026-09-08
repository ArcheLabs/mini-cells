"""Static/unit guards for PCU-OBJECTIVE-ALIGNMENT-001."""

from __future__ import annotations

import inspect

import torch

from minicells.pcu_kill_001 import objective_alignment


def test_final_objective_holds_non_objective_variables_fixed() -> None:
    assert objective_alignment.ENGINEERING_SEED == 26090501
    assert objective_alignment.TARGET_LAYER == 7
    assert objective_alignment.TARGET_K == 64
    assert objective_alignment.CANDIDATE_POOL_SIZE == 16
    assert objective_alignment.LEARNING_RATE == 1e-3
    assert objective_alignment.MAX_OPTIMIZER_STEPS == 128
    assert objective_alignment.MAX_TRAINING_TOKENS == 500_000
    assert objective_alignment.BATCH_SIZE == 8
    assert objective_alignment.ASSOCIATION_FLOOR == 0.80
    assert objective_alignment.DIRECT_CAPABILITY_FLOOR == 0.80


def test_ranking_diagnostic_matches_oracle_tie_break_semantics() -> None:
    candidates = ("VBBBB", "VAAAA", "VCCCC")
    scores = torch.tensor([1.0, 1.0, 0.0])
    row = objective_alignment._ranking_diagnostic(candidates, "VBBBB", scores)
    # Equal score is broken lexicographically, matching synthetic._rank_candidate.
    assert row["winner"] == "VAAAA"
    assert row["correct_rank"] == 2
    assert row["exact"] is False
    assert row["correct_margin"] == 0.0


def test_final_decision_separates_association_from_generation() -> None:
    classify = objective_alignment._classify
    assert classify(
        direct_accuracy=0.85,
        eval_ranking_accuracy=0.90,
        ce_baseline_accuracy=0.265625,
    ) == "OBJECTIVE_ALIGNMENT_RESCUES_LOCAL_CELL_MUTATION"
    assert classify(
        direct_accuracy=0.50,
        eval_ranking_accuracy=0.90,
        ce_baseline_accuracy=0.265625,
    ) == "ASSOCIATION_LEARNED_GENERATION_UNRESOLVED"
    assert classify(
        direct_accuracy=0.40,
        eval_ranking_accuracy=0.50,
        ce_baseline_accuracy=0.265625,
    ) == "OBJECTIVE_ALIGNMENT_IMPROVES_BUT_DOES_NOT_RESCUE"
    assert classify(
        direct_accuracy=0.20,
        eval_ranking_accuracy=0.50,
        ce_baseline_accuracy=0.265625,
    ) == "OBJECTIVE_ALIGNMENT_DID_NOT_RESCUE"


def test_final_experiment_reuses_exact_k64_cells_and_never_reallocates() -> None:
    source = inspect.getsource(objective_alignment.run_objective_alignment_diagnostic)
    training = inspect.getsource(objective_alignment._train_ranking_branch)
    module = inspect.getsource(objective_alignment)
    assert '"causal_variable": "training_objective_only"' in source
    assert '"target_layer": TARGET_LAYER' in source
    assert '"selected_k": TARGET_K' in source
    assert '"selected_cells": list(baseline["selected"])' in source
    # The diagnostic owns the frozen baseline binding; the helper must only
    # consume the already-selected Cell list and must never allocate again.
    assert 'baseline["selected"],' in source
    assert "selected_cells: Sequence[str]" in training
    assert "_selected_map(selected_cells, TARGET_LAYER)" in training
    assert "full_model_task_conditioned_allocation" not in training
    assert "allocate_topk" not in training
    assert "full_model_task_conditioned_allocation" not in module
    assert "allocate_topk" not in module
    assert "OBJECTIVE_ALIGNMENT_ALLOCATION_DRIFT" in source


def test_ranking_objective_uses_context_oracle_completion_semantics() -> None:
    scorer = inspect.getsource(objective_alignment._candidate_scores_tensor)
    loss = inspect.getsource(objective_alignment._ranking_loss_for_sample)
    assert "_completion_encoding" in inspect.getsource(objective_alignment._prepare_candidate_batch)
    assert "values.mean()" in scorer
    assert "use_cache=False" in scorer
    assert "F.cross_entropy" in loss
    assert "RANKING_TEMPERATURE" in loss
    assert "_candidate_pool" in loss


def test_effective_batch_is_preserved_by_gradient_accumulation() -> None:
    source = inspect.getsource(objective_alignment._train_ranking_branch)
    assert "loss / float(config.batch_size)" in source
    assert "optimizer.step()" in source
    assert "sample_microbatch" in source


def test_final_diagnostic_is_engineering_only() -> None:
    source = inspect.getsource(objective_alignment)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source
