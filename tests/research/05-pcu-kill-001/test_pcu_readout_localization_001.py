"""Static/unit guards for PCU-READOUT-LOCALIZATION-001."""

from __future__ import annotations

import inspect

from minicells.pcu_kill_001 import readout_localization


def test_readout_localization_replays_exact_published_hybrid() -> None:
    assert readout_localization.ENGINEERING_SEED == 26090501
    assert readout_localization.TARGET_LAYER == 7
    assert readout_localization.TARGET_K == 64
    assert readout_localization.CE_WEIGHT == 0.25
    assert readout_localization.LEARNING_RATE == 1e-3
    assert readout_localization.MAX_OPTIMIZER_STEPS == 128
    assert readout_localization.BATCH_SIZE == 8
    assert readout_localization.EXPECTED_HYBRID_RANKING_TRAIN == 1.0
    assert readout_localization.EXPECTED_HYBRID_RANKING_EVAL == 0.8359375
    assert readout_localization.EXPECTED_HYBRID_DIRECT == 0.03125
    assert readout_localization.HYBRID_SCIENTIFIC_SOURCE_COMMIT == (
        "0241475a387a9114415cf7ed143670dd5c7e1b3b"
    )
    assert readout_localization.HYBRID_CORE_BLOB_SHA == (
        "851c77cdd283def0698ebe721ea8bf216f5ed556"
    )


def test_readout_localization_is_observational_after_hybrid_replay() -> None:
    module = inspect.getsource(readout_localization)
    runner = inspect.getsource(readout_localization.run_readout_localization_diagnostic)
    assert '"causal_variable": "none_observational_readout_localization"' in runner
    assert '"training_changed": False' in runner
    assert "_train_hybrid_branch(" in runner
    assert "torch.optim" not in module
    assert ".backward(" not in module
    assert "optimizer.step" not in module
    assert "allocate_topk" not in module
    assert "full_model_task_conditioned_allocation" not in module


def test_gold_prefix_diagnostic_measures_full_vocab_target_rank() -> None:
    source = inspect.getsource(readout_localization.gold_prefix_token_readout)
    assert "target_rank = 1 + int((token_logits > target_logit).sum())" in source
    assert "top1_id = int(token_logits.argmax())" in source
    assert '"first_token_top1_accuracy"' in source
    assert '"later_token_top1_accuracy"' in source
    assert '"sequence_all_tokens_top1_accuracy"' in source


def test_forced_prefix_diagnostic_actually_greedy_generates_suffix() -> None:
    source = inspect.getsource(readout_localization.forced_prefix_recovery)
    helper = inspect.getsource(readout_localization._greedy_suffix_batch)
    assert "case[\"prompt_ids\"] + case[\"answer_ids\"][:forced]" in source
    assert "model.generate(**kwargs)" in helper
    assert '"do_sample": False' in helper
    assert '"use_cache": True' in helper
    assert "produced == expected" in source
    assert '"minimal_forced_tokens_reaching_floor"' in source


def test_readout_decision_separates_three_structural_cases() -> None:
    classify = readout_localization._classify
    assert classify(
        first_token_top1=0.20,
        later_token_top1=0.95,
        force1_suffix=0.90,
        force2_suffix=0.95,
    ) == "FIRST_TOKEN_READOUT_BOTTLENECK_SUPPORTED"
    assert classify(
        first_token_top1=0.40,
        later_token_top1=0.95,
        force1_suffix=0.50,
        force2_suffix=0.90,
    ) == "EARLY_TOKEN_READOUT_BOTTLENECK_SUPPORTED"
    assert classify(
        first_token_top1=0.90,
        later_token_top1=0.95,
        force1_suffix=0.50,
        force2_suffix=0.60,
    ) == "AUTOREGRESSIVE_TRAJECTORY_INSTABILITY_SUPPORTED"
    assert classify(
        first_token_top1=0.90,
        later_token_top1=0.60,
        force1_suffix=0.50,
        force2_suffix=0.50,
    ) == "SINGLE_LAYER_GOLD_PREFIX_READOUT_INADEQUATE"


def test_readout_localization_is_engineering_only() -> None:
    source = inspect.getsource(readout_localization)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source
