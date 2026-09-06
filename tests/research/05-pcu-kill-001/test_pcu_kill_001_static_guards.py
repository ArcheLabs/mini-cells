"""Static regression guards for scientific-path wiring.

They complement the numeric toy tests without loading Granite or consuming a
formal seed.
"""

from __future__ import annotations

import inspect

from minicells.pcu_kill_001 import experiment
from minicells.pcu_kill_001 import execution
from minicells.pcu_kill_001.task_training import task_conditioned_allocation


def _scientific_pipeline_source() -> str:
    """Inspect the scientific worker beneath any audit-only runtime wrapper."""
    pipeline = experiment._run_shared_scientific_pipeline
    while hasattr(pipeline, "_pcu_original_pipeline"):
        pipeline = pipeline._pcu_original_pipeline
    return inspect.getsource(pipeline)


def test_scientific_allocation_uses_task_ce_not_random_hidden_energy() -> None:
    source = inspect.getsource(task_conditioned_allocation)
    assert "cached_task_loss" in source
    assert "loss.backward()" in source
    assert "torch.randn" not in source


def test_capacity_ladder_selects_first_passing_k_under_bounded_search() -> None:
    source = _scientific_pipeline_source()
    assert "for k in k_values" in source
    assert 'row["passes"] and selected_k is None' in source
    assert 'direct_a.exact >= 0.80 and direct_b.exact >= 0.80' in source
    assert "ENGINEERING_LR_CANDIDATES" in source
    assert "if frozen_config is not None" in source


def test_g0_guard_precedes_shared_engineering_pipeline() -> None:
    source = inspect.getsource(execution.run_engineering)
    g0_position = source.index("_g0_preflight")
    delegate_position = source.index("_run_engineering_pinned_revision")
    assert g0_position < delegate_position
    assert 'if not g0["passed"]' in source
    assert "_write_g0_failure" in source
    assert "resolved_revision" in source


def test_g0_guard_precedes_shared_formal_pipeline() -> None:
    source = inspect.getsource(execution.run_formal_execution)
    assert source.index("_g0_preflight") < source.index("_experiment.run_formal_execution")
    assert 'if not g0["passed"]' in source
    assert "_write_g0_failure" in source
