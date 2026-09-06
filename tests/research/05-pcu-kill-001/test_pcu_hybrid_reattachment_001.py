"""Static/unit guards for PCU-HYBRID-REATTACHMENT-001."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from minicells.pcu_kill_001 import hybrid_reattachment as experiment


class _ToyRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta_weight = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        self.register_buffer("parent_weight", torch.tensor([5.0, 7.0, 11.0]))


def test_delta_intervention_is_exact_and_reversible() -> None:
    runtime = _ToyRuntime()
    before = runtime.delta_weight.detach().clone()
    before_sha = experiment.delta_sha256(runtime)
    with experiment.temporarily_zero_cell_deltas(runtime) as identity:
        assert torch.count_nonzero(runtime.delta_weight) == 0
        assert identity["trained_sha256"] == before_sha
        assert identity["zero_sha256"] != before_sha
    assert torch.equal(runtime.delta_weight, before)
    assert experiment.delta_sha256(runtime) == before_sha


def test_logit_diff_is_strict_and_shape_checked() -> None:
    left = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    right = left.clone()
    right[0, 1, 0] += 0.25
    diff = experiment.compare_logits(left, right)
    assert diff.max_abs == 0.25
    assert diff.mean_abs == 0.0625
    assert diff.elements == 4


def test_classifier_requires_causal_gain_equivalence_reversibility_and_locality() -> None:
    supported = experiment.classify_reattachment(
        replay_matches=True,
        equivalence_max_abs=0.0,
        off_equivalence_max_abs=0.0,
        restoration_max_abs=0.0,
        ranking_on=0.82,
        ranking_off=0.06,
        margin_gain=1.0,
        control_nll_increase=0.01,
    )
    assert supported == "ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED"

    no_gain = experiment.classify_reattachment(
        replay_matches=True,
        equivalence_max_abs=0.0,
        off_equivalence_max_abs=0.0,
        restoration_max_abs=0.0,
        ranking_on=0.82,
        ranking_off=0.82,
        margin_gain=0.0,
        control_nll_increase=0.0,
    )
    assert no_gain == "NO_CAUSAL_EXPRESSION_ENGINEERING"

    bad_equivalence = experiment.classify_reattachment(
        replay_matches=True,
        equivalence_max_abs=experiment.EQUIVALENCE_MAX_ABS_LOGIT_DIFF * 2,
        off_equivalence_max_abs=0.0,
        restoration_max_abs=0.0,
        ranking_on=0.82,
        ranking_off=0.06,
        margin_gain=1.0,
        control_nll_increase=0.0,
    )
    assert bad_equivalence == "ZERO_STATE_EQUIVALENCE_FAILED"


def test_experiment_reuses_exact_ranking_only_k64_without_new_bridge() -> None:
    source = inspect.getsource(experiment.run_hybrid_reattachment_diagnostic)
    module = inspect.getsource(experiment)
    assert "_load_baselines" in source
    assert "_train_ranking_branch" in source
    assert "baseline.selected_cells" in source
    assert "temporarily_zero_cell_deltas(runtime)" in source
    assert '"objective": "16-way-candidate-ranking-only"' in source
    assert '"ce_readout_regularizer": False' in source
    assert '"new_bridge": False' in source
    assert "_train_hybrid_branch" not in module
    assert "allocate_topk" not in module
    assert "full_model_task_conditioned_allocation" not in module


def test_source_is_the_association_learned_generation_unresolved_artifact() -> None:
    assert experiment.PUBLISHED_SOURCE_ROOT == experiment.OBJECTIVE_BASELINE_ROOT
    assert experiment.EXPECTED_PUBLISHED_RANKING_ACCURACY == 0.8203125
    assert experiment.EXPECTED_PUBLISHED_DIRECT_ACCURACY == 0.0


def test_success_is_not_cell_alone_takeover_and_formal_is_not_claimed() -> None:
    source = inspect.getsource(experiment.run_hybrid_reattachment_diagnostic)
    assert '"cell_alone_takeover_required": False' in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source
    assert '"formal_decision": "RESERVED_UNRUN"' in source


def test_primary_thresholds_are_predeclared_and_strong() -> None:
    assert experiment.TARGET_LAYER == 7
    assert experiment.TARGET_K == 64
    assert experiment.ENGINEERING_SEED == 26090501
    assert experiment.ASSOCIATION_FLOOR == 0.80
    assert experiment.MIN_CAUSAL_RANKING_GAIN == 0.50
    assert experiment.EQUIVALENCE_MAX_ABS_LOGIT_DIFF == 1e-5
    assert experiment.RESTORATION_MAX_ABS_LOGIT_DIFF == 1e-5
    assert experiment.MAX_CONTROL_ANSWER_NLL_INCREASE == 0.10
