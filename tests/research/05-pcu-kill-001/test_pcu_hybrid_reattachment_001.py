"""Static/unit guards for PCU-HYBRID-REATTACHMENT-001 protocol v3."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

from minicells.pcu_kill_001 import hybrid_reattachment_v3 as experiment


class _ToyRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta_weight = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        self.register_buffer("parent_weight", torch.tensor([5.0, 7.0, 11.0]))


def test_alpha_intervention_scales_only_delta_and_restores_exactly() -> None:
    runtime = _ToyRuntime()
    before = runtime.delta_weight.detach().clone()
    with experiment.temporarily_scale_cell_deltas(runtime, 0.25) as identity:
        assert torch.equal(runtime.delta_weight, before * 0.25)
        assert identity["alpha"] == 0.25
        assert identity["scaled_sha256"] != identity["trained_sha256"]
    assert torch.equal(runtime.delta_weight, before)


def test_primary_classifier_uses_same_graph_gate_not_native_g0_drift() -> None:
    signature = inspect.signature(experiment.classify_primary_v3)
    assert "same_graph_equivalence_max_abs" in signature.parameters
    assert "native_g0" not in signature.parameters
    supported = experiment.classify_primary_v3(
        replay_matches=True,
        same_graph_equivalence_max_abs=0.0,
        restoration_max_abs=0.0,
        ranking_on=0.82,
        ranking_off=0.06,
        margin_gain=1.0,
        control_nll_increase=0.01,
    )
    assert supported == "ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED"


def test_primary_classifier_reports_locality_failure_without_erasing_causal_support() -> None:
    status = experiment.classify_primary_v3(
        replay_matches=True,
        same_graph_equivalence_max_abs=0.0,
        restoration_max_abs=0.0,
        ranking_on=0.8203125,
        ranking_off=0.0625,
        margin_gain=3.18,
        control_nll_increase=0.319,
    )
    assert status == "CAUSAL_HYBRID_CONSUMPTION_SUPPORTED_LOCALITY_FAILED"


def test_same_graph_and_restoration_fail_closed() -> None:
    bad_equivalence = experiment.classify_primary_v3(
        replay_matches=True,
        same_graph_equivalence_max_abs=experiment.EQUIVALENCE_MAX_ABS_LOGIT_DIFF * 2,
        restoration_max_abs=0.0,
        ranking_on=0.82,
        ranking_off=0.06,
        margin_gain=1.0,
        control_nll_increase=0.0,
    )
    assert bad_equivalence == "SAME_GRAPH_ZERO_STATE_EQUIVALENCE_FAILED"
    bad_restore = experiment.classify_primary_v3(
        replay_matches=True,
        same_graph_equivalence_max_abs=0.0,
        restoration_max_abs=experiment.RESTORATION_MAX_ABS_LOGIT_DIFF * 2,
        ranking_on=0.82,
        ranking_off=0.06,
        margin_gain=1.0,
        control_nll_increase=0.0,
    )
    assert bad_restore == "REVERSIBILITY_FAILED"


def test_amplitude_grid_and_thresholds_are_frozen() -> None:
    assert experiment.PROTOCOL_VERSION == 3
    assert experiment.ALPHA_SWEEP == (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    assert experiment.EQUIVALENCE_MAX_ABS_LOGIT_DIFF == 1e-5
    assert experiment.RESTORATION_MAX_ABS_LOGIT_DIFF == 1e-5
    assert experiment.ASSOCIATION_FLOOR == 0.80
    assert experiment.MIN_CAUSAL_RANKING_GAIN == 0.50
    assert experiment.MAX_CONTROL_ANSWER_NLL_INCREASE == 0.10


def test_sweep_classifier_requires_positive_alpha_joint_pass() -> None:
    rows = [
        {"alpha": 0.0, "joint_pass": False},
        {"alpha": 0.5, "joint_pass": True},
        {"alpha": 1.0, "joint_pass": False},
    ]
    assert experiment.classify_sweep(
        rows, replay_matches=True, restoration_exact=True
    ) == "AMPLITUDE_SWEEP_FINDS_LOCALITY_COMPATIBLE_POINT"
    rows[1]["joint_pass"] = False
    assert experiment.classify_sweep(
        rows, replay_matches=True, restoration_exact=True
    ) == "AMPLITUDE_SWEEP_NO_LOCALITY_COMPATIBLE_POINT"


def test_v3_reuses_ranking_only_mutation_without_bridge_router_or_extra_sweep_training() -> None:
    primary = inspect.getsource(experiment.run_primary_arm)
    sweep = inspect.getsource(experiment.run_amplitude_sweep_arm)
    replay = inspect.getsource(experiment.replay_published_mutation)
    assert "_train_ranking_branch" in replay
    assert "baseline.selected_cells" in replay
    assert "temporarily_zero_cell_deltas" in primary
    assert "temporarily_scale_cell_deltas" in sweep
    assert '"additional_training_after_replay": False' in sweep
    module = inspect.getsource(experiment)
    assert "_train_hybrid_branch" not in module
    assert "allocate_topk" not in module
    assert "full_model_task_conditioned_allocation" not in module


def test_v3_explicitly_records_native_g0_as_non_gating_diagnostic() -> None:
    source = inspect.getsource(experiment.run_primary_arm)
    assert '"native_G0_is_diagnostic_only": True' in source
    assert '"gate_definition": "PARENT_ZERO_DELTA_vs_CELL_OFF_same_cellular_graph"' in source
    assert "parent_vs_off_A" in source
    assert "parent_vs_off_B" in source


def test_dual_gpu_runner_and_publisher_guards_exist() -> None:
    root = Path(__file__).resolve().parents[3]
    runner = (root / "scripts/research/run_pcu_hybrid_reattachment_001.py").read_text(encoding="utf-8")
    publisher = (root / "scripts/research/publish_pcu_hybrid_reattachment_001.py").read_text(encoding="utf-8")
    assert "torch.cuda.device_count() < 2" in runner
    assert '"cuda:0": "primary_causal_reattachment"' in runner
    assert '"cuda:1": "amplitude_sweep"' in runner
    assert "RESERVED_UNTOUCHED" in publisher
    assert '".png"' in publisher
    assert "thresholds_changed" in publisher
