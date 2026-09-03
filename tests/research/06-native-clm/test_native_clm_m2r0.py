from __future__ import annotations

import math

import torch

from minicells.native_clm_m2r0 import (
    M2R0Thresholds,
    classify_optimizer_invariant,
    measure_realized_update_invariant,
    project_realized_updates_,
    snapshot_cell_weights,
    summarize_invariant_rows,
)
from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig


def _model() -> NativeCLM:
    config = NativeCLMConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=4,
        n_layers=1,
        n_heads=1,
        d_ff=8,
        initial_cells=1,
        active_cells=1,
        cellular_layer_index=0,
        certificate_max_rank=2,
        tie_embeddings=False,
    )
    model = NativeCLM(config)
    cell = model.cellular.cells[0]
    with torch.no_grad():
        q = torch.tensor([1.0, 2.0, 0.0, 0.0])
        q = q / torch.linalg.vector_norm(q)
        cell.certificate_basis[0].copy_(q)
        cell.certificate_rank.fill_(1)
        cell.weight.zero_()
    return model


def test_realized_update_metric_detects_certificate_violation() -> None:
    model = _model()
    before = snapshot_cell_weights(model)
    with torch.no_grad():
        model.cellular.cells[0].weight[0, 0] = 1.0
    rows = measure_realized_update_invariant(model, before, arm="bad", step=1)
    assert len(rows) == 1
    assert rows[0]["eligible_for_ratio"] is True
    assert rows[0]["violation_ratio"] > 0.1


def test_zero_update_is_not_counted_in_ratio_distribution() -> None:
    model = _model()
    before = snapshot_cell_weights(model)
    rows = measure_realized_update_invariant(
        model,
        before,
        arm="zero",
        step=1,
        minimum_update_norm=1e-12,
    )
    summary = summarize_invariant_rows(rows)
    assert len(rows) == 1
    assert rows[0]["eligible_for_ratio"] is False
    assert summary["certificate_ranked_cell_steps"] == 1
    assert summary["audited_cell_updates"] == 0
    assert summary["tiny_or_zero_updates_skipped"] == 1


def test_final_update_projection_restores_nullspace_invariant() -> None:
    model = _model()
    before = snapshot_cell_weights(model)
    with torch.no_grad():
        model.cellular.cells[0].weight[0, :2] = torch.tensor([1.0, -1.0])
    retained = project_realized_updates_(model, before)
    rows = measure_realized_update_invariant(model, before, arm="safe", step=1)
    assert 0.0 < retained[0] <= 1.0
    assert rows[0]["violation_ratio"] < 1e-6


def test_first_adam_step_can_break_projected_gradient_invariant() -> None:
    q = torch.tensor([1.0, 2.0]) / math.sqrt(5.0)
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([2.0, -1.0])
    assert abs(float(torch.dot(parameter.grad, q))) < 1e-7
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    before = parameter.detach().clone()
    optimizer.step()
    delta = parameter.detach() - before
    assert abs(float(torch.dot(delta, q))) > 1e-5


def test_weight_decay_can_break_projected_gradient_invariant() -> None:
    q = torch.tensor([1.0, 2.0]) / math.sqrt(5.0)
    parameter = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
    parameter.grad = torch.tensor([2.0, -1.0])
    assert abs(float(torch.dot(parameter.grad, q))) < 1e-7
    optimizer = torch.optim.SGD([parameter], lr=1e-2, momentum=0.0, weight_decay=0.1)
    before = parameter.detach().clone()
    optimizer.step()
    delta = parameter.detach() - before
    assert abs(float(torch.dot(delta, q))) > 1e-6


def _summary(maximum: float, p95: float, n: int = 200) -> dict[str, float | int]:
    return {
        "audited_cell_updates": n,
        "violation_ratio_max": maximum,
        "violation_ratio_p95": p95,
    }


def _references() -> dict[str, dict[str, float | int]]:
    return {
        "sgd_no_decay_grad_projection": _summary(1e-7, 1e-8),
        "adamw_final_update_projection": _summary(1e-7, 1e-8),
    }


def test_classification_distinguishes_preconditioner_decay_and_both() -> None:
    thresholds = M2R0Thresholds()

    preconditioner = {
        **_references(),
        "current_adamw_grad_projection": _summary(0.1, 0.05),
        "adamw_no_decay_grad_projection": _summary(0.08, 0.04),
        "sgd_with_decay_grad_projection": _summary(1e-7, 1e-8),
    }
    assert (
        classify_optimizer_invariant(preconditioner, thresholds)
        == "ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT"
    )

    decay = {
        **_references(),
        "current_adamw_grad_projection": _summary(0.1, 0.05),
        "adamw_no_decay_grad_projection": _summary(1e-7, 1e-8),
        "sgd_with_decay_grad_projection": _summary(0.08, 0.04),
    }
    assert (
        classify_optimizer_invariant(decay, thresholds)
        == "WEIGHT_DECAY_BREAKS_UPDATE_INVARIANT"
    )

    both = {
        **_references(),
        "current_adamw_grad_projection": _summary(0.1, 0.05),
        "adamw_no_decay_grad_projection": _summary(0.08, 0.04),
        "sgd_with_decay_grad_projection": _summary(0.07, 0.03),
    }
    assert (
        classify_optimizer_invariant(both, thresholds)
        == "BOTH_PRECONDITIONER_AND_WEIGHT_DECAY_BREAK_UPDATE_INVARIANT"
    )


def test_classification_requires_reference_coverage() -> None:
    thresholds = M2R0Thresholds()
    summaries = {
        **_references(),
        "current_adamw_grad_projection": _summary(0.1, 0.05),
        "adamw_no_decay_grad_projection": _summary(0.08, 0.04),
        "sgd_with_decay_grad_projection": _summary(0.07, 0.03),
    }
    summaries["sgd_no_decay_grad_projection"] = _summary(1e-7, 1e-8, n=8)
    assert classify_optimizer_invariant(summaries, thresholds) == "INCONCLUSIVE_REFERENCE_FAILURE"


def test_summary_reports_distribution_over_eligible_updates_only() -> None:
    rows = [
        {
            "violation_ratio": value,
            "update_norm": 1.0,
            "certificate_rank": 4,
            "eligible_for_ratio": True,
        }
        for value in (0.0, 0.1, 0.2, 0.3, 0.4)
    ]
    rows.append(
        {
            "violation_ratio": 0.0,
            "update_norm": 0.0,
            "certificate_rank": 4,
            "eligible_for_ratio": False,
        }
    )
    summary = summarize_invariant_rows(rows)
    assert summary["certificate_ranked_cell_steps"] == 6
    assert summary["audited_cell_updates"] == 5
    assert summary["tiny_or_zero_updates_skipped"] == 1
    assert summary["certificate_rank_min"] == 4
    assert summary["certificate_rank_max"] == 4
    assert abs(float(summary["violation_ratio_median"]) - 0.2) < 1e-12
    assert abs(float(summary["violation_ratio_max"]) - 0.4) < 1e-12
