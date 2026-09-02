from __future__ import annotations

import torch

from minicells.native_clm_m2r0 import project_realized_updates_, snapshot_cell_weights
from minicells.native_clm_m2r0b import (
    M2R0BNumericalThresholds,
    classify_numerical_reference,
    diagnose_optimizer_mechanics,
    measure_numerical_reference,
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
        cell.weight.fill_(0.125)
    return model


def _safe_delta(scale: float) -> torch.Tensor:
    delta = torch.zeros(4, 4)
    delta[0, 0] = 2.0 * scale
    delta[0, 1] = -1.0 * scale
    return delta


def test_tiny_safe_parameter_transaction_is_explained_by_machine_floor() -> None:
    model = _model()
    before = snapshot_cell_weights(model)
    proposal = _safe_delta(1e-6)
    raw_after = [before[0] + proposal]
    with torch.no_grad():
        model.cellular.cells[0].weight.copy_(raw_after[0])

    rows = measure_numerical_reference(
        model,
        before,
        raw_after,
        [proposal],
        arm="sgd_no_decay_grad_projection",
        step=1,
        roundoff_bound_multiplier=8.0,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["eligible_for_ratio"] is True
    assert row["matched_safe_ideal_rho"] < 1e-10
    assert row["committed_excess_factor"] <= 2.0


def test_structural_violation_is_far_above_machine_floor() -> None:
    model = _model()
    before = snapshot_cell_weights(model)
    bad = torch.zeros(4, 4)
    bad[0, 0] = 1e-3
    bad[0, 1] = 2e-3
    raw_after = [before[0] + bad]
    with torch.no_grad():
        model.cellular.cells[0].weight.copy_(raw_after[0])

    rows = measure_numerical_reference(
        model,
        before,
        raw_after,
        [_safe_delta(1e-6)],
        arm="bad",
        step=1,
        roundoff_bound_multiplier=8.0,
    )
    row = rows[0]
    assert row["committed_rho"] > 0.1
    assert row["committed_excess_factor"] > 16.0


def test_realized_update_projection_reduces_commit_to_machine_floor() -> None:
    model = _model()
    before = snapshot_cell_weights(model)
    bad = torch.zeros(4, 4)
    bad[0, 0] = 1e-3
    bad[0, 1] = 2e-3
    with torch.no_grad():
        model.cellular.cells[0].weight.add_(bad)
    raw_after = snapshot_cell_weights(model)
    retained = project_realized_updates_(model, before)

    rows = measure_numerical_reference(
        model,
        before,
        raw_after,
        [_safe_delta(1e-6)],
        arm="adamw_final_update_projection",
        step=1,
        roundoff_bound_multiplier=8.0,
    )
    assert 0.0 <= retained[0] <= 1.0
    assert rows[0]["optimizer_raw_rho"] > 0.1
    assert rows[0]["committed_excess_factor"] <= 2.0


def _summary(
    *,
    excess_p95: float,
    excess_max: float,
    rho_p95: float,
    ideal_max: float = 1e-14,
    n: int = 200,
) -> dict[str, float | int]:
    return {
        "audited_cell_updates": n,
        "matched_safe_ideal_rho_max": ideal_max,
        "committed_excess_factor_p95": excess_p95,
        "committed_excess_factor_max": excess_max,
        "committed_rho_p95": rho_p95,
    }


def _passing_reference() -> dict[str, float | int]:
    return _summary(excess_p95=0.8, excess_max=1.2, rho_p95=1e-6)


def test_numerical_reference_closure_and_mechanics_diagnosis() -> None:
    thresholds = M2R0BNumericalThresholds()
    summaries = {
        "current_adamw_grad_projection": _summary(
            excess_p95=1000.0, excess_max=2000.0, rho_p95=0.15
        ),
        "adamw_no_decay_grad_projection": _summary(
            excess_p95=800.0, excess_max=1600.0, rho_p95=0.15
        ),
        "sgd_no_decay_grad_projection": _passing_reference(),
        "sgd_with_decay_grad_projection": _summary(
            excess_p95=100.0, excess_max=300.0, rho_p95=0.4
        ),
        "adamw_final_update_projection": _passing_reference(),
    }
    assert (
        classify_numerical_reference(summaries, thresholds)
        == "R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF"
    )
    assert (
        diagnose_optimizer_mechanics(summaries, thresholds)
        == "BOTH_PRECONDITIONER_AND_WEIGHT_DECAY_BREAK_UPDATE_INVARIANT"
    )


def test_failed_sgd_reference_stays_inconclusive() -> None:
    thresholds = M2R0BNumericalThresholds()
    summaries = {
        "current_adamw_grad_projection": _passing_reference(),
        "adamw_no_decay_grad_projection": _passing_reference(),
        "sgd_no_decay_grad_projection": _summary(
            excess_p95=3.0, excess_max=5.0, rho_p95=0.01
        ),
        "sgd_with_decay_grad_projection": _passing_reference(),
        "adamw_final_update_projection": _passing_reference(),
    }
    assert (
        classify_numerical_reference(summaries, thresholds)
        == "INCONCLUSIVE_SGD_NUMERICAL_REFERENCE_FAILURE"
    )
    assert diagnose_optimizer_mechanics(summaries, thresholds) is None


def test_failed_final_projection_reference_stays_inconclusive() -> None:
    thresholds = M2R0BNumericalThresholds()
    summaries = {
        "current_adamw_grad_projection": _passing_reference(),
        "adamw_no_decay_grad_projection": _passing_reference(),
        "sgd_no_decay_grad_projection": _passing_reference(),
        "sgd_with_decay_grad_projection": _passing_reference(),
        "adamw_final_update_projection": _summary(
            excess_p95=3.0, excess_max=5.0, rho_p95=1e-5
        ),
    }
    assert (
        classify_numerical_reference(summaries, thresholds)
        == "INCONCLUSIVE_FINAL_PROJECTION_NUMERICAL_REFERENCE_FAILURE"
    )
