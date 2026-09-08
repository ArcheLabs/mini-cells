"""Native CLM v0 M2-R0b numerical-reference audit helpers.

M2-R0 observed that the algebraic SGD/no-decay reference failed a strict relative
rho threshold even though its update is, in exact arithmetic, a scalar multiple of
the certificate-projected gradient. R0b separates three effects:

1. the projected-gradient analytic transaction;
2. finite-precision parameter commit, fl(W + dW) - W;
3. the optimizer-realized/actually committed transaction.

The audit is diagnostic only. It consumes no continual-language formal seeds and
does not change the historical M2 decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .native_clm_v0 import NativeCell, NativeCLM


@dataclass(frozen=True)
class M2R0BNumericalThresholds:
    minimum_audited_cell_updates_per_arm: int = 128
    roundoff_bound_multiplier: float = 8.0
    reference_floor_excess_p95_max: float = 2.0
    reference_floor_excess_max_max: float = 4.0
    ideal_projection_rho_max: float = 1e-10
    material_floor_excess_p95_min: float = 16.0
    material_violation_rho_p95_min: float = 1e-4


def _orthonormal_certificate_rows(cell: NativeCell, *, device: torch.device) -> Tensor:
    """Return an fp64 orthonormal basis for the stored certificate span."""

    rank = cell.rank
    if rank <= 0:
        return torch.empty((0, cell.d_model), device=device, dtype=torch.float64)
    q = cell.certificate_basis[:rank].detach().to(device=device, dtype=torch.float64)
    # The persistent basis is stored in parameter precision. QR removes small
    # orthogonality defects without changing the represented certificate span.
    q_columns = torch.linalg.qr(q.transpose(0, 1), mode="reduced").Q
    return q_columns.transpose(0, 1).contiguous()


def _project_fp64(value: Tensor, q_rows: Tensor) -> Tensor:
    value64 = value.detach().to(device=q_rows.device, dtype=torch.float64)
    if q_rows.numel() == 0:
        return value64
    return value64 - value64.matmul(q_rows.transpose(0, 1)).matmul(q_rows)


def _residual_norm(value64: Tensor, q_rows: Tensor) -> float:
    if q_rows.numel() == 0:
        return 0.0
    residual = value64.matmul(q_rows.transpose(0, 1))
    return float(torch.linalg.vector_norm(residual).item())


def _rho(value64: Tensor, q_rows: Tensor) -> tuple[float, float, float]:
    norm = float(torch.linalg.vector_norm(value64).item())
    residual = _residual_norm(value64, q_rows)
    ratio = residual / (norm + 1e-30) if norm > 0.0 else 0.0
    return norm, residual, ratio


def _simulate_parameter_commit(baseline: Tensor, delta64: Tensor) -> Tensor:
    """Simulate the parameter-dtype transaction fl(W + dW) - W."""

    delta_param = delta64.to(device=baseline.device, dtype=baseline.dtype)
    committed = baseline + delta_param
    return (committed - baseline).to(dtype=torch.float64)


def _roundoff_bound(
    baseline: Tensor,
    safe_delta64: Tensor,
    *,
    multiplier: float,
) -> float:
    if not baseline.dtype.is_floating_point:
        raise TypeError("M2-R0b requires floating-point Cell parameters")
    eps = float(torch.finfo(baseline.dtype).eps)
    baseline_norm = float(torch.linalg.vector_norm(baseline.detach().double()).item())
    delta_norm = float(torch.linalg.vector_norm(safe_delta64).item())
    # Addition and subtraction each round once. The frozen multiplier is a
    # conservative envelope for matrix-wide accumulation of those elementwise errors.
    return float(multiplier) * eps * (baseline_norm + delta_norm + 1e-30)


@torch.no_grad()
def snapshot_cell_gradients(model: NativeCLM, *, lr: float) -> list[Tensor]:
    """Capture the gradient-level analytic transaction -lr*g after projection/clip."""

    updates: list[Tensor] = []
    for cell in model.cellular.cells:
        grad = cell.weight.grad
        if grad is None:
            updates.append(torch.zeros_like(cell.weight))
        else:
            updates.append((-float(lr) * grad.detach()).clone())
    return updates


@torch.no_grad()
def measure_numerical_reference(
    model: NativeCLM,
    before: list[Tensor],
    raw_after_optimizer: list[Tensor],
    gradient_updates: list[Tensor],
    *,
    arm: str,
    step: int,
    roundoff_bound_multiplier: float,
    minimum_update_norm: float = 1e-12,
) -> list[dict[str, Any]]:
    """Measure analytic, float-commit, optimizer, and final-commit invariants.

    `raw_after_optimizer` is captured immediately after optimizer.step() and before
    any realized-update projection. `model` is measured after the arm's optional
    final-update repair, so `committed_delta` is the transaction that would persist.
    """

    expected = model.cell_count
    if not (len(before) == len(raw_after_optimizer) == len(gradient_updates) == expected):
        raise ValueError("M2-R0b snapshots do not match Cell count")
    if minimum_update_norm < 0:
        raise ValueError("minimum_update_norm must be non-negative")
    if roundoff_bound_multiplier <= 0:
        raise ValueError("roundoff_bound_multiplier must be positive")

    rows: list[dict[str, Any]] = []
    for cell_id, (cell, baseline, raw_after, gradient_update) in enumerate(
        zip(
            model.cellular.cells,
            before,
            raw_after_optimizer,
            gradient_updates,
            strict=True,
        )
    ):
        rank = cell.rank
        if rank <= 0:
            continue
        baseline = baseline.to(device=cell.weight.device, dtype=cell.weight.dtype)
        raw_after = raw_after.to(device=cell.weight.device, dtype=cell.weight.dtype)
        q = _orthonormal_certificate_rows(cell, device=cell.weight.device)

        stored_q = cell.certificate_basis[:rank].detach().to(
            device=cell.weight.device, dtype=torch.float64
        )
        gram = stored_q.matmul(stored_q.transpose(0, 1))
        eye = torch.eye(rank, device=gram.device, dtype=gram.dtype)
        basis_orthogonality_error = float(torch.linalg.vector_norm(gram - eye).item())

        gradient64 = gradient_update.detach().to(device=cell.weight.device, dtype=torch.float64)
        gradient_norm, gradient_residual, gradient_rho = _rho(gradient64, q)
        gradient_safe64 = _project_fp64(gradient64, q)
        _, gradient_safe_residual, gradient_safe_rho = _rho(gradient_safe64, q)
        gradient_floor_delta64 = _simulate_parameter_commit(baseline, gradient_safe64)
        _, gradient_floor_residual, gradient_floor_rho = _rho(gradient_floor_delta64, q)

        raw_delta64 = (raw_after - baseline).to(dtype=torch.float64)
        raw_norm, raw_residual, raw_rho = _rho(raw_delta64, q)

        # This matched safe transaction uses the optimizer update's own magnitude and
        # tangent component. It therefore estimates the machine floor at the same
        # scale as the observed optimizer proposal rather than at SGD's scale.
        matched_safe64 = _project_fp64(raw_delta64, q)
        matched_safe_norm, matched_safe_residual, matched_safe_rho = _rho(matched_safe64, q)
        matched_floor_delta64 = _simulate_parameter_commit(baseline, matched_safe64)
        matched_floor_norm, matched_floor_residual, matched_floor_rho = _rho(
            matched_floor_delta64, q
        )
        roundoff_bound = _roundoff_bound(
            baseline,
            matched_safe64,
            multiplier=roundoff_bound_multiplier,
        )
        floor_envelope = max(matched_floor_residual, roundoff_bound, 1e-30)

        committed_delta64 = (cell.weight.detach() - baseline).to(dtype=torch.float64)
        committed_norm, committed_residual, committed_rho = _rho(committed_delta64, q)
        eligible = committed_norm > minimum_update_norm
        excess_factor = committed_residual / floor_envelope if eligible else 0.0

        rows.append(
            {
                "arm": arm,
                "step": int(step),
                "cell_id": int(cell_id),
                "certificate_rank": int(rank),
                "parameter_dtype": str(cell.weight.dtype),
                "machine_epsilon": float(torch.finfo(cell.weight.dtype).eps),
                "basis_orthogonality_error": basis_orthogonality_error,
                "gradient_analytic_norm": gradient_norm,
                "gradient_analytic_violation_norm": gradient_residual,
                "gradient_analytic_rho": gradient_rho,
                "gradient_safe_ideal_violation_norm": gradient_safe_residual,
                "gradient_safe_ideal_rho": gradient_safe_rho,
                "gradient_float_commit_violation_norm": gradient_floor_residual,
                "gradient_float_commit_rho": gradient_floor_rho,
                "optimizer_raw_update_norm": raw_norm,
                "optimizer_raw_violation_norm": raw_residual,
                "optimizer_raw_rho": raw_rho,
                "matched_safe_update_norm": matched_safe_norm,
                "matched_safe_ideal_violation_norm": matched_safe_residual,
                "matched_safe_ideal_rho": matched_safe_rho,
                "matched_safe_float_commit_norm": matched_floor_norm,
                "matched_safe_float_commit_violation_norm": matched_floor_residual,
                "matched_safe_float_commit_rho": matched_floor_rho,
                "roundoff_bound_norm": roundoff_bound,
                "machine_floor_envelope_norm": floor_envelope,
                "committed_update_norm": committed_norm,
                "committed_violation_norm": committed_residual,
                "committed_rho": committed_rho,
                "committed_excess_factor": excess_factor,
                "eligible_for_ratio": bool(eligible),
            }
        )
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def summarize_numerical_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row.get("eligible_for_ratio", True))]

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in eligible]

    def dist(name: str) -> dict[str, float]:
        xs = values(name)
        if not xs:
            return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
        return {
            "mean": sum(xs) / len(xs),
            "p95": _percentile(xs, 0.95),
            "max": max(xs),
        }

    ranks = [int(row["certificate_rank"]) for row in eligible]
    result: dict[str, Any] = {
        "certificate_ranked_cell_steps": len(rows),
        "audited_cell_updates": len(eligible),
        "tiny_or_zero_updates_skipped": len(rows) - len(eligible),
        "certificate_rank_min": min(ranks) if ranks else 0,
        "certificate_rank_max": max(ranks) if ranks else 0,
    }
    for field in (
        "basis_orthogonality_error",
        "gradient_analytic_rho",
        "gradient_safe_ideal_rho",
        "gradient_float_commit_rho",
        "optimizer_raw_rho",
        "matched_safe_ideal_rho",
        "matched_safe_float_commit_rho",
        "committed_rho",
        "committed_excess_factor",
        "optimizer_raw_update_norm",
        "committed_update_norm",
        "matched_safe_float_commit_violation_norm",
        "roundoff_bound_norm",
        "machine_floor_envelope_norm",
    ):
        distribution = dist(field)
        result[f"{field}_mean"] = distribution["mean"]
        result[f"{field}_p95"] = distribution["p95"]
        result[f"{field}_max"] = distribution["max"]
    return result


def _reference_pass(summary: dict[str, Any], thresholds: M2R0BNumericalThresholds) -> bool:
    return (
        int(summary["audited_cell_updates"])
        >= thresholds.minimum_audited_cell_updates_per_arm
        and float(summary["matched_safe_ideal_rho_max"])
        <= thresholds.ideal_projection_rho_max
        and float(summary["committed_excess_factor_p95"])
        <= thresholds.reference_floor_excess_p95_max
        and float(summary["committed_excess_factor_max"])
        <= thresholds.reference_floor_excess_max_max
    )


def _material_violation(
    summary: dict[str, Any], thresholds: M2R0BNumericalThresholds
) -> bool:
    return (
        float(summary["committed_excess_factor_p95"])
        >= thresholds.material_floor_excess_p95_min
        and float(summary["committed_rho_p95"])
        >= thresholds.material_violation_rho_p95_min
    )


def classify_numerical_reference(
    arm_summaries: dict[str, dict[str, Any]],
    thresholds: M2R0BNumericalThresholds,
) -> str:
    required = {
        "current_adamw_grad_projection",
        "adamw_no_decay_grad_projection",
        "sgd_no_decay_grad_projection",
        "sgd_with_decay_grad_projection",
        "adamw_final_update_projection",
    }
    if set(arm_summaries) != required:
        raise ValueError("M2-R0b arm set does not match frozen protocol")
    if not _reference_pass(arm_summaries["sgd_no_decay_grad_projection"], thresholds):
        return "INCONCLUSIVE_SGD_NUMERICAL_REFERENCE_FAILURE"
    if not _reference_pass(arm_summaries["adamw_final_update_projection"], thresholds):
        return "INCONCLUSIVE_FINAL_PROJECTION_NUMERICAL_REFERENCE_FAILURE"
    return "R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF"


def diagnose_optimizer_mechanics(
    arm_summaries: dict[str, dict[str, Any]],
    thresholds: M2R0BNumericalThresholds,
) -> str | None:
    classification = classify_numerical_reference(arm_summaries, thresholds)
    if classification != "R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF":
        return None

    current_bad = _material_violation(
        arm_summaries["current_adamw_grad_projection"], thresholds
    )
    preconditioner_bad = _material_violation(
        arm_summaries["adamw_no_decay_grad_projection"], thresholds
    )
    decay_bad = _material_violation(
        arm_summaries["sgd_with_decay_grad_projection"], thresholds
    )
    if not current_bad:
        return "CURRENT_UPDATE_INVARIANT_HOLDS_AFTER_NUMERICAL_CONTROL"
    if preconditioner_bad and decay_bad:
        return "BOTH_PRECONDITIONER_AND_WEIGHT_DECAY_BREAK_UPDATE_INVARIANT"
    if preconditioner_bad:
        return "ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT"
    if decay_bad:
        return "WEIGHT_DECAY_BREAKS_UPDATE_INVARIANT"
    return "MIXED_OR_INTERACTION_UPDATE_INVARIANT_VIOLATION"
