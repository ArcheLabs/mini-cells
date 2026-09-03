"""Native CLM v0 M2-R0 optimizer update-invariant audit helpers.

M2-R0 does not change the continual-learning algorithm. It measures whether the
*realized parameter delta* produced by an optimizer satisfies the certificate
nullspace invariant that M2 intended to enforce:

    DeltaW Q^T ~= 0.

The distinction matters for optimizers such as AdamW: projecting a raw gradient
is not algebraically equivalent to projecting the complete realized parameter
update after elementwise preconditioning and decoupled weight decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .native_clm_v0 import NativeCell, NativeCLM


@dataclass(frozen=True)
class M2R0Thresholds:
    minimum_audited_cell_updates_per_arm: int = 128
    reference_max_violation_ratio: float = 2e-5
    reference_p95_violation_ratio: float = 5e-6
    material_violation_ratio: float = 1e-4


def _right_nullspace_projection(value: Tensor, cell: NativeCell) -> Tensor:
    """Return value @ (I - Q^T Q) without materializing the identity matrix."""

    if cell.rank <= 0:
        return value
    q = cell.certificate_basis[: cell.rank].to(device=value.device, dtype=value.dtype)
    return value - value.matmul(q.transpose(0, 1)).matmul(q)


@torch.no_grad()
def snapshot_cell_weights(model: NativeCLM) -> list[Tensor]:
    return [cell.weight.detach().clone() for cell in model.cellular.cells]


@torch.no_grad()
def project_realized_updates_(model: NativeCLM, before: list[Tensor]) -> list[float]:
    """Project the complete realized optimizer delta and write it back in-place.

    `before` must be captured immediately before optimizer.step(). The optimizer is
    allowed to produce its normal AdamW proposal (including moment preconditioning
    and decoupled weight decay). We then replace the actual parameter delta with its
    certificate-nullspace projection.

    Optimizer state is deliberately left untouched: M2-R0 audits the parameter
    transaction invariant, not a new optimizer-state design.
    """

    if len(before) != model.cell_count:
        raise ValueError("M2-R0 weight snapshot does not match Cell count")
    retained_ratios: list[float] = []
    for cell, baseline in zip(model.cellular.cells, before, strict=True):
        baseline = baseline.to(device=cell.weight.device, dtype=cell.weight.dtype)
        raw_delta = cell.weight.detach() - baseline
        raw_norm = float(torch.linalg.vector_norm(raw_delta).item())
        safe_delta = _right_nullspace_projection(raw_delta, cell)
        safe_norm = float(torch.linalg.vector_norm(safe_delta).item())
        cell.weight.copy_(baseline + safe_delta)
        retained_ratios.append(1.0 if raw_norm <= 1e-20 else safe_norm / raw_norm)
    return retained_ratios


@torch.no_grad()
def measure_realized_update_invariant(
    model: NativeCLM,
    before: list[Tensor],
    *,
    arm: str,
    step: int,
    minimum_update_norm: float = 1e-12,
) -> list[dict[str, Any]]:
    """Measure certificate violation on the actual Cell parameter delta.

    Certificate-ranked Cell/steps with a numerically zero parameter transaction are
    retained in the raw audit table but marked ineligible for the rho distribution.
    This prevents inactive/zero updates from artificially improving p95 coverage.
    """

    if len(before) != model.cell_count:
        raise ValueError("M2-R0 weight snapshot does not match Cell count")
    if minimum_update_norm < 0:
        raise ValueError("minimum_update_norm must be non-negative")
    rows: list[dict[str, Any]] = []
    for cell_id, (cell, baseline) in enumerate(
        zip(model.cellular.cells, before, strict=True)
    ):
        rank = cell.rank
        if rank <= 0:
            continue
        baseline = baseline.to(device=cell.weight.device, dtype=cell.weight.dtype)
        delta = cell.weight.detach() - baseline
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        q = cell.certificate_basis[:rank].to(device=delta.device, dtype=delta.dtype)
        residual = delta.matmul(q.transpose(0, 1))
        violation_norm = float(torch.linalg.vector_norm(residual).item())
        eligible = delta_norm > minimum_update_norm
        violation_ratio = violation_norm / (delta_norm + 1e-12) if eligible else 0.0
        rows.append(
            {
                "arm": arm,
                "step": int(step),
                "cell_id": int(cell_id),
                "certificate_rank": int(rank),
                "update_norm": delta_norm,
                "violation_norm": violation_norm,
                "violation_ratio": violation_ratio,
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


def summarize_invariant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_rows = [row for row in rows if bool(row.get("eligible_for_ratio", True))]
    ratios = [float(row["violation_ratio"]) for row in eligible_rows]
    update_norms = [float(row["update_norm"]) for row in eligible_rows]
    ranks = [int(row["certificate_rank"]) for row in eligible_rows]
    return {
        "certificate_ranked_cell_steps": len(rows),
        "audited_cell_updates": len(eligible_rows),
        "tiny_or_zero_updates_skipped": len(rows) - len(eligible_rows),
        "certificate_rank_min": min(ranks) if ranks else 0,
        "certificate_rank_max": max(ranks) if ranks else 0,
        "violation_ratio_mean": sum(ratios) / len(ratios) if ratios else float("nan"),
        "violation_ratio_median": _percentile(ratios, 0.5),
        "violation_ratio_p95": _percentile(ratios, 0.95),
        "violation_ratio_max": max(ratios) if ratios else float("nan"),
        "update_norm_mean": sum(update_norms) / len(update_norms) if update_norms else 0.0,
    }


def _reference_pass(summary: dict[str, Any], thresholds: M2R0Thresholds) -> bool:
    return (
        int(summary["audited_cell_updates"])
        >= thresholds.minimum_audited_cell_updates_per_arm
        and float(summary["violation_ratio_max"])
        <= thresholds.reference_max_violation_ratio
        and float(summary["violation_ratio_p95"])
        <= thresholds.reference_p95_violation_ratio
    )


def _material_violation(summary: dict[str, Any], thresholds: M2R0Thresholds) -> bool:
    return float(summary["violation_ratio_max"]) > thresholds.material_violation_ratio


def classify_optimizer_invariant(
    arm_summaries: dict[str, dict[str, Any]],
    thresholds: M2R0Thresholds,
) -> str:
    required = {
        "current_adamw_grad_projection",
        "adamw_no_decay_grad_projection",
        "sgd_no_decay_grad_projection",
        "sgd_with_decay_grad_projection",
        "adamw_final_update_projection",
    }
    if set(arm_summaries) != required:
        raise ValueError("M2-R0 arm set does not match frozen protocol")

    sgd_ok = _reference_pass(arm_summaries["sgd_no_decay_grad_projection"], thresholds)
    final_projection_ok = _reference_pass(
        arm_summaries["adamw_final_update_projection"], thresholds
    )
    if not (sgd_ok and final_projection_ok):
        return "INCONCLUSIVE_REFERENCE_FAILURE"

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
        return "CURRENT_UPDATE_INVARIANT_HOLDS"
    if preconditioner_bad and decay_bad:
        return "BOTH_PRECONDITIONER_AND_WEIGHT_DECAY_BREAK_UPDATE_INVARIANT"
    if preconditioner_bad:
        return "ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT"
    if decay_bad:
        return "WEIGHT_DECAY_BREAKS_UPDATE_INVARIANT"
    return "MIXED_OR_INTERACTION_UPDATE_INVARIANT_VIOLATION"
