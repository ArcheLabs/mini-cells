"""Core Validation 009A diagnostic bridge — right-side collapse robustness.

This module is intentionally post-confirmation and non-decisional.  It explains
why the positive Core 009A factor geometry was strongly asymmetric without
modifying the frozen 009A protocol, winner lock, gates, or scientific result.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .real_representation_006_experiment import ProjectedSequence
from .real_representation_009a_experiment import (
    _covariance_bases,
    _cumulative_energy,
    _project_left,
    _project_right,
    _project_two_sided,
    _summary,
    normalize_write,
)

_EPS = 1e-12


@dataclass(frozen=True)
class BridgeSequence:
    partition: str
    token_sha256: str
    q: torch.Tensor
    z: torch.Tensor
    raw_write: torch.Tensor


@dataclass(frozen=True)
class ZTransformState:
    mean: torch.Tensor
    centered_covariance: torch.Tensor
    whitening: torch.Tensor
    mean_direction: torch.Tensor
    mean_direction_projector: torch.Tensor
    raw_second_moment: torch.Tensor


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    an = float(torch.linalg.norm(a).item())
    bn = float(torch.linalg.norm(b).item())
    if an <= _EPS or bn <= _EPS:
        return 0.0
    return abs(float(torch.dot(a, b).item()) / (an * bn))


def _eigensystem(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vals, vecs = torch.linalg.eigh(cov.to(dtype=torch.float64))
    order = torch.argsort(vals, descending=True)
    return vals[order].clamp_min(0.0), vecs[:, order].contiguous()


def _participation_rank(values: torch.Tensor) -> float:
    values = values.clamp_min(0.0)
    total = float(values.sum().item())
    denom = float(torch.square(values).sum().item())
    if total <= _EPS or denom <= _EPS:
        return 0.0
    return total * total / denom


def _dimension_at_energy(values: torch.Tensor, threshold: float) -> int:
    total = float(values.sum().item())
    if total <= _EPS:
        return 0
    csum = torch.cumsum(values, dim=0) / total
    hit = torch.nonzero(csum >= float(threshold), as_tuple=False)
    return int(hit[0].item()) + 1 if len(hit) else int(values.numel())


def _spectrum_payload(values: torch.Tensor, dims: list[int], thresholds: list[float]) -> dict[str, Any]:
    return {
        "curve": _cumulative_energy(values, dims),
        "participation_rank": _participation_rank(values),
        "dimension_at_energy": {
            str(float(t)): _dimension_at_energy(values, float(t)) for t in thresholds
        },
    }


def extract_bridge_sequences(
    sequences: list[ProjectedSequence],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> list[BridgeSequence]:
    """Extract token-level projected gradients while reproducing raw 009A writes.

    `raw_write` is accumulated in float32 exactly like Core 007/009A before
    conversion to float64.  The token q/z tensors are retained in float64 only
    for the diagnostic controls.
    """
    out: list[BridgeSequence] = []
    weight = lm_head_weight.to(device=device, dtype=torch.float32)
    u_dev = u.to(device=device, dtype=torch.float32)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        hidden = torch.stack([s.hidden for s in batch]).to(device=device, dtype=torch.float32)
        labels = torch.stack([s.labels for s in batch]).to(device=device, dtype=torch.long)
        hidden = hidden.detach().requires_grad_(True)
        logits = F.linear(hidden, weight)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
        )
        grad_h = torch.autograd.grad(loss, hidden)[0]
        projected_grad = torch.einsum("bth,hd->btd", grad_h, u_dev)
        for i, seq in enumerate(batch):
            z32 = seq.z.to(device=device, dtype=torch.float32)
            raw = torch.einsum("to,ti->oi", projected_grad[i], z32) / max(seq.tokens, 1)
            out.append(
                BridgeSequence(
                    partition=str(seq.partition),
                    token_sha256=seq.token_sha256,
                    q=projected_grad[i].detach().cpu().to(dtype=torch.float64),
                    z=seq.z.detach().cpu().to(dtype=torch.float64),
                    raw_write=normalize_write(raw),
                )
            )
    return out


def fit_z_transform_state(
    sequences: list[BridgeSequence], *, whitening_floor_fraction: float
) -> ZTransformState:
    train_z = torch.cat([s.z for s in sequences if s.partition == "train"], dim=0)
    if train_z.numel() == 0:
        raise ValueError("bridge requires training tokens")
    mean = train_z.mean(dim=0)
    centered = train_z - mean
    centered_cov = centered.T @ centered / max(int(centered.shape[0]), 1)
    vals, vecs = _eigensystem(centered_cov)
    max_eval = max(float(vals[0].item()) if len(vals) else 0.0, _EPS)
    floor = max(max_eval * float(whitening_floor_fraction), _EPS)
    whitening = vecs @ torch.diag(torch.rsqrt(vals.clamp_min(floor))) @ vecs.T
    mean_norm = float(torch.linalg.norm(mean).item())
    mean_direction = mean / mean_norm if mean_norm > _EPS else torch.zeros_like(mean)
    ident = torch.eye(train_z.shape[1], dtype=torch.float64)
    mean_projector = ident - torch.outer(mean_direction, mean_direction)
    raw_second = train_z.T @ train_z / max(int(train_z.shape[0]), 1)
    return ZTransformState(
        mean=mean,
        centered_covariance=centered_cov,
        whitening=whitening,
        mean_direction=mean_direction,
        mean_direction_projector=mean_projector,
        raw_second_moment=raw_second,
    )


def apply_z_transform(z: torch.Tensor, name: str, state: ZTransformState) -> torch.Tensor:
    if name == "raw":
        return z
    if name == "centered":
        return z - state.mean
    if name == "whitened":
        return (z - state.mean) @ state.whitening
    if name == "mean_direction_removed":
        return z @ state.mean_direction_projector
    raise ValueError(f"unknown z control: {name}")


def _condition_rows(
    sequences: list[BridgeSequence], name: str, state: ZTransformState
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in sequences:
        z = apply_z_transform(seq.z, name, state)
        if name == "raw":
            g = seq.raw_write
        else:
            write = torch.einsum("to,ti->oi", seq.q, z) / max(int(z.shape[0]), 1)
            g = normalize_write(write)
        rows.append(
            {
                "partition": seq.partition,
                "token_sha256": seq.token_sha256,
                "z": z,
                "g": g,
            }
        )
    return rows


def _token_right_covariances(
    sequences: list[BridgeSequence], name: str, state: ZTransformState
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dim = int(sequences[0].z.shape[1])
    normalized = torch.zeros(dim, dim, dtype=torch.float64)
    weighted = torch.zeros(dim, dim, dtype=torch.float64)
    count = 0
    total_weight = 0.0
    for seq in sequences:
        if seq.partition != "train":
            continue
        z = apply_z_transform(seq.z, name, state)
        qnorm2 = torch.square(seq.q).sum(dim=1)
        znorm2 = torch.square(z).sum(dim=1)
        mask = (qnorm2 > _EPS) & (znorm2 > _EPS)
        if not bool(mask.any()):
            continue
        zv = z[mask]
        q2 = qnorm2[mask]
        z2 = znorm2[mask]
        zhat = zv / torch.sqrt(z2)[:, None]
        normalized += zhat.T @ zhat
        token_weight = q2
        weighted += zv.T @ (zv * token_weight[:, None])
        count += int(mask.sum().item())
        total_weight += float((q2 * z2).sum().item())
    if count:
        normalized /= count
    if total_weight > _EPS:
        # The scalar normalization does not affect eigenvectors or cumulative energy,
        # but keeps magnitudes numerically comparable across controls.
        weighted /= total_weight
    return normalized, weighted, count


def _condition_geometry(
    rows: list[dict[str, Any]],
    sequences: list[BridgeSequence],
    name: str,
    state: ZTransformState,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    geom = protocol["write_geometry"]
    dims = [int(x) for x in geom["dimension_grid"]]
    thresholds = [float(x) for x in geom["energy_thresholds"]]
    right_grid = [int(x) for x in geom["two_sided_right_grid"]]
    left_ref = int(geom["two_sided_left_reference_dim"])
    train = [r for r in rows if r["partition"] == "train"]
    eval_rows = [r for r in rows if r["partition"] == "eval"]
    left, right, lvals, rvals = _covariance_bases(train)

    right_only: list[dict[str, Any]] = []
    for d in right_grid:
        for partition, subset in (("train", train), ("eval", eval_rows)):
            right_only.append(
                {
                    "partition": partition,
                    "dimension": d,
                    **_summary(subset, lambda g, d=d: _project_right(g, right, d)),
                }
            )

    two_sided: list[dict[str, Any]] = []
    for n in right_grid:
        for partition, subset in (("train", train), ("eval", eval_rows)):
            two_sided.append(
                {
                    "partition": partition,
                    "left_dim": left_ref,
                    "right_dim": n,
                    **_summary(
                        subset,
                        lambda g, n=n: _project_two_sided(g, left, right, left_ref, n),
                    ),
                }
            )

    left_ref_summary = []
    for partition, subset in (("train", train), ("eval", eval_rows)):
        left_ref_summary.append(
            {
                "partition": partition,
                "dimension": left_ref,
                **_summary(subset, lambda g: _project_left(g, left, left_ref)),
            }
        )

    token_norm_cov, token_weight_cov, token_count = _token_right_covariances(
        sequences, name, state
    )
    token_norm_vals, token_norm_vecs = _eigensystem(token_norm_cov)
    token_weight_vals, token_weight_vecs = _eigensystem(token_weight_cov)

    payload = {
        "condition": name,
        "sequence_left_spectrum": _spectrum_payload(lvals, dims, thresholds),
        "sequence_right_spectrum": _spectrum_payload(rvals, dims, thresholds),
        "token_normalized_right_spectrum": _spectrum_payload(token_norm_vals, dims, thresholds),
        "token_energy_weighted_right_spectrum": _spectrum_payload(token_weight_vals, dims, thresholds),
        "token_count": token_count,
        "right_only": right_only,
        "two_sided_56": two_sided,
        "left_56": left_ref_summary,
    }
    cache = {
        "left": left,
        "right": right,
        "left_values": lvals,
        "right_values": rvals,
        "token_normalized_right_pc1": token_norm_vecs[:, 0],
        "token_energy_weighted_right_pc1": token_weight_vecs[:, 0],
    }
    return payload, cache


def _top1_ablation(
    raw_rows: list[dict[str, Any]],
    raw_right: torch.Tensor,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    dims = [int(x) for x in protocol["write_geometry"]["dimension_grid"]]
    thresholds = [float(x) for x in protocol["write_geometry"]["energy_thresholds"]]
    r1 = raw_right[:, 0]
    projector = torch.eye(len(r1), dtype=torch.float64) - torch.outer(r1, r1)
    by_partition: dict[str, dict[str, list[float]]] = {
        "train": {"fro": [], "action": [], "removed_action": []},
        "eval": {"fro": [], "action": [], "removed_action": []},
    }
    residual_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        g = row["g"]
        z = row["z"]
        residual = g @ projector
        removed = g - residual
        target_action = z @ g.T
        residual_action = z @ residual.T
        removed_action = z @ removed.T
        denom_g = max(float(torch.linalg.norm(g).item()), _EPS)
        denom_action = max(float(torch.linalg.norm(target_action).item()), _EPS)
        bucket = by_partition[str(row["partition"])]
        bucket["fro"].append(float(torch.linalg.norm(residual).item()) / denom_g)
        bucket["action"].append(float(torch.linalg.norm(residual_action).item()) / denom_action)
        bucket["removed_action"].append(float(torch.linalg.norm(removed_action).item()) / denom_action)
        if float(torch.linalg.norm(residual).item()) > _EPS:
            residual_rows.append(
                {
                    "partition": row["partition"],
                    "z": z,
                    "g": normalize_write(residual),
                }
            )

    train_residual = [r for r in residual_rows if r["partition"] == "train"]
    if train_residual:
        _, _, _, residual_rvals = _covariance_bases(train_residual)
        residual_spectrum = _spectrum_payload(residual_rvals, dims, thresholds)
    else:
        residual_spectrum = {
            "curve": [{"dimension": d, "cumulative_energy": 0.0} for d in dims],
            "participation_rank": 0.0,
            "dimension_at_energy": {str(t): 0 for t in thresholds},
        }

    summaries = {}
    for partition, values in by_partition.items():
        summaries[partition] = {
            "count": len(values["fro"]),
            "median_residual_frobenius_fraction": _median(values["fro"]),
            "mean_residual_frobenius_fraction": _mean(values["fro"]),
            "median_residual_local_action_fraction": _median(values["action"]),
            "mean_residual_local_action_fraction": _mean(values["action"]),
            "median_removed_component_local_action_fraction": _median(values["removed_action"]),
            "mean_removed_component_local_action_fraction": _mean(values["removed_action"]),
        }
    return {
        "partition_summary": summaries,
        "normalized_residual_training_write_count": len(train_residual),
        "residual_right_spectrum": residual_spectrum,
    }


def _curve_energy(payload: dict[str, Any], dimension: int) -> float:
    return float(
        next(r["cumulative_energy"] for r in payload["curve"] if int(r["dimension"]) == dimension)
    )


def _find_two_sided(condition: dict[str, Any], partition: str, right_dim: int) -> dict[str, Any]:
    return next(
        r
        for r in condition["two_sided_56"]
        if r["partition"] == partition and int(r["right_dim"]) == int(right_dim)
    )


def run_bridge(
    sequences: list[BridgeSequence], protocol: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("no bridge sequences")
    controls = ["raw", "centered", "whitened", "mean_direction_removed"]
    state = fit_z_transform_state(
        sequences,
        whitening_floor_fraction=float(
            protocol["controls"]["whitening_eigenvalue_floor_fraction_of_max"]
        ),
    )

    conditions: dict[str, Any] = {}
    caches: dict[str, dict[str, torch.Tensor]] = {}
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    for name in controls:
        rows = _condition_rows(sequences, name, state)
        result, cache = _condition_geometry(rows, sequences, name, state, protocol)
        rows_by_condition[name] = rows
        conditions[name] = result
        caches[name] = cache

    raw_right_pc1 = caches["raw"]["right"][:, 0]
    raw_second_vals, raw_second_vecs = _eigensystem(state.raw_second_moment)
    centered_vals, centered_vecs = _eigensystem(state.centered_covariance)
    alignment = {
        "raw_sequence_right_pc1_vs_train_token_mean_direction": _abs_cos(
            raw_right_pc1, state.mean_direction
        ),
        "raw_sequence_right_pc1_vs_raw_token_normalized_right_pc1": _abs_cos(
            raw_right_pc1, caches["raw"]["token_normalized_right_pc1"]
        ),
        "raw_sequence_right_pc1_vs_raw_token_energy_weighted_right_pc1": _abs_cos(
            raw_right_pc1, caches["raw"]["token_energy_weighted_right_pc1"]
        ),
        "raw_sequence_right_pc1_vs_raw_z_second_moment_pc1": _abs_cos(
            raw_right_pc1, raw_second_vecs[:, 0]
        ),
        "raw_sequence_right_pc1_vs_centered_z_covariance_pc1": _abs_cos(
            raw_right_pc1, centered_vecs[:, 0]
        ),
        "raw_z_second_moment_top1_energy": float(raw_second_vals[0].item())
        / max(float(raw_second_vals.sum().item()), _EPS),
        "centered_z_covariance_top1_energy": float(centered_vals[0].item())
        / max(float(centered_vals.sum().item()), _EPS),
    }

    ablation = _top1_ablation(rows_by_condition["raw"], caches["raw"]["right"], protocol)
    raw_56x8 = _find_two_sided(conditions["raw"], "eval", 8)

    return {
        "format": "minicells.core-validation.009a-right-collapse-bridge-seed.v1",
        "experiment_id": protocol["experiment_id"],
        "seed": int(seed),
        "scientific_decision": False,
        "source_009a_status_changed": False,
        "conditions": conditions,
        "alignment": alignment,
        "top1_ablation": ablation,
        "raw_source_reference": {
            "left_dim": 56,
            "right_dim": 8,
            "eval_median_local_action_residual": float(raw_56x8["median_local_action_residual"]),
            "eval_median_frobenius_residual": float(raw_56x8["median_frobenius_residual"]),
        },
    }


def spectrum_energy(condition: dict[str, Any], key: str, dimension: int) -> float:
    """Public helper used by reporter/tests."""
    return _curve_energy(condition[key], dimension)
