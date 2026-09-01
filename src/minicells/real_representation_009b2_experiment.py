"""Core Validation 009B-2 — Persistent Effect Geometry.

The scientific object is the 64-dimensional carrier effect

    a_i = Ghat_i r

where Ghat is the normalized full write from Core 009B-1 and r is the
train-fitted common carrier. This module deliberately does not add routing,
sparsity, certificates or model mutation.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch

from .real_representation_009b1_experiment import CausalSequence, fit_train_carrier

_EPS = 1e-12


@dataclass(frozen=True)
class EffectSequence:
    partition: str
    source: str
    token_sha256: str
    effect: torch.Tensor
    effect_norm: float


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _p90(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    idx = max(0, min(len(ys) - 1, math.ceil(0.90 * len(ys)) - 1))
    return float(ys[idx])


def build_effect_sequences(sequences: list[CausalSequence]) -> tuple[list[EffectSequence], torch.Tensor]:
    carrier = fit_train_carrier(sequences).to(dtype=torch.float64)
    out: list[EffectSequence] = []
    for seq in sequences:
        if seq.partition not in {"train", "eval"}:
            continue
        effect = seq.ghat.to(dtype=torch.float64) @ carrier
        norm = float(torch.linalg.norm(effect).item())
        out.append(EffectSequence(seq.partition, seq.source, seq.token_sha256, effect, norm))
    if not out:
        raise ValueError("009B-2 requires train/eval effect sequences")
    return out, carrier


def _matrix(rows: list[EffectSequence]) -> torch.Tensor:
    if not rows:
        raise ValueError("empty effect rows")
    return torch.stack([r.effect for r in rows]).to(dtype=torch.float64)


def fit_uncentered_basis(rows: list[EffectSequence]) -> tuple[torch.Tensor, torch.Tensor]:
    a = _matrix(rows)
    cov = a.T @ a / max(int(a.shape[0]), 1)
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, descending=True)
    return vecs[:, order].contiguous(), vals[order].clamp_min(0.0)


def normalized_residual(effect: torch.Tensor, basis: torch.Tensor, dim: int | None = None) -> float:
    use = basis if dim is None else basis[:, : max(0, min(int(dim), int(basis.shape[1])))]
    denom = max(float(torch.linalg.norm(effect).item()), _EPS)
    if use.numel() == 0 or use.shape[1] == 0:
        return 1.0
    projected = use @ (use.T @ effect)
    return float(torch.linalg.norm(effect - projected).item()) / denom


def projected_cosine(effect: torch.Tensor, basis: torch.Tensor, dim: int) -> float:
    use = basis[:, : max(0, min(int(dim), int(basis.shape[1])))]
    denom = max(float(torch.linalg.norm(effect).item()), _EPS)
    if use.numel() == 0 or use.shape[1] == 0:
        return 0.0
    projected = use @ (use.T @ effect)
    pnorm = float(torch.linalg.norm(projected).item())
    if pnorm <= _EPS:
        return 0.0
    return float(torch.dot(effect, projected).item()) / (denom * pnorm)


def summarize_basis(rows: list[EffectSequence], basis: torch.Tensor, dim: int) -> dict[str, Any]:
    residuals = [normalized_residual(r.effect, basis, dim) for r in rows]
    cosines = [projected_cosine(r.effect, basis, dim) for r in rows]
    return {
        "count": len(rows),
        "dimension": int(dim),
        "median_normalized_residual": _median(residuals),
        "mean_normalized_residual": _mean(residuals),
        "p90_normalized_residual": _p90(residuals),
        "median_projected_cosine": _median(cosines),
    }


def spectrum_payload(values: torch.Tensor, dims: list[int]) -> list[dict[str, float | int]]:
    total = max(float(values.sum().item()), _EPS)
    csum = torch.cumsum(values, dim=0) / total
    out = []
    for d in dims:
        idx = max(0, min(int(d), int(values.numel()))) - 1
        out.append({"dimension": int(d), "cumulative_energy": 0.0 if idx < 0 else float(csum[idx].item())})
    return out


def run_discovery(sequences: list[CausalSequence], protocol: dict[str, Any], *, seed: int) -> dict[str, Any]:
    effects, carrier = build_effect_sequences(sequences)
    train = [r for r in effects if r.partition == "train"]
    eval_rows = [r for r in effects if r.partition == "eval"]
    basis, values = fit_uncentered_basis(train)
    dims = [int(x) for x in protocol["offline_geometry"]["dimension_grid"]]
    dimension_rows: list[dict[str, Any]] = []
    for d in dims:
        tr = summarize_basis(train, basis, d)
        ev = summarize_basis(eval_rows, basis, d)
        dimension_rows.append({
            "seed": int(seed), "dimension": d, "train": tr, "eval": ev,
            "train_to_eval_median_residual_gap": float(ev["median_normalized_residual"]) - float(tr["median_normalized_residual"]),
        })
    norms = [r.effect_norm for r in effects]
    return {
        "format": "minicells.core-validation.persistent-effect-geometry-discovery-seed.v1",
        "experiment_id": protocol["experiment_id"], "seed": int(seed), "scientific_decision": False,
        "carrier_norm": float(torch.linalg.norm(carrier).item()), "train_count": len(train), "eval_count": len(eval_rows),
        "effect_norm": {"median": _median(norms), "mean": _mean(norms), "minimum": min(norms), "maximum": max(norms)},
        "spectrum": spectrum_payload(values, dims), "dimension_rows": dimension_rows,
    }


def discovery_candidate_row(seed_payload: dict[str, Any], dimension: int, protocol: dict[str, Any]) -> dict[str, Any]:
    row = next(r for r in seed_payload["dimension_rows"] if int(r["dimension"]) == int(dimension))
    gates = protocol["discovery"]["gates"]
    ev = row["eval"]
    viable = (
        float(ev["median_normalized_residual"]) <= float(gates["maximum_eval_median_residual"])
        and float(ev["p90_normalized_residual"]) <= float(gates["maximum_eval_p90_residual"])
        and float(row["train_to_eval_median_residual_gap"]) <= float(gates["maximum_train_to_eval_median_residual_gap"])
        and int(dimension) <= int(gates["maximum_locked_dimension"])
    )
    return {
        "seed": int(seed_payload["seed"]), "dimension": int(dimension),
        "eval_median_normalized_residual": float(ev["median_normalized_residual"]),
        "eval_p90_normalized_residual": float(ev["p90_normalized_residual"]),
        "train_to_eval_median_residual_gap": float(row["train_to_eval_median_residual_gap"]), "viable": bool(viable),
    }


def select_discovery_dimension(seed_payloads: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[int | None, list[dict[str, Any]]]:
    candidates = [int(x) for x in protocol["offline_geometry"]["compact_candidate_dimensions"]]
    summary = []
    for d in candidates:
        per_seed = [discovery_candidate_row(p, d, protocol) for p in seed_payloads]
        summary.append({"dimension": d, "all_completed_seed_rows_viable": bool(per_seed) and all(x["viable"] for x in per_seed), "per_seed": per_seed})
    if len(seed_payloads) != len(protocol["discovery"]["seeds"]):
        return None, summary
    for row in summary:
        if row["all_completed_seed_rows_viable"]:
            return int(row["dimension"]), summary
    return None, summary


def _ordering_key(seed: int, ordering: str, token_sha256: str) -> str:
    return hashlib.sha256(f"{seed}:{ordering}:{token_sha256}".encode()).hexdigest()


def ordered_train_rows(train: list[EffectSequence], *, seed: int, ordering: str) -> list[EffectSequence]:
    if ordering == "canonical":
        return list(train)
    if not ordering.startswith("sha-"):
        raise ValueError(f"unknown ordering: {ordering}")
    return sorted(train, key=lambda r: _ordering_key(seed, ordering, r.token_sha256))


def _append_orthonormal(basis: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    v = vector.to(dtype=torch.float64)
    if basis.numel() and basis.shape[1] > 0:
        for _ in range(2):
            v = v - basis @ (basis.T @ v)
    norm = float(torch.linalg.norm(v).item())
    if norm <= _EPS:
        return basis
    v = v / norm
    return v[:, None] if basis.numel() == 0 or basis.shape[1] == 0 else torch.cat([basis, v[:, None]], dim=1)


def online_incremental_basis(train: list[EffectSequence], eval_rows: list[EffectSequence], *, seed: int, ordering: str, threshold: float, checkpoints: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)) -> dict[str, Any]:
    stream = ordered_train_rows(train, seed=seed, ordering=ordering)
    ambient = int(stream[0].effect.numel()) if stream else 64
    basis = torch.empty(ambient, 0, dtype=torch.float64)
    growth_rows = []
    growth_events = late_growth_events = 0
    late_start = len(stream) // 2
    checkpoint_positions = {max(1, min(len(stream), int(math.ceil(frac * len(stream))))): frac for frac in checkpoints}
    curve = []
    for idx, row in enumerate(stream, start=1):
        before = normalized_residual(row.effect, basis)
        grew = False
        if before > float(threshold):
            residual = row.effect if basis.shape[1] == 0 else row.effect - basis @ (basis.T @ row.effect)
            old_k = int(basis.shape[1])
            basis = _append_orthonormal(basis, residual)
            grew = int(basis.shape[1]) > old_k
            if grew:
                growth_events += 1
                if idx > late_start:
                    late_growth_events += 1
        growth_rows.append({"position": idx, "token_sha256": row.token_sha256, "residual_before": before, "grew": grew, "dimension_after": int(basis.shape[1])})
        if idx in checkpoint_positions:
            curve.append({"fraction": float(checkpoint_positions[idx]), "writes_seen": idx, "dimension": int(basis.shape[1])})
    eval_residuals = [normalized_residual(r.effect, basis) for r in eval_rows]
    n = len(stream); final_k = int(basis.shape[1]); late_n = max(n - late_start, 1)
    return {
        "ordering": ordering, "train_writes": n, "final_dimension": final_k, "growth_events": growth_events,
        "new_coordinates_per_100_writes": 100.0 * growth_events / max(n, 1),
        "late_growth_events": late_growth_events, "late_growth_per_100_writes": 100.0 * late_growth_events / late_n,
        "independent_memory_units": n, "independent_memory_compression_ratio": float(n / max(final_k, 1)),
        "eval_median_normalized_residual": _median(eval_residuals), "eval_p90_normalized_residual": _p90(eval_residuals),
        "eval_mean_normalized_residual": _mean(eval_residuals), "growth_curve": curve, "growth_rows": growth_rows,
    }


def run_confirmation(sequences: list[CausalSequence], protocol: dict[str, Any], *, seed: int, locked_dimension: int) -> dict[str, Any]:
    effects, carrier = build_effect_sequences(sequences)
    train = [r for r in effects if r.partition == "train"]
    eval_rows = [r for r in effects if r.partition == "eval"]
    basis, values = fit_uncentered_basis(train)
    tr = summarize_basis(train, basis, locked_dimension); ev = summarize_basis(eval_rows, basis, locked_dimension)
    threshold = float(protocol["online_growth"]["residual_threshold_tau"])
    online = [online_incremental_basis(train, eval_rows, seed=seed, ordering=name, threshold=threshold) for name in protocol["online_growth"]["orderings"]]
    return {
        "format": "minicells.core-validation.persistent-effect-geometry-confirmation-seed.v1",
        "experiment_id": protocol["experiment_id"], "seed": int(seed), "scientific_decision": False,
        "locked_dimension": int(locked_dimension), "carrier_norm": float(torch.linalg.norm(carrier).item()),
        "train_count": len(train), "eval_count": len(eval_rows),
        "offline": {"train": tr, "eval": ev, "train_to_eval_median_residual_gap": float(ev["median_normalized_residual"]) - float(tr["median_normalized_residual"]), "spectrum": spectrum_payload(values, [int(x) for x in protocol["offline_geometry"]["dimension_grid"]])},
        "online": online,
    }


def confirmation_gate_row(payload: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    gates = protocol["confirmation"]["gates"]; offline = payload["offline"]; online = payload["online"]; locked = int(payload["locked_dimension"])
    max_final = max(int(r["final_dimension"]) for r in online)
    max_over_locked = max(float(r["final_dimension"]) / max(locked, 1) for r in online)
    max_eval_median = max(float(r["eval_median_normalized_residual"]) for r in online)
    max_eval_p90 = max(float(r["eval_p90_normalized_residual"]) for r in online)
    max_late = max(float(r["late_growth_per_100_writes"]) for r in online)
    min_compression = min(float(r["independent_memory_compression_ratio"]) for r in online)
    checks = {
        "offline_eval_median": float(offline["eval"]["median_normalized_residual"]) <= float(gates["maximum_offline_eval_median_residual"]),
        "offline_eval_p90": float(offline["eval"]["p90_normalized_residual"]) <= float(gates["maximum_offline_eval_p90_residual"]),
        "offline_generalization": float(offline["train_to_eval_median_residual_gap"]) <= float(gates["maximum_offline_train_to_eval_median_residual_gap"]),
        "online_final_dimension": max_final <= int(gates["maximum_online_final_dimension"]),
        "online_over_locked_dimension": max_over_locked <= float(gates["maximum_online_over_locked_dimension_ratio"]),
        "online_eval_median": max_eval_median <= float(gates["maximum_online_eval_median_residual"]),
        "online_eval_p90": max_eval_p90 <= float(gates["maximum_online_eval_p90_residual"]),
        "late_growth": max_late <= float(gates["maximum_late_growth_per_100_writes"]),
        "memory_compression": min_compression >= float(gates["minimum_independent_memory_compression_ratio"]),
    }
    return {
        "seed": int(payload["seed"]), "locked_dimension": locked,
        "offline_eval_median_residual": float(offline["eval"]["median_normalized_residual"]),
        "offline_eval_p90_residual": float(offline["eval"]["p90_normalized_residual"]),
        "offline_train_to_eval_median_residual_gap": float(offline["train_to_eval_median_residual_gap"]),
        "maximum_online_final_dimension": max_final, "maximum_online_over_locked_dimension_ratio": max_over_locked,
        "maximum_online_eval_median_residual": max_eval_median, "maximum_online_eval_p90_residual": max_eval_p90,
        "maximum_late_growth_per_100_writes": max_late, "minimum_independent_memory_compression_ratio": min_compression,
        **checks, "pass": all(checks.values()),
    }
