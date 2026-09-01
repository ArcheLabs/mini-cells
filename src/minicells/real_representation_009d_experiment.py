"""Core Validation 009D — Compositional Operator Geometry.

Representation-only continuation after Core 009C.  The carrier-compressed effect
a_i = Ghat_i r is deliberately not used as the representation target.  Instead
this module keeps the full normalized 64x64 write operator and asks whether the
joint left/right structure contains reusable linear/separable organization.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch

from .real_representation_009a_experiment import _covariance_bases
from .real_representation_009b1_experiment import CausalSequence, fit_train_carrier

_EPS = 1e-12


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _p90(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    return float(ys[max(0, min(len(ys) - 1, math.ceil(0.90 * len(ys)) - 1))])


def _rel_improvement(baseline: float, value: float) -> float:
    return (float(baseline) - float(value)) / max(float(baseline), _EPS)


@dataclass(frozen=True)
class OperatorRow:
    partition: str
    source: str
    token_sha256: str
    z: torch.Tensor
    g: torch.Tensor


@dataclass(frozen=True)
class FactorRow:
    partition: str
    source: str
    token_sha256: str
    sigma: float
    left: torch.Tensor
    right: torch.Tensor


def operator_rows(sequences: list[CausalSequence]) -> list[OperatorRow]:
    rows: list[OperatorRow] = []
    for seq in sequences:
        if seq.partition not in {"train", "eval"}:
            continue
        rows.append(OperatorRow(str(seq.partition), str(seq.source), seq.token_sha256, seq.z.detach().cpu().to(dtype=torch.float64), seq.ghat.detach().cpu().to(dtype=torch.float64)))
    if not rows:
        raise ValueError("009D requires train/eval operator rows")
    return rows


def _as_009a_rows(rows: list[OperatorRow]) -> list[dict[str, Any]]:
    return [{"partition": r.partition, "token_sha256": r.token_sha256, "z": r.z, "g": r.g} for r in rows]


def _fro_residual(g: torch.Tensor, approx: torch.Tensor) -> float:
    return float(torch.linalg.norm(g - approx).item()) / max(float(torch.linalg.norm(g).item()), _EPS)


def _action_residual(row: OperatorRow, approx: torch.Tensor) -> float:
    target = row.z @ row.g.T
    pred = row.z @ approx.T
    return float(torch.linalg.norm(target - pred).item()) / max(float(torch.linalg.norm(target).item()), _EPS)


def summarize_approx(rows: list[OperatorRow], approx_fn: Any) -> dict[str, Any]:
    fro, action = [], []
    for row in rows:
        approx = approx_fn(row)
        fro.append(_fro_residual(row.g, approx))
        action.append(_action_residual(row, approx))
    return {"count": len(rows), "median_frobenius_residual": _median(fro), "mean_frobenius_residual": _mean(fro), "p90_frobenius_residual": _p90(fro), "median_local_action_residual": _median(action), "mean_local_action_residual": _mean(action), "p90_local_action_residual": _p90(action)}


def _best_rank1(g: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(g.to(dtype=torch.float64), full_matrices=False)
    left, right, sigma = u[:, 0].contiguous(), vh[0].contiguous(), float(s[0].item())
    return sigma * torch.outer(left, right), sigma, left, right


def factor_rows(rows: list[OperatorRow], carrier: torch.Tensor) -> list[FactorRow]:
    carrier = carrier.to(dtype=torch.float64)
    out = []
    for row in rows:
        _, sigma, left, right = _best_rank1(row.g)
        if float(torch.dot(right, carrier).item()) < 0.0:
            left, right = -left, -right
        out.append(FactorRow(row.partition, row.source, row.token_sha256, sigma, left, right))
    return out


def _project_dense(row: OperatorRow, left: torch.Tensor, right: torch.Tensor, m: int, n: int) -> torch.Tensor:
    l, r = left[:, :m], right[:, :n]
    return l @ (l.T @ row.g @ r) @ r.T


def _rank1_core(row: OperatorRow, left: torch.Tensor, right: torch.Tensor, m: int, n: int) -> torch.Tensor:
    l, r = left[:, :m], right[:, :n]
    rank1, _, _, _ = _best_rank1(l.T @ row.g @ r)
    return l @ rank1 @ r.T


def _sparse_core(row: OperatorRow, left: torch.Tensor, right: torch.Tensor, m: int, n: int, active: int) -> torch.Tensor:
    l, r = left[:, :m], right[:, :n]
    core = l.T @ row.g @ r
    flat = core.flatten()
    s = min(int(active), int(flat.numel()))
    sparse_flat = torch.zeros_like(flat)
    if s > 0:
        idx = torch.topk(torch.abs(flat), k=s, largest=True, sorted=False).indices
        sparse_flat[idx] = flat[idx]
    return l @ sparse_flat.reshape_as(core) @ r.T


def _generator(seed: int, tag: str) -> torch.Generator:
    h = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(h[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF)


def _orthogonal(dim: int, *, seed: int, tag: str) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=_generator(seed, tag), dtype=torch.float64))
    signs = torch.sign(torch.diag(r)); signs[signs == 0] = 1.0
    return q * signs[None, :]


def rotated_factor_bases(left: torch.Tensor, right: torch.Tensor, m: int, n: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    return left[:, :m] @ _orthogonal(m, seed=seed, tag=f"left-rotation-{m}-{n}"), right[:, :n] @ _orthogonal(n, seed=seed, tag=f"right-rotation-{m}-{n}")


def _vector_pca_basis(train: list[OperatorRow], maximum_dimension: int) -> torch.Tensor:
    a = torch.stack([r.g.flatten() for r in train]).to(dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(a @ a.T)
    order = torch.argsort(vals, descending=True); vals, vecs = vals[order], vecs[:, order]
    cols = []
    for j in range(min(int(maximum_dimension), int(vals.numel()))):
        lam = float(vals[j].item())
        if lam <= _EPS: break
        cols.append((a.T @ vecs[:, j]) / math.sqrt(lam))
    return torch.stack(cols, dim=1).contiguous() if cols else torch.empty(a.shape[1], 0, dtype=torch.float64)


def _vector_pca_approx(row: OperatorRow, basis: torch.Tensor, dimension: int) -> torch.Tensor:
    d = min(int(dimension), int(basis.shape[1]))
    if d <= 0: return torch.zeros_like(row.g)
    b, v = basis[:, :d], row.g.flatten()
    return (b @ (b.T @ v)).reshape_as(row.g)


def _spectrum(vectors: list[torch.Tensor], dims: list[int], thresholds: list[float]) -> dict[str, Any]:
    x = torch.stack(vectors).to(dtype=torch.float64)
    vals = torch.linalg.eigvalsh(x.T @ x / max(len(vectors), 1)).flip(0).clamp_min(0.0)
    total = max(float(vals.sum().item()), _EPS); csum = torch.cumsum(vals, dim=0) / total
    curve = [{"dimension": int(d), "cumulative_energy": float(csum[min(int(d), len(csum)) - 1].item())} for d in dims]
    participation = total * total / max(float(torch.square(vals).sum().item()), _EPS)
    at = {}
    for threshold in thresholds:
        hit = torch.nonzero(csum >= float(threshold), as_tuple=False); at[str(float(threshold))] = int(hit[0].item()) + 1 if len(hit) else int(vals.numel())
    return {"curve": curve, "participation_rank": participation, "dimension_at_energy": at}


def factor_spectra(factors: list[FactorRow], protocol: dict[str, Any]) -> dict[str, Any]:
    train = [r for r in factors if r.partition == "train"]
    dims = [int(x) for x in protocol["diagnostics"]["factor_spectra"]["dimensions"]]; thresholds = [float(x) for x in protocol["diagnostics"]["factor_spectra"]["energy_thresholds"]]
    return {"left": _spectrum([r.left for r in train], dims, thresholds), "right": _spectrum([r.right for r in train], dims, thresholds)}


def _description_bits(active: int, dictionary_size: int) -> dict[str, int]:
    support = int(active) * int(math.ceil(math.log2(max(int(dictionary_size), 2))))
    return {"support_bits": support, "with_fp16_coefficients": support + 16 * int(active), "with_fp32_coefficients": support + 32 * int(active)}


def _ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = x.to(dtype=torch.float64), y.to(dtype=torch.float64)
    xm, xs, ym = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6), y.mean(0)
    a, b = (x - xm) / xs, y - ym
    w = torch.linalg.solve(a.T @ a + float(lam) * torch.eye(a.shape[1], dtype=torch.float64), a.T @ b)
    return xm, xs, ym, w


def _ridge_predict(x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    xm, xs, ym, w = state
    return ((x.to(dtype=torch.float64) - xm) / xs) @ w + ym


def _fold(token_sha256: str, seed: int, folds: int = 4) -> int:
    h = hashlib.sha256(f"{seed}:ridge-fold:{token_sha256}".encode()).digest(); return int.from_bytes(h[:8], "little") % int(folds)


def _factor_xy(factors: list[FactorRow], left: torch.Tensor, right: torch.Tensor, m: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    l, r = left[:, :m], right[:, :n]
    return torch.stack([r.T @ row.right for row in factors]).to(dtype=torch.float64), torch.stack([l.T @ row.left for row in factors]).to(dtype=torch.float64)


def _select_lambda(train: list[FactorRow], x: torch.Tensor, y: torch.Tensor, lambdas: list[float], seed: int) -> tuple[float, torch.Tensor]:
    folds = torch.tensor([_fold(r.token_sha256, seed) for r in train], dtype=torch.long); scores = []
    for lam in lambdas:
        residuals = []
        for f in range(4):
            tr, va = folds != f, folds == f
            if not bool(tr.any()) or not bool(va.any()): continue
            pred = _ridge_predict(x[va], _ridge_fit(x[tr], y[tr], lam)); denom = torch.linalg.norm(y[va], dim=1).clamp_min(_EPS)
            residuals.extend((torch.linalg.norm(pred - y[va], dim=1) / denom).tolist())
        scores.append((_median(residuals), float(lam)))
    if not scores: raise RuntimeError("right-conditioned ridge CV produced no folds")
    best = min(scores, key=lambda z: (z[0], z[1]))[1]; oof = torch.zeros_like(y)
    for f in range(4):
        tr, va = folds != f, folds == f
        if bool(tr.any()) and bool(va.any()): oof[va] = _ridge_predict(x[va], _ridge_fit(x[tr], y[tr], best))
    return best, oof


def _permute_targets(y: torch.Tensor, train: list[FactorRow], seed: int) -> torch.Tensor:
    order = sorted(range(len(train)), key=lambda i: hashlib.sha256(f"{seed}:permute-left:{train[i].token_sha256}".encode()).hexdigest())
    if len(order) <= 1: return y.clone()
    shifted = order[1:] + order[:1]; out = torch.empty_like(y)
    for dst, src in zip(order, shifted): out[dst] = y[src]
    return out


def _left_cosines(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    denom = (torch.linalg.norm(pred, dim=1) * torch.linalg.norm(target, dim=1)).clamp_min(_EPS)
    return [float(x) for x in (torch.sum(pred * target, dim=1) / denom).tolist()]


def _reconstruct_from_factor_codes(factors: list[FactorRow], predicted_left: torch.Tensor, x_right: torch.Tensor, left: torch.Tensor, right: torch.Tensor, m: int, n: int) -> dict[str, torch.Tensor]:
    l, r = left[:, :m], right[:, :n]
    return {row.token_sha256: float(row.sigma) * torch.outer(l @ predicted_left[i], r @ x_right[i]) for i, row in enumerate(factors)}


def _summary_from_map(rows: list[OperatorRow], approx: dict[str, torch.Tensor]) -> dict[str, Any]:
    return summarize_approx(rows, lambda row: approx[row.token_sha256])


def _nearest_right_prediction(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor) -> torch.Tensor:
    tx = train_x / torch.linalg.norm(train_x, dim=1, keepdim=True).clamp_min(_EPS); ex = eval_x / torch.linalg.norm(eval_x, dim=1, keepdim=True).clamp_min(_EPS)
    return train_y[torch.argmax(ex @ tx.T, dim=1)]


def right_conditioned_payload(rows: list[OperatorRow], factors: list[FactorRow], left: torch.Tensor, right: torch.Tensor, protocol: dict[str, Any], *, seed: int, phase: str) -> dict[str, Any]:
    m = int(protocol["right_conditioned_operator"]["left_effect_dim"]); n = int(protocol["right_conditioned_operator"]["right_address_dim"]); lambdas = [float(x) for x in protocol["right_conditioned_operator"]["ridge_lambda_grid"]]
    train_f = [r for r in factors if r.partition == "train"]; eval_f = [r for r in factors if r.partition == "eval"]; train_rows = [r for r in rows if r.partition == "train"]; eval_rows = [r for r in rows if r.partition == "eval"]
    tx, ty = _factor_xy(train_f, left, right, m, n); ex, ey = _factor_xy(eval_f, left, right, m, n)
    selected_lambda, oof = _select_lambda(train_f, tx, ty, lambdas, seed); pred = _ridge_predict(ex, _ridge_fit(tx, ty, selected_lambda))
    mean_pred = ty.mean(0, keepdim=True).repeat(len(eval_f), 1); nn_pred = _nearest_right_prediction(tx, ty, ex); perm_y = _permute_targets(ty, train_f, seed)
    perm_lambda, _ = _select_lambda(train_f, tx, perm_y, lambdas, seed); perm_pred = _ridge_predict(ex, _ridge_fit(tx, perm_y, perm_lambda))
    train_summary = _summary_from_map(train_rows, _reconstruct_from_factor_codes(train_f, oof, tx, left, right, m, n)); eval_summary = _summary_from_map(eval_rows, _reconstruct_from_factor_codes(eval_f, pred, ex, left, right, m, n))
    mean_summary = _summary_from_map(eval_rows, _reconstruct_from_factor_codes(eval_f, mean_pred, ex, left, right, m, n)); nn_summary = _summary_from_map(eval_rows, _reconstruct_from_factor_codes(eval_f, nn_pred, ex, left, right, m, n)); perm_summary = _summary_from_map(eval_rows, _reconstruct_from_factor_codes(eval_f, perm_pred, ex, left, right, m, n))
    gap = max(0.0, float(eval_summary["median_local_action_residual"]) - float(train_summary["median_local_action_residual"]))
    out = {"family": "right_conditioned", "selected_ridge_lambda": selected_lambda, "permuted_selected_ridge_lambda": perm_lambda, "train_oof": train_summary, "eval": eval_summary, "mean_left_eval": mean_summary, "nearest_right_eval": nn_summary, "permuted_ridge_eval": perm_summary, "eval_median_left_factor_cosine": _median(_left_cosines(pred, ey)), "train_eval_median_action_gap": gap, "relative_median_action_improvement_over_mean_left": _rel_improvement(float(mean_summary["median_local_action_residual"]), float(eval_summary["median_local_action_residual"])), "relative_median_action_improvement_over_permuted_ridge": _rel_improvement(float(perm_summary["median_local_action_residual"]), float(eval_summary["median_local_action_residual"]))}
    gates = protocol[phase]["right_conditioned_gates"]
    checks = {"eval_median_action": float(eval_summary["median_local_action_residual"]) <= float(gates["maximum_eval_median_local_action_residual"]), "eval_p90_action": float(eval_summary["p90_local_action_residual"]) <= float(gates["maximum_eval_p90_local_action_residual"]), "eval_median_frobenius": float(eval_summary["median_frobenius_residual"]) <= float(gates["maximum_eval_median_frobenius_residual"]), "generalization_gap": gap <= float(gates["maximum_train_eval_median_action_gap"]), "left_factor_cosine": out["eval_median_left_factor_cosine"] >= float(gates["minimum_eval_median_left_factor_cosine"]), "beats_mean_left": out["relative_median_action_improvement_over_mean_left"] >= float(gates["minimum_relative_median_action_improvement_over_mean_left"]), "beats_permuted_ridge": out["relative_median_action_improvement_over_permuted_ridge"] >= float(gates["minimum_relative_median_action_improvement_over_permuted_ridge"])}
    out["checks"] = checks; out["viable"] = all(checks.values()); return out


def sparse_tensor_payloads(train: list[OperatorRow], eval_rows: list[OperatorRow], left: torch.Tensor, right: torch.Tensor, vector_basis: torch.Tensor, protocol: dict[str, Any], *, seed: int, phase: str, locked_active: int | None) -> list[dict[str, Any]]:
    m = int(protocol["validated_factor_subspace"]["left_dim"]); n = int(protocol["validated_factor_subspace"]["right_dim"]); grid = [int(locked_active)] if locked_active is not None else [int(x) for x in protocol["sparse_tensor_core"]["active_coordinate_grid"]]
    rot_left, rot_right = rotated_factor_bases(left, right, m, n, seed=seed); storage_dim = int(protocol["global_baselines"]["storage_matched_dimension"]); storage_summary = summarize_approx(eval_rows, lambda row: _vector_pca_approx(row, vector_basis, storage_dim)); rows = []
    for active in grid:
        train_summary = summarize_approx(train, lambda row, s=active: _sparse_core(row, left, right, m, n, s)); eval_summary = summarize_approx(eval_rows, lambda row, s=active: _sparse_core(row, left, right, m, n, s)); null_summary = summarize_approx(eval_rows, lambda row, s=active: _sparse_core(row, rot_left, rot_right, m, n, s)); active_pca = summarize_approx(eval_rows, lambda row, d=active: _vector_pca_approx(row, vector_basis, d))
        gap = max(0.0, float(eval_summary["median_local_action_residual"]) - float(train_summary["median_local_action_residual"]))
        payload = {"family": "sparse_tensor", "active_coordinates": active, "dictionary_size": m * n, "description_bits": _description_bits(active, m * n), "shared_parameter_count": int(protocol["sparse_tensor_core"]["shared_parameter_count"]), "train": train_summary, "eval": eval_summary, "rotated_null_eval": null_summary, "storage_matched_vector_pca_eval": storage_summary, "active_matched_vector_pca_eval": active_pca, "train_eval_median_action_gap": gap, "relative_median_action_improvement_over_rotated_null": _rel_improvement(float(null_summary["median_local_action_residual"]), float(eval_summary["median_local_action_residual"])), "relative_median_action_improvement_over_storage_matched_vector_pca": _rel_improvement(float(storage_summary["median_local_action_residual"]), float(eval_summary["median_local_action_residual"])), "relative_median_action_improvement_over_active_matched_vector_pca": _rel_improvement(float(active_pca["median_local_action_residual"]), float(eval_summary["median_local_action_residual"]))}
        gates = protocol[phase]["sparse_tensor_gates"]
        checks = {"complexity": active <= int(protocol["sparse_tensor_core"]["compact_active_limit"]), "eval_median_action": float(eval_summary["median_local_action_residual"]) <= float(gates["maximum_eval_median_local_action_residual"]), "eval_p90_action": float(eval_summary["p90_local_action_residual"]) <= float(gates["maximum_eval_p90_local_action_residual"]), "eval_median_frobenius": float(eval_summary["median_frobenius_residual"]) <= float(gates["maximum_eval_median_frobenius_residual"]), "generalization_gap": gap <= float(gates["maximum_train_eval_median_action_gap"]), "beats_rotated_null": payload["relative_median_action_improvement_over_rotated_null"] >= float(gates["minimum_relative_median_action_improvement_over_rotated_null"]), "beats_storage_matched_pca": payload["relative_median_action_improvement_over_storage_matched_vector_pca"] >= float(gates["minimum_relative_median_action_improvement_over_storage_matched_vector_pca"])}
        payload["checks"] = checks; payload["viable"] = all(checks.values()); rows.append(payload)
    return rows


def _rank1_guard(dense: dict[str, Any], compressed: dict[str, Any], protocol: dict[str, Any], *, phase: str) -> dict[str, Any]:
    action_excess = max(0.0, float(compressed["median_local_action_residual"]) - float(dense["median_local_action_residual"])); fro_excess = max(0.0, float(compressed["median_frobenius_residual"]) - float(dense["median_frobenius_residual"])); gates = protocol[phase]["rank1_core_guard"]
    checks = {"action_excess": action_excess <= float(gates["maximum_eval_median_local_action_excess_over_dense_56x8"]), "frobenius_excess": fro_excess <= float(gates["maximum_eval_median_frobenius_excess_over_dense_56x8"]), "absolute_action": float(compressed["median_local_action_residual"]) <= float(gates["maximum_eval_median_local_action_residual"])}
    return {"eval_median_local_action_excess_over_dense_56x8": action_excess, "eval_median_frobenius_excess_over_dense_56x8": fro_excess, "checks": checks, "pass": all(checks.values())}


def run_geometry(sequences: list[CausalSequence], protocol: dict[str, Any], *, seed: int, phase: str = "discovery", locked: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = operator_rows(sequences); train = [r for r in rows if r.partition == "train"]; eval_rows = [r for r in rows if r.partition == "eval"]; left, right, _, _ = _covariance_bases(_as_009a_rows(train)); m = int(protocol["validated_factor_subspace"]["left_dim"]); n = int(protocol["validated_factor_subspace"]["right_dim"])
    carrier = fit_train_carrier(sequences).to(dtype=torch.float64); factors = factor_rows(rows, carrier)
    rank1_oracle = {"train": summarize_approx(train, lambda row: _best_rank1(row.g)[0]), "eval": summarize_approx(eval_rows, lambda row: _best_rank1(row.g)[0])}; dense = {"train": summarize_approx(train, lambda row: _project_dense(row, left, right, m, n)), "eval": summarize_approx(eval_rows, lambda row: _project_dense(row, left, right, m, n))}; rank1_core = {"train": summarize_approx(train, lambda row: _rank1_core(row, left, right, m, n)), "eval": summarize_approx(eval_rows, lambda row: _rank1_core(row, left, right, m, n))}; guard = _rank1_guard(dense["eval"], rank1_core["eval"], protocol, phase=phase)
    max_pca = max(int(x) for x in protocol["global_baselines"]["active_matched_dimensions"]); vector_basis = _vector_pca_basis(train, max_pca); pca_curve = [{"dimension": int(d), "shared_parameter_count": 4096 * int(d), "eval": summarize_approx(eval_rows, lambda row, d=int(d): _vector_pca_approx(row, vector_basis, d))} for d in protocol["global_baselines"]["active_matched_dimensions"]]
    sparse_rows = []
    if locked is None or locked.get("family") == "sparse_tensor": sparse_rows = sparse_tensor_payloads(train, eval_rows, left, right, vector_basis, protocol, seed=seed, phase=phase, locked_active=None if locked is None else int(locked["active_coordinates"]))
    conditional = None
    if locked is None or locked.get("family") == "right_conditioned": conditional = right_conditioned_payload(rows, factors, left, right, protocol, seed=seed, phase=phase)
    return {"format": f"minicells.core-validation.compositional-operator-geometry-{phase}-seed.v1", "experiment_id": protocol["experiment_id"], "seed": int(seed), "scientific_decision": False, "train_count": len(train), "eval_count": len(eval_rows), "carrier_norm": float(torch.linalg.norm(carrier).item()), "factor_spectra": factor_spectra(factors, protocol), "per_write_rank1_oracle": rank1_oracle, "dense_56x8": dense, "rank1_core_56x8": rank1_core, "rank1_core_guard": guard, "vectorized_pca": pca_curve, "sparse_tensor_configs": sparse_rows, "right_conditioned": conditional}


def select_discovery_lock(payloads: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    expected = [int(x) for x in protocol["discovery"]["seeds"]]; completed = sorted(int(p["seed"]) for p in payloads); complete = len(payloads) == len(expected) and completed == sorted(expected); guards_pass = bool(payloads) and all(bool(p["rank1_core_guard"]["pass"]) for p in payloads); sparse_summary = []
    if payloads:
        per_seed = [{int(r["active_coordinates"]): r for r in p.get("sparse_tensor_configs", [])} for p in payloads]; common = sorted(set.intersection(*(set(x) for x in per_seed))) if per_seed else []
        for active in common:
            rs = [x[active] for x in per_seed]; sparse_summary.append({"active_coordinates": active, "all_completed_seed_rows_viable": all(bool(r["viable"]) for r in rs), "per_seed": rs})
    conditional_rows = [p.get("right_conditioned") for p in payloads]; conditional_viable = bool(conditional_rows) and all(r is not None and bool(r["viable"]) for r in conditional_rows); lock = None
    if complete and guards_pass:
        viable_sparse = [r for r in sparse_summary if r["all_completed_seed_rows_viable"]]
        if viable_sparse: lock = {"family": "sparse_tensor", "active_coordinates": int(min(viable_sparse, key=lambda r: int(r["active_coordinates"]))["active_coordinates"])}
        elif conditional_viable: lock = {"family": "right_conditioned"}
    return lock, {"rank1_core_guard_all_completed_seeds": guards_pass, "sparse_tensor": sparse_summary, "right_conditioned_all_completed_seed_rows_viable": conditional_viable, "right_conditioned_per_seed": conditional_rows}
