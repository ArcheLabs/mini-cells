"""Core Validation 009C — Sparse / Local Effect Geometry.

Tests whether the carrier-preserved causal effects rejected by Core 009B-2's
single compact global subspace are nevertheless reusable as either sparse
compositions of an overcomplete dictionary or a train-only union of local
low-dimensional charts.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any

import torch

from .real_representation_009b2_experiment import (
    EffectSequence,
    build_effect_sequences,
    fit_uncentered_basis,
    normalized_residual,
    summarize_basis,
)

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


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / max(float(torch.linalg.norm(v).item()), _EPS)


def _matrix(rows: list[EffectSequence]) -> torch.Tensor:
    return torch.stack([r.effect for r in rows]).to(dtype=torch.float64)


def _stable_order(rows: list[EffectSequence], seed: int, tag: str) -> list[EffectSequence]:
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{tag}:{r.token_sha256}".encode()).hexdigest())


def _generator(seed: int, tag: str) -> torch.Generator:
    h = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    value = int.from_bytes(h[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF
    return torch.Generator(device="cpu").manual_seed(value)


def omp_code(effect: torch.Tensor, dictionary: torch.Tensor, sparsity: int) -> tuple[torch.Tensor, float]:
    """Exact-refit OMP; dictionary has shape [ambient, atoms]."""
    y = effect.to(dtype=torch.float64)
    d = dictionary.to(dtype=torch.float64)
    support: list[int] = []
    coeff = torch.empty(0, dtype=torch.float64)
    residual = y.clone()
    steps = min(int(sparsity), int(d.shape[1]))
    for _ in range(steps):
        scores = torch.abs(d.T @ residual)
        if support:
            scores[torch.tensor(support, dtype=torch.long)] = -1.0
        idx = int(torch.argmax(scores).item())
        if idx in support or float(scores[idx].item()) <= _EPS:
            break
        support.append(idx)
        selected = d[:, support]
        coeff = torch.linalg.lstsq(selected, y[:, None]).solution[:, 0]
        residual = y - selected @ coeff
    full = torch.zeros(d.shape[1], dtype=torch.float64)
    if support:
        full[torch.tensor(support, dtype=torch.long)] = coeff
    denom = max(float(torch.linalg.norm(y).item()), _EPS)
    return full, float(torch.linalg.norm(residual).item()) / denom


def _normalize_dictionary(dictionary: torch.Tensor, *, seed: int, tag: str) -> torch.Tensor:
    d = dictionary.to(dtype=torch.float64).clone()
    gen = _generator(seed, tag)
    for j in range(d.shape[1]):
        norm = float(torch.linalg.norm(d[:, j]).item())
        if norm <= _EPS:
            d[:, j] = torch.randn(d.shape[0], generator=gen, dtype=torch.float64)
        d[:, j] = _unit(d[:, j])
    return d


def initial_dictionary(train: list[EffectSequence], atom_count: int, *, seed: int) -> torch.Tensor:
    ordered = _stable_order(train, seed, "dictionary-init")
    atoms = [_unit(r.effect.to(dtype=torch.float64)) for r in ordered[:atom_count]]
    if len(atoms) < atom_count:
        gen = _generator(seed, "dictionary-fill")
        ambient = int(train[0].effect.numel())
        atoms.extend(_unit(torch.randn(ambient, generator=gen, dtype=torch.float64)) for _ in range(atom_count - len(atoms)))
    return torch.stack(atoms, dim=1)


def fit_sparse_dictionary(
    train: list[EffectSequence], *, atom_count: int, sparsity: int, iterations: int, seed: int
) -> torch.Tensor:
    a = _matrix(train)
    d = initial_dictionary(train, atom_count, seed=seed)
    for iteration in range(int(iterations)):
        codes = torch.stack([omp_code(row.effect, d, sparsity)[0] for row in train])
        # MOD update: min_X ||C X - A||, with X = D^T.
        x = torch.linalg.lstsq(codes, a).solution
        d = _normalize_dictionary(x.T, seed=seed, tag=f"mod-{iteration}")
    return d


def random_dictionary(ambient: int, atom_count: int, *, seed: int) -> torch.Tensor:
    gen = _generator(seed, f"random-dictionary-{atom_count}")
    d = torch.randn(ambient, atom_count, generator=gen, dtype=torch.float64)
    return _normalize_dictionary(d, seed=seed, tag=f"random-dictionary-normalize-{atom_count}")


def summarize_sparse(rows: list[EffectSequence], dictionary: torch.Tensor, sparsity: int) -> dict[str, Any]:
    residuals: list[float] = []
    active: list[int] = []
    for row in rows:
        code, residual = omp_code(row.effect, dictionary, sparsity)
        residuals.append(residual)
        active.append(int(torch.count_nonzero(torch.abs(code) > 1e-10).item()))
    return {
        "count": len(rows),
        "median_normalized_residual": _median(residuals),
        "mean_normalized_residual": _mean(residuals),
        "p90_normalized_residual": _p90(residuals),
        "median_active_coordinates": _median([float(x) for x in active]),
        "maximum_active_coordinates": max(active) if active else 0,
    }


def _description_bits(atom_count: int, sparsity: int) -> dict[str, int]:
    support = int(sparsity) * int(math.ceil(math.log2(max(atom_count, 2))))
    return {"support_bits": support, "with_fp16_coefficients": support + 16 * sparsity, "with_fp32_coefficients": support + 32 * sparsity}


def sparse_config_payload(
    train: list[EffectSequence], eval_rows: list[EffectSequence], *, atom_count: int, sparsity: int, iterations: int, seed: int
) -> dict[str, Any]:
    learned = fit_sparse_dictionary(train, atom_count=atom_count, sparsity=sparsity, iterations=iterations, seed=seed)
    null = random_dictionary(int(train[0].effect.numel()), atom_count, seed=seed)
    return {
        "family": "sparse",
        "atom_count": int(atom_count),
        "sparsity": int(sparsity),
        "description_bits": _description_bits(atom_count, sparsity),
        "train": summarize_sparse(train, learned, sparsity),
        "eval": summarize_sparse(eval_rows, learned, sparsity),
        "null_eval": summarize_sparse(eval_rows, null, sparsity),
    }


def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a / torch.linalg.norm(a, dim=1, keepdim=True).clamp_min(_EPS)
    bb = b / torch.linalg.norm(b, dim=1, keepdim=True).clamp_min(_EPS)
    return aa @ bb.T


def fit_spherical_clusters(train: list[EffectSequence], *, chart_count: int, iterations: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = _matrix(train)
    ordered = _stable_order(train, seed, f"chart-init-{chart_count}")
    centroids = torch.stack([_unit(r.effect.to(dtype=torch.float64)) for r in ordered[:chart_count]])
    assignment = torch.zeros(len(train), dtype=torch.long)
    for _ in range(int(iterations)):
        assignment = torch.argmax(_cosine_matrix(a, centroids), dim=1)
        new = []
        for k in range(chart_count):
            members = a[assignment == k]
            if len(members) == 0:
                new.append(centroids[k])
            else:
                new.append(_unit(members.mean(dim=0)))
        updated = torch.stack(new)
        if torch.equal(torch.argmax(_cosine_matrix(a, updated), dim=1), assignment):
            centroids = updated
            break
        centroids = updated
    assignment = torch.argmax(_cosine_matrix(a, centroids), dim=1)
    return centroids, assignment


def _basis_from_vectors(vectors: torch.Tensor, dim: int) -> torch.Tensor:
    ambient = int(vectors.shape[1])
    if int(vectors.shape[0]) == 0:
        return torch.empty(ambient, 0, dtype=torch.float64)
    cov = vectors.T @ vectors / max(int(vectors.shape[0]), 1)
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, descending=True)
    return vecs[:, order[: min(dim, ambient)]].contiguous()


def fit_local_charts(
    train: list[EffectSequence], *, chart_count: int, local_dim: int, iterations: int, seed: int
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    centroids, assignment = fit_spherical_clusters(train, chart_count=chart_count, iterations=iterations, seed=seed)
    a = _matrix(train)
    bases: list[torch.Tensor] = []
    counts: list[int] = []
    for k in range(chart_count):
        members = a[assignment == k]
        counts.append(int(members.shape[0]))
        bases.append(_basis_from_vectors(members, local_dim))
    return centroids, bases, counts


def summarize_local(rows: list[EffectSequence], centroids: torch.Tensor, bases: list[torch.Tensor]) -> dict[str, Any]:
    a = _matrix(rows)
    assignments = torch.argmax(_cosine_matrix(a, centroids), dim=1)
    residuals = [normalized_residual(row.effect, bases[int(assignments[i].item())]) for i, row in enumerate(rows)]
    return {
        "count": len(rows),
        "median_normalized_residual": _median(residuals),
        "mean_normalized_residual": _mean(residuals),
        "p90_normalized_residual": _p90(residuals),
    }


def _hash_chart(token_sha256: str, seed: int, chart_count: int) -> int:
    h = hashlib.sha256(f"{seed}:null-chart:{token_sha256}".encode()).digest()
    return int.from_bytes(h[:8], "little") % chart_count


def random_chart_null(
    train: list[EffectSequence], eval_rows: list[EffectSequence], *, chart_count: int, local_dim: int, seed: int
) -> dict[str, Any]:
    ambient = int(train[0].effect.numel())
    buckets: list[list[torch.Tensor]] = [[] for _ in range(chart_count)]
    for row in train:
        buckets[_hash_chart(row.token_sha256, seed, chart_count)].append(row.effect.to(dtype=torch.float64))
    bases = [_basis_from_vectors(torch.stack(bucket), local_dim) if bucket else torch.empty(ambient, 0, dtype=torch.float64) for bucket in buckets]
    residuals = [normalized_residual(row.effect, bases[_hash_chart(row.token_sha256, seed, chart_count)]) for row in eval_rows]
    return {"count": len(eval_rows), "median_normalized_residual": _median(residuals), "mean_normalized_residual": _mean(residuals), "p90_normalized_residual": _p90(residuals)}


def local_config_payload(
    train: list[EffectSequence], eval_rows: list[EffectSequence], *, chart_count: int, local_dim: int, iterations: int, seed: int
) -> dict[str, Any]:
    centroids, bases, counts = fit_local_charts(train, chart_count=chart_count, local_dim=local_dim, iterations=iterations, seed=seed)
    return {
        "family": "local",
        "chart_count": int(chart_count),
        "local_dimension": int(local_dim),
        "train_chart_counts": counts,
        "train": summarize_local(train, centroids, bases),
        "eval": summarize_local(eval_rows, centroids, bases),
        "null_eval": random_chart_null(train, eval_rows, chart_count=chart_count, local_dim=local_dim, seed=seed),
    }


def _relative_improvement(baseline: float, value: float) -> float:
    return (float(baseline) - float(value)) / max(float(baseline), _EPS)


def annotate_config(config: dict[str, Any], global32: dict[str, Any], protocol: dict[str, Any], *, phase: str) -> dict[str, Any]:
    out = dict(config)
    median = float(config["eval"]["median_normalized_residual"])
    p90 = float(config["eval"]["p90_normalized_residual"])
    global_median = float(global32["median_normalized_residual"])
    null_median = float(config["null_eval"]["median_normalized_residual"])
    out["relative_median_improvement_over_global32"] = _relative_improvement(global_median, median)
    out["relative_median_improvement_over_matched_null"] = _relative_improvement(null_median, median)
    gates = protocol[phase]["gates"]
    checks = {
        "eval_median": median <= float(gates["maximum_eval_median_residual"]),
        "eval_p90": p90 <= float(gates["maximum_eval_p90_residual"]),
        "beats_global32": out["relative_median_improvement_over_global32"] >= float(gates["minimum_relative_median_improvement_over_global32"]),
        "beats_null": out["relative_median_improvement_over_matched_null"] >= float(gates["minimum_relative_median_improvement_over_matched_null"]),
    }
    if config["family"] == "sparse" and phase == "discovery":
        checks["complexity"] = int(config["sparsity"]) <= int(gates["maximum_sparse_active_coordinates"])
    if config["family"] == "local" and phase == "discovery":
        checks["complexity"] = int(config["local_dimension"]) <= int(gates["maximum_local_dimension"]) and int(config["chart_count"]) <= int(gates["maximum_chart_count"])
    out["checks"] = checks
    out["viable"] = all(checks.values())
    return out


def run_geometry(sequences: list[Any], protocol: dict[str, Any], *, seed: int, phase: str = "discovery", locked: dict[str, Any] | None = None) -> dict[str, Any]:
    effects, carrier = build_effect_sequences(sequences)
    train = [r for r in effects if r.partition == "train"]
    eval_rows = [r for r in effects if r.partition == "eval"]
    basis, _ = fit_uncentered_basis(train)
    global32 = summarize_basis(eval_rows, basis, int(protocol["global_baseline"]["dimension"]))
    sparse_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []

    if locked is None or locked.get("family") == "sparse":
        atom_counts = [int(locked["atom_count"])] if locked else [int(x) for x in protocol["sparse_dictionary"]["atom_counts"]]
        sparsities = [int(locked["sparsity"])] if locked else [int(x) for x in protocol["sparse_dictionary"]["sparsities"]]
        for k in atom_counts:
            for s in sparsities:
                row = sparse_config_payload(train, eval_rows, atom_count=k, sparsity=s, iterations=int(protocol["sparse_dictionary"]["iterations"]), seed=seed)
                sparse_rows.append(annotate_config(row, global32, protocol, phase=phase))

    if locked is None or locked.get("family") == "local":
        chart_counts = [int(locked["chart_count"])] if locked else [int(x) for x in protocol["local_subspaces"]["chart_counts"]]
        local_dims = [int(locked["local_dimension"])] if locked else [int(x) for x in protocol["local_subspaces"]["local_dimensions"]]
        for c in chart_counts:
            for d in local_dims:
                row = local_config_payload(train, eval_rows, chart_count=c, local_dim=d, iterations=int(protocol["local_subspaces"]["iterations"]), seed=seed)
                local_rows.append(annotate_config(row, global32, protocol, phase=phase))

    return {
        "format": f"minicells.core-validation.sparse-local-effect-geometry-{phase}-seed.v1",
        "experiment_id": protocol["experiment_id"],
        "seed": int(seed),
        "scientific_decision": False,
        "carrier_norm": float(torch.linalg.norm(carrier).item()),
        "train_count": len(train),
        "eval_count": len(eval_rows),
        "global32_eval": global32,
        "sparse_configs": sparse_rows,
        "local_configs": local_rows,
    }


def _config_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row["family"] == "sparse":
        return ("sparse", int(row["atom_count"]), int(row["sparsity"]))
    return ("local", int(row["chart_count"]), int(row["local_dimension"]))


def select_discovery_lock(seed_payloads: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not seed_payloads:
        return None, []
    per_seed_maps: list[dict[tuple[Any, ...], dict[str, Any]]] = []
    for payload in seed_payloads:
        rows = payload.get("sparse_configs", []) + payload.get("local_configs", [])
        per_seed_maps.append({_config_key(r): r for r in rows})
    keys = sorted(set.intersection(*(set(m) for m in per_seed_maps))) if per_seed_maps else []
    summary: list[dict[str, Any]] = []
    for key in keys:
        rows = [m[key] for m in per_seed_maps]
        summary.append({"key": list(key), "family": key[0], "all_completed_seed_rows_viable": all(bool(r["viable"]) for r in rows), "per_seed": rows})
    if len(seed_payloads) != len(protocol["discovery"]["seeds"]):
        return None, summary
    viable = [r for r in summary if r["all_completed_seed_rows_viable"]]
    if not viable:
        return None, summary
    def rank(item: dict[str, Any]) -> tuple[float, float, float]:
        rows = item["per_seed"]
        if item["family"] == "sparse":
            complexity = float(rows[0]["sparsity"])
            secondary = float(rows[0]["atom_count"])
        else:
            complexity = float(rows[0]["local_dimension"])
            secondary = float(rows[0]["chart_count"])
        residual = max(float(r["eval"]["median_normalized_residual"]) for r in rows)
        return complexity, residual, secondary
    best = min(viable, key=rank)
    row = best["per_seed"][0]
    if best["family"] == "sparse":
        lock = {"family": "sparse", "atom_count": int(row["atom_count"]), "sparsity": int(row["sparsity"])}
    else:
        lock = {"family": "local", "chart_count": int(row["chart_count"]), "local_dimension": int(row["local_dimension"])}
    return lock, summary
