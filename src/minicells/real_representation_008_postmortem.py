"""Core 008 postmortem functional-capacity diagnostics.

All routines are offline diagnostics over already-observed Core 008 seeds. They
must not mutate or reinterpret the frozen Core 008 scientific decision.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

import torch

_EPS = 1e-12


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def normalize_write(write: torch.Tensor) -> torch.Tensor:
    x = write.detach().cpu().to(torch.float64)
    return x / max(float(torch.linalg.norm(x).item()), _EPS)


def make_rows(projected: list[Any], signatures: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seq in projected:
        out.append({
            "token_sha256": seq.token_sha256,
            "partition": str(seq.partition),
            "z": seq.z.detach().cpu().to(torch.float64),
            "g": normalize_write(signatures[seq.token_sha256].write_matrix),
        })
    return out


def _rank_approx(g: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    r = min(rank, int(s.numel()))
    return (u[:, :r] * s[:r]) @ vh[:r]


def fro_residual(g: torch.Tensor, approx: torch.Tensor) -> float:
    return float(torch.linalg.norm(g - approx).item() / max(torch.linalg.norm(g).item(), _EPS))


def action_residual(z: torch.Tensor, g: torch.Tensor, approx: torch.Tensor) -> float:
    target = z @ g.T
    pred = z @ approx.T
    return float(torch.linalg.norm(target - pred).item() / max(torch.linalg.norm(target).item(), _EPS))


def per_write_svd(rows: list[dict[str, Any]], ranks: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in ("train", "eval"):
        subset = [r for r in rows if r["partition"] == part]
        for rank in ranks:
            fr, ar = [], []
            for row in subset:
                approx = _rank_approx(row["g"], rank)
                fr.append(fro_residual(row["g"], approx))
                ar.append(action_residual(row["z"], row["g"], approx))
            out.append({
                "partition": part,
                "rank": rank,
                "count": len(subset),
                "median_frobenius_residual": _median(fr),
                "mean_frobenius_residual": _mean(fr),
                "median_local_action_residual": _median(ar),
                "mean_local_action_residual": _mean(ar),
            })
    return out


def fit_global_pca(train: list[dict[str, Any]], max_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([r["g"].reshape(-1) for r in train])
    mean = x.mean(dim=0)
    centered = x - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return mean, vh[: min(max_dim, vh.shape[0])].contiguous()


def _pca_coeff(g: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return basis @ (g.reshape(-1) - mean)


def _pca_reconstruct(coeff: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor, dim: int) -> torch.Tensor:
    return (mean + coeff @ basis).reshape(dim, dim)


def global_pca_diagnostics(rows: list[dict[str, Any]], dimensions: list[int], sparsities: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if r["partition"] == "train"]
    eval_rows = [r for r in rows if r["partition"] == "eval"]
    max_dim = max(dimensions)
    mean, full_basis = fit_global_pca(train, max_dim)
    dim = int(train[0]["g"].shape[0])
    dense_rows: list[dict[str, Any]] = []
    for k in dimensions:
        basis = full_basis[:k]
        for part, subset in (("train", train), ("eval", eval_rows)):
            fr, ar = [], []
            for row in subset:
                coeff = _pca_coeff(row["g"], mean, basis)
                approx = _pca_reconstruct(coeff, mean, basis, dim)
                fr.append(fro_residual(row["g"], approx))
                ar.append(action_residual(row["z"], row["g"], approx))
            dense_rows.append({
                "partition": part,
                "dimension": k,
                "count": len(subset),
                "median_frobenius_residual": _median(fr),
                "mean_frobenius_residual": _mean(fr),
                "median_local_action_residual": _median(ar),
                "mean_local_action_residual": _mean(ar),
            })
    sparse_rows: list[dict[str, Any]] = []
    basis = full_basis[:max_dim]
    for sparsity in sparsities:
        k = min(sparsity, basis.shape[0])
        for part, subset in (("train", train), ("eval", eval_rows)):
            fr, ar = [], []
            for row in subset:
                coeff = _pca_coeff(row["g"], mean, basis)
                if k < coeff.numel():
                    keep = torch.topk(torch.abs(coeff), k=k).indices
                    mask = torch.zeros_like(coeff, dtype=torch.bool)
                    mask[keep] = True
                    coeff = torch.where(mask, coeff, torch.zeros_like(coeff))
                approx = _pca_reconstruct(coeff, mean, basis, dim)
                fr.append(fro_residual(row["g"], approx))
                ar.append(action_residual(row["z"], row["g"], approx))
            sparse_rows.append({
                "partition": part,
                "basis_dimension": int(basis.shape[0]),
                "sparsity": k,
                "count": len(subset),
                "median_frobenius_residual": _median(fr),
                "mean_frobenius_residual": _mean(fr),
                "median_local_action_residual": _median(ar),
                "mean_local_action_residual": _mean(ar),
            })
    return dense_rows, sparse_rows


def _omp(g: torch.Tensor, atoms: list[torch.Tensor], top_k: int) -> torch.Tensor:
    if not atoms:
        return torch.zeros(0, dtype=torch.float64)
    d = torch.stack([a.reshape(-1) for a in atoms], dim=1)
    y = g.reshape(-1)
    coeff = torch.zeros(len(atoms), dtype=torch.float64)
    residual = y.clone()
    chosen: list[int] = []
    norms = torch.linalg.norm(d, dim=0).clamp_min(_EPS)
    for _ in range(min(top_k, len(atoms))):
        score = torch.abs(d.T @ residual) / norms
        if chosen:
            score[chosen] = -1.0
        idx = int(torch.argmax(score).item())
        if float(score[idx].item()) <= 1e-12:
            break
        chosen.append(idx)
        sub = d[:, chosen]
        sol = torch.linalg.lstsq(sub, y[:, None]).solution[:, 0]
        residual = y - sub @ sol
        for j, atom_idx in enumerate(chosen):
            coeff[atom_idx] = sol[j]
    return coeff


def _reconstruct_atoms(coeff: torch.Tensor, atoms: list[torch.Tensor]) -> torch.Tensor:
    out = torch.zeros_like(atoms[0])
    for c, atom in zip(coeff, atoms):
        out += float(c.item()) * atom
    return out


def _project_atom_rank(atom: torch.Tensor, rank: int) -> torch.Tensor:
    a = _rank_approx(atom, rank)
    norm = float(torch.linalg.norm(a).item())
    return a / max(norm, _EPS)


def fit_factorized_dictionary(train: list[dict[str, Any]], atom_rank: int, total_rank_units: int, top_k: int, refinement_rounds: int) -> list[torch.Tensor]:
    atom_count = max(1, total_rank_units // atom_rank)
    dim = train[0]["g"].shape[0]
    atoms: list[torch.Tensor] = []
    # Greedy residual seeding: each atom explains the currently worst represented train write.
    for _ in range(atom_count):
        best_row, best_residual, best_norm = None, None, -1.0
        for row in train:
            if atoms:
                coeff = _omp(row["g"], atoms, min(top_k, len(atoms)))
                residual = row["g"] - _reconstruct_atoms(coeff, atoms)
            else:
                residual = row["g"]
            n = float(torch.linalg.norm(residual).item())
            if n > best_norm:
                best_row, best_residual, best_norm = row, residual, n
        if best_row is None or best_residual is None or best_norm <= 1e-10:
            break
        atoms.append(_project_atom_rank(best_residual, atom_rank))
    # Cheap deterministic refinement: replace each atom by the rank-r projection of
    # its coefficient-weighted residual aggregate, then recode on the next sweep.
    for _ in range(refinement_rounds):
        codes = [_omp(r["g"], atoms, min(top_k, len(atoms))) for r in train]
        for j in range(len(atoms)):
            numerator = torch.zeros(dim, dim, dtype=torch.float64)
            denom = 0.0
            for row, code in zip(train, codes):
                alpha = float(code[j].item())
                if abs(alpha) <= 1e-10:
                    continue
                other = torch.zeros(dim, dim, dtype=torch.float64)
                for k, (c, atom) in enumerate(zip(code, atoms)):
                    if k != j:
                        other += float(c.item()) * atom
                numerator += alpha * (row["g"] - other)
                denom += alpha * alpha
            if denom > 1e-12:
                atoms[j] = _project_atom_rank(numerator / denom, atom_rank)
    return atoms


def factorized_dictionary_diagnostics(rows: list[dict[str, Any]], atom_ranks: list[int], total_rank_units: int, top_k: int, refinement_rounds: int) -> list[dict[str, Any]]:
    train = [r for r in rows if r["partition"] == "train"]
    eval_rows = [r for r in rows if r["partition"] == "eval"]
    out: list[dict[str, Any]] = []
    for atom_rank in atom_ranks:
        atoms = fit_factorized_dictionary(train, atom_rank, total_rank_units, top_k, refinement_rounds)
        for part, subset in (("train", train), ("eval", eval_rows)):
            fr, ar, active = [], [], []
            for row in subset:
                coeff = _omp(row["g"], atoms, min(top_k, len(atoms)))
                approx = _reconstruct_atoms(coeff, atoms)
                fr.append(fro_residual(row["g"], approx))
                ar.append(action_residual(row["z"], row["g"], approx))
                active.append(float((torch.abs(coeff) > 1e-10).sum().item()))
            out.append({
                "partition": part,
                "atom_rank": atom_rank,
                "atom_count": len(atoms),
                "total_rank_units": len(atoms) * atom_rank,
                "max_active_atoms": min(top_k, len(atoms)),
                "median_active_atoms": _median(active),
                "count": len(subset),
                "median_frobenius_residual": _median(fr),
                "mean_frobenius_residual": _mean(fr),
                "median_local_action_residual": _median(ar),
                "mean_local_action_residual": _mean(ar),
            })
    return out


def classify(seed_payload: dict[str, Any], reference: float = 0.35) -> str:
    per = seed_payload["per_write_svd"]
    pca = seed_payload["global_pca"]
    fac = seed_payload["factorized_dictionary"]
    per16 = next(r for r in per if r["partition"] == "eval" and int(r["rank"]) == 16)
    pca32 = next(r for r in pca if r["partition"] == "eval" and int(r["dimension"]) == 32)
    fac_eval = [r for r in fac if r["partition"] == "eval"]
    factorized_ok = any(float(r["median_local_action_residual"]) <= reference for r in fac_eval)
    if float(pca32["median_local_action_residual"]) <= reference:
        return "SHARED_LOW_DIMENSIONAL_STRUCTURE_PRESENT" if not factorized_ok else "BUDGET_MATCHED_FACTORIZED_STRUCTURE_PRESENT"
    if float(per16["median_local_action_residual"]) <= reference:
        return "PER_WRITE_LOW_RANK_BUT_NOT_SHARED"
    if factorized_ok:
        return "BUDGET_MATCHED_FACTORIZED_STRUCTURE_PRESENT"
    per8 = next(r for r in per if r["partition"] == "eval" and int(r["rank"]) == 8)
    if float(per8["median_local_action_residual"]) > reference and float(pca32["median_local_action_residual"]) > reference:
        return "NO_STRONG_COMPRESSION_EVIDENCE"
    return "MIXED_CAPACITY_EVIDENCE"


def run_capacity_diagnostics(projected: list[Any], signatures: dict[str, Any], protocol: dict[str, Any], *, seed: int, core008_reference: dict[str, Any]) -> dict[str, Any]:
    rows = make_rows(projected, signatures)
    geom = protocol["write_geometry"]
    budget = protocol["factor_budget"]
    per = per_write_svd(rows, [int(x) for x in geom["ranks"]])
    pca, sparse = global_pca_diagnostics(rows, [int(x) for x in geom["global_dimensions"]], [int(x) for x in geom["sparsities"]])
    fac = factorized_dictionary_diagnostics(
        rows,
        [int(x) for x in budget["dictionary_atom_ranks"]],
        int(budget["total_rank_units"]),
        int(budget["maximum_active_atoms"]),
        int(budget["greedy_refinement_rounds"]),
    )
    payload = {
        "format": "minicells.core008-postmortem.functional-capacity-seed.v1",
        "seed": seed,
        "scientific_decision": False,
        "core008_reference": core008_reference,
        "per_write_svd": per,
        "global_pca": pca,
        "pca_sparsity": sparse,
        "factorized_dictionary": fac,
    }
    payload["classification"] = classify(payload, float(protocol["interpretation_reference"]["core008_target_local_action_residual"]))
    return payload
