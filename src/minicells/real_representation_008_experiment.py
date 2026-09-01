"""Core Validation 008 — Certified Adaptive Functional Atoms.

Operate on frozen Core 006/007 projected representations and foundation-path
write-demand signatures. The experiment forms sparse low-rank functional atoms
online while every mutation of an already-used atom is projected through a
bounded activation-subspace certificate.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import torch

_EPS = 1e-12


@dataclass
class FunctionalTarget:
    token_sha256: str
    partition: str
    pooled_z: torch.Tensor
    z: torch.Tensor
    target: torch.Tensor


@dataclass
class CertifiedAtom:
    atom_id: int
    matrix: torch.Tensor
    rank_units: int
    max_rank: int
    covariance: torch.Tensor
    q: torch.Tensor
    key_sum: torch.Tensor
    key_weight: float = 0.0
    uses: int = 0

    @property
    def key(self) -> torch.Tensor:
        if self.key_weight <= 0:
            return torch.zeros_like(self.key_sum)
        return self.key_sum / self.key_weight


@dataclass
class VariantSpec:
    name: str
    maximum_atoms: int
    maximum_rank_per_atom: int
    adaptive_append_rank: bool
    maximum_append_rank_per_action: int = 1


@dataclass
class VariantState:
    spec: VariantSpec
    dim: int
    certificate_energy: float
    factor_budget: int
    atoms: list[CertifiedAtom] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    update_records: list[dict[str, Any]] = field(default_factory=list)
    hidden_history: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    @property
    def used_factor_scalars(self) -> int:
        return sum(a.rank_units for a in self.atoms) * 2 * self.dim

    @property
    def remaining_rank_units(self) -> int:
        return max(0, (self.factor_budget - self.used_factor_scalars) // (2 * self.dim))


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def normalize_write(write: torch.Tensor) -> torch.Tensor:
    x = write.detach().cpu().to(dtype=torch.float64)
    return x / max(float(torch.linalg.norm(x).item()), _EPS)


def make_targets(projected: list[Any], signatures: dict[str, Any]) -> list[FunctionalTarget]:
    out: list[FunctionalTarget] = []
    for seq in projected:
        sig = signatures[seq.token_sha256]
        out.append(
            FunctionalTarget(
                token_sha256=seq.token_sha256,
                partition=str(seq.partition),
                pooled_z=seq.pooled.detach().cpu().to(torch.float64),
                z=seq.z.detach().cpu().to(torch.float64),
                target=normalize_write(sig.write_matrix),
            )
        )
    return out


def _low_rank(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0 or float(torch.linalg.norm(matrix).item()) <= _EPS:
        return torch.zeros_like(matrix)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    r = min(rank, int((s > 1e-12).sum().item()))
    if r <= 0:
        return torch.zeros_like(matrix)
    return (u[:, :r] * s[:r]) @ vh[:r]


def _certificate_from_covariance(cov: torch.Tensor, energy: float) -> torch.Tensor:
    dim = cov.shape[0]
    total = float(torch.trace(cov).item())
    if total <= _EPS:
        return torch.zeros(dim, 0, dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, descending=True)
    vals = vals[order].clamp_min(0.0)
    vecs = vecs[:, order]
    csum = torch.cumsum(vals, dim=0) / max(float(vals.sum().item()), _EPS)
    hits = torch.nonzero(csum >= energy)
    rank = int(hits[0].item()) + 1 if len(hits) else dim
    return vecs[:, :rank].contiguous()


def _refresh_dependency(atom: CertifiedAtom, z: torch.Tensor, pooled: torch.Tensor, weight: float, energy: float) -> None:
    atom.covariance += z.T @ z
    atom.q = _certificate_from_covariance(atom.covariance, energy)
    w = max(abs(float(weight)), 1e-6)
    atom.key_sum += w * pooled
    atom.key_weight += w
    atom.uses += 1


def _safe_delta(target: torch.Tensor, q: torch.Tensor, rank: int) -> torch.Tensor:
    dim = target.shape[1]
    if q.numel() == 0:
        projected = target
    else:
        p_free = torch.eye(dim, dtype=torch.float64) - q @ q.T
        projected = target @ p_free
    return _low_rank(projected, rank)


def _flatten_atoms(atoms: list[CertifiedAtom]) -> torch.Tensor:
    if not atoms:
        return torch.zeros(0, 0, dtype=torch.float64)
    return torch.stack([a.matrix.reshape(-1) for a in atoms], dim=1)


def sparse_coefficients(target: torch.Tensor, atoms: list[CertifiedAtom], top_k: int) -> torch.Tensor:
    n = len(atoms)
    if n == 0:
        return torch.zeros(0, dtype=torch.float64)
    dictionary = _flatten_atoms(atoms)
    y = target.reshape(-1)
    residual = y.clone()
    selected: list[int] = []
    coeff = torch.zeros(n, dtype=torch.float64)
    for _ in range(min(top_k, n)):
        norms = torch.linalg.norm(dictionary, dim=0).clamp_min(_EPS)
        scores = torch.abs(dictionary.T @ residual) / norms
        if selected:
            scores[selected] = -1.0
        idx = int(torch.argmax(scores).item())
        if float(scores[idx].item()) <= 1e-10:
            break
        selected.append(idx)
        d = dictionary[:, selected]
        sol = torch.linalg.lstsq(d, y.unsqueeze(1)).solution[:, 0]
        residual = y - d @ sol
        for j, atom_idx in enumerate(selected):
            coeff[atom_idx] = sol[j]
        if float(torch.linalg.norm(residual).item()) <= 1e-10:
            break
    return coeff


def reconstruct(coeff: torch.Tensor, atoms: list[CertifiedAtom], dim: int) -> torch.Tensor:
    if not atoms or coeff.numel() == 0:
        return torch.zeros(dim, dim, dtype=torch.float64)
    out = torch.zeros(dim, dim, dtype=torch.float64)
    for c, atom in zip(coeff, atoms):
        out += float(c.item()) * atom.matrix
    return out


def frobenius_residual(target: torch.Tensor, approx: torch.Tensor) -> float:
    return float(torch.linalg.norm(target - approx).item() / max(torch.linalg.norm(target).item(), _EPS))


def local_action_residual(z: torch.Tensor, target: torch.Tensor, approx: torch.Tensor) -> float:
    desired = z @ target.T
    actual = z @ approx.T
    return float(torch.linalg.norm(desired - actual).item() / max(torch.linalg.norm(desired).item(), _EPS))


def _constraint_violation(delta: torch.Tensor, q: torch.Tensor) -> float:
    if q.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(delta @ q).item() / max(torch.linalg.norm(delta).item(), _EPS))


def _hidden_drift(delta: torch.Tensor, history: list[torch.Tensor]) -> float:
    if not history:
        return 0.0
    z = torch.cat(history, dim=0)
    drift = z @ delta.T
    denom = max(float(torch.linalg.norm(z).item() * torch.linalg.norm(delta).item()), _EPS)
    return float(torch.linalg.norm(drift).item() / denom)


def _spawn(state: VariantState, residual: torch.Tensor) -> int | None:
    if len(state.atoms) >= state.spec.maximum_atoms or state.remaining_rank_units < 1:
        return None
    matrix = _low_rank(residual, 1)
    if float(torch.linalg.norm(matrix).item()) <= 1e-10:
        return None
    atom_id = len(state.atoms)
    state.atoms.append(
        CertifiedAtom(
            atom_id=atom_id,
            matrix=matrix,
            rank_units=1,
            max_rank=state.spec.maximum_rank_per_atom,
            covariance=torch.zeros(state.dim, state.dim, dtype=torch.float64),
            q=torch.zeros(state.dim, 0, dtype=torch.float64),
            key_sum=torch.zeros(state.dim, dtype=torch.float64),
        )
    )
    state.hidden_history[atom_id] = []
    return atom_id


def _candidate_append_ranks(state: VariantState, atom: CertifiedAtom) -> list[int]:
    room = min(atom.max_rank - atom.rank_units, state.remaining_rank_units)
    if room <= 0:
        return []
    if state.spec.adaptive_append_rank:
        return list(range(1, min(room, state.spec.maximum_append_rank_per_action) + 1))
    return [1]


def _best_safe_update(
    state: VariantState,
    target: FunctionalTarget,
    coeff: torch.Tensor,
    residual: torch.Tensor,
    top_k: int,
    target_residual: float,
) -> tuple[int, torch.Tensor, int, float] | None:
    active = [i for i, c in enumerate(coeff.tolist()) if abs(float(c)) > 1e-8]
    if state.spec.name == "monolithic_certified" and state.atoms:
        active = [0]
    best: tuple[int, torch.Tensor, int, float] | None = None
    before = frobenius_residual(target.target, target.target - residual)
    for idx in active:
        atom = state.atoms[idx]
        alpha = float(coeff[idx].item()) if idx < coeff.numel() else 1.0
        if abs(alpha) < 1e-5:
            alpha = 1.0
        for rank in _candidate_append_ranks(state, atom):
            delta = _safe_delta(residual / alpha, atom.q, rank)
            if float(torch.linalg.norm(delta).item()) <= 1e-10:
                continue
            old = atom.matrix
            atom.matrix = old + delta
            test_coeff = sparse_coefficients(target.target, state.atoms, top_k)
            test_recon = reconstruct(test_coeff, state.atoms, state.dim)
            after = frobenius_residual(target.target, test_recon)
            atom.matrix = old
            improvement = before - after
            score = improvement / max(rank, 1)
            if best is None or score > best[3] + 1e-12 or (abs(score - best[3]) <= 1e-12 and rank < best[2]):
                best = (idx, delta, rank, score)
            if state.spec.adaptive_append_rank and after <= target_residual:
                break
    return best


def train_variant(
    targets: list[FunctionalTarget],
    spec: VariantSpec,
    *,
    certificate_energy: float,
    factor_budget: int,
    top_k: int,
    target_residual: float,
    minimum_update_improvement: float,
    max_growth_actions: int,
) -> VariantState:
    dim = int(targets[0].target.shape[0])
    state = VariantState(spec=spec, dim=dim, certificate_energy=certificate_energy, factor_budget=factor_budget)
    train = [x for x in targets if x.partition == "train"]
    for step, target in enumerate(train):
        atoms_before = len(state.atoms)
        rank_before = sum(a.rank_units for a in state.atoms)
        reused_existing = False
        growth_actions = 0
        coeff = sparse_coefficients(target.target, state.atoms, top_k)
        recon = reconstruct(coeff, state.atoms, dim)
        residual_value = frobenius_residual(target.target, recon)
        while residual_value > target_residual and growth_actions < max_growth_actions:
            residual = target.target - recon
            update = _best_safe_update(state, target, coeff, residual, top_k, target_residual)
            applied = False
            if update is not None:
                idx, delta, rank, improvement_per_rank = update
                improvement = improvement_per_rank * rank
                if improvement >= minimum_update_improvement:
                    atom = state.atoms[idx]
                    violation = _constraint_violation(delta, atom.q)
                    drift = _hidden_drift(delta, state.hidden_history.get(idx, []))
                    atom.matrix += delta
                    atom.rank_units += rank
                    state.update_records.append(
                        {
                            "step": step,
                            "atom_id": idx,
                            "rank_added": rank,
                            "constraint_violation": violation,
                            "hidden_history_drift": drift,
                            "improvement": improvement,
                        }
                    )
                    reused_existing = atom.uses > 0
                    applied = True
            if not applied:
                if _spawn(state, residual) is None:
                    break
                applied = True
            growth_actions += 1
            coeff = sparse_coefficients(target.target, state.atoms, top_k)
            recon = reconstruct(coeff, state.atoms, dim)
            residual_value = frobenius_residual(target.target, recon)
        coeff = sparse_coefficients(target.target, state.atoms, top_k)
        recon = reconstruct(coeff, state.atoms, dim)
        active = [i for i, c in enumerate(coeff.tolist()) if abs(float(c)) > 1e-8]
        for idx in active:
            atom = state.atoms[idx]
            if atom.uses > 0:
                reused_existing = True
            _refresh_dependency(atom, target.z, target.pooled_z, float(coeff[idx].item()), certificate_energy)
            state.hidden_history[idx].append(target.z.clone())
        state.records.append(
            {
                "step": step,
                "token_sha256": target.token_sha256,
                "frobenius_residual": frobenius_residual(target.target, recon),
                "local_action_residual": local_action_residual(target.z, target.target, recon),
                "active_atoms": active,
                "active_count": len(active),
                "reused_existing": reused_existing,
                "atoms_spawned": len(state.atoms) - atoms_before,
                "rank_units_added": sum(a.rank_units for a in state.atoms) - rank_before,
                "unresolved": frobenius_residual(target.target, recon) > target_residual,
                "factor_budget_fraction": state.used_factor_scalars / factor_budget,
            }
        )
    return state


def _final_oracle_coefficients(targets: list[FunctionalTarget], state: VariantState, top_k: int) -> torch.Tensor:
    if not targets:
        return torch.zeros(0, len(state.atoms), dtype=torch.float64)
    return torch.stack([sparse_coefficients(t.target, state.atoms, top_k) for t in targets])


def fit_router(train: list[FunctionalTarget], coeff: torch.Tensor, ridge: float) -> torch.Tensor:
    if coeff.shape[1] == 0:
        return torch.zeros(train[0].pooled_z.numel() + 1, 0, dtype=torch.float64)
    x = torch.stack([t.pooled_z for t in train])
    x = torch.cat([x, torch.ones(len(train), 1, dtype=torch.float64)], dim=1)
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + ridge * eye, x.T @ coeff)


def deploy_coefficients(targets: list[FunctionalTarget], router: torch.Tensor, top_k: int) -> torch.Tensor:
    if not targets:
        return torch.zeros(0, router.shape[1], dtype=torch.float64)
    x = torch.stack([t.pooled_z for t in targets])
    x = torch.cat([x, torch.ones(len(targets), 1, dtype=torch.float64)], dim=1)
    out = x @ router
    if out.shape[1] > top_k:
        keep = torch.topk(torch.abs(out), k=top_k, dim=1).indices
        mask = torch.zeros_like(out, dtype=torch.bool)
        mask.scatter_(1, keep, True)
        out = torch.where(mask, out, torch.zeros_like(out))
    return out


def _coefficient_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    den = float(torch.linalg.norm(a).item() * torch.linalg.norm(b).item())
    if den <= _EPS:
        return 1.0 if float(torch.linalg.norm(a - b).item()) <= _EPS else 0.0
    return float(torch.dot(a, b).item() / den)


def evaluate_variant(targets: list[FunctionalTarget], state: VariantState, *, top_k: int, router_ridge: float) -> dict[str, Any]:
    train = [x for x in targets if x.partition == "train"]
    eval_rows = [x for x in targets if x.partition == "eval"]
    train_coeff = _final_oracle_coefficients(train, state, top_k)
    eval_oracle = _final_oracle_coefficients(eval_rows, state, top_k)
    router = fit_router(train, train_coeff, router_ridge)
    eval_deploy = deploy_coefficients(eval_rows, router, top_k)
    per_eval: list[dict[str, Any]] = []
    for i, target in enumerate(eval_rows):
        oc, dc = eval_oracle[i], eval_deploy[i]
        orecon = reconstruct(oc, state.atoms, state.dim)
        drecon = reconstruct(dc, state.atoms, state.dim)
        os = {j for j, c in enumerate(oc.tolist()) if abs(float(c)) > 1e-8}
        ds = {j for j, c in enumerate(dc.tolist()) if abs(float(c)) > 1e-8}
        per_eval.append(
            {
                "token_sha256": target.token_sha256,
                "oracle_frobenius_residual": frobenius_residual(target.target, orecon),
                "oracle_local_action_residual": local_action_residual(target.z, target.target, orecon),
                "deploy_frobenius_residual": frobenius_residual(target.target, drecon),
                "deploy_local_action_residual": local_action_residual(target.z, target.target, drecon),
                "coefficient_cosine": _coefficient_cosine(oc, dc),
                "active_set_overlap": len(os & ds) / max(len(os | ds), 1),
            }
        )
    q_fracs = [a.q.shape[1] / state.dim for a in state.atoms]
    violations = [float(r["constraint_violation"]) for r in state.update_records]
    drifts = [float(r["hidden_history_drift"]) for r in state.update_records]
    online = state.records
    return {
        "variant": state.spec.name,
        "atom_count": len(state.atoms),
        "total_rank_units": sum(a.rank_units for a in state.atoms),
        "factor_scalars": state.used_factor_scalars,
        "factor_budget_fraction": state.used_factor_scalars / state.factor_budget,
        "online_reuse_fraction": _mean([1.0 if r["reused_existing"] else 0.0 for r in online]),
        "spawned_atoms_per_train_sequence": len(state.atoms) / max(len(online), 1),
        "unresolved_write_fraction": _mean([1.0 if r["unresolved"] else 0.0 for r in online]),
        "median_online_local_action_residual": _median([float(r["local_action_residual"]) for r in online]),
        "median_certificate_rank_fraction": _median(q_fracs),
        "maximum_certificate_rank_fraction": max(q_fracs, default=0.0),
        "maximum_certificate_constraint_violation": max(violations, default=0.0),
        "p95_hidden_history_drift": _percentile(drifts, 0.95),
        "median_hidden_history_drift": _median(drifts),
        "median_eval_oracle_frobenius_residual": _median([float(r["oracle_frobenius_residual"]) for r in per_eval]),
        "median_eval_oracle_local_action_residual": _median([float(r["oracle_local_action_residual"]) for r in per_eval]),
        "median_eval_deploy_frobenius_residual": _median([float(r["deploy_frobenius_residual"]) for r in per_eval]),
        "median_eval_deploy_local_action_residual": _median([float(r["deploy_local_action_residual"]) for r in per_eval]),
        "median_coefficient_cosine": _median([float(r["coefficient_cosine"]) for r in per_eval]),
        "median_active_set_overlap": _median([float(r["active_set_overlap"]) for r in per_eval]),
        "eval_records": per_eval,
        "online_records": online,
        "update_records": state.update_records,
        "atom_metrics": [
            {
                "atom_id": a.atom_id,
                "rank_units": a.rank_units,
                "uses": a.uses,
                "certificate_rank": a.q.shape[1],
                "certificate_rank_fraction": a.q.shape[1] / state.dim,
            }
            for a in state.atoms
        ],
    }


def run_seed(projected: list[Any], signatures: dict[str, Any], protocol: dict[str, Any], *, seed: int) -> dict[str, Any]:
    targets = make_targets(projected, signatures)
    comp, cert, budget = protocol["composition"], protocol["certificate"], protocol["budget"]
    variants: dict[str, Any] = {}
    for name, row in protocol["variants"].items():
        spec = VariantSpec(
            name=name,
            maximum_atoms=int(row["maximum_atoms"]),
            maximum_rank_per_atom=int(row["maximum_rank_per_atom"]),
            adaptive_append_rank=bool(row.get("adaptive_append_rank", False)),
            maximum_append_rank_per_action=int(row.get("maximum_append_rank_per_action", 1)),
        )
        state = train_variant(
            targets,
            spec,
            certificate_energy=float(cert["energy"]),
            factor_budget=int(budget["conceptual_factor_scalars"]),
            top_k=int(comp["maximum_active_atoms"]),
            target_residual=float(comp["target_normalized_frobenius_residual"]),
            minimum_update_improvement=float(comp["minimum_safe_update_improvement"]),
            max_growth_actions=int(comp["maximum_growth_actions_per_sequence"]),
        )
        variants[name] = evaluate_variant(
            targets,
            state,
            top_k=int(comp["maximum_active_atoms"]),
            router_ridge=float(comp["router_ridge"]),
        )
    return {
        "format": "minicells.core-validation.certified-functional-atoms.seed.v1",
        "experiment_id": "core-validation-008",
        "seed": seed,
        "scientific_decision": False,
        "variant_results": variants,
    }
