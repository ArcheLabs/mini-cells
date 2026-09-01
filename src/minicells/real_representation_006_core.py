"""Core math/state for Core Validation 006.

This module has no Hugging Face dependency. It implements the bounded-state
continual-learning mechanism tested on real frozen language-model representations:
fixed addresses, linear writable Cells, per-address second-moment certificates,
safe free-subspace fitting, and dependency-partitioned mitosis.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import torch

_EPS = 1e-12


def orthonormal_columns(rows: int, cols: int, *, seed: int, dtype=torch.float32) -> torch.Tensor:
    if cols > rows:
        raise ValueError("cols must be <= rows")
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.to(dtype=dtype)


def kmeans(points: torch.Tensor, clusters: int, *, seed: int, iterations: int = 25) -> torch.Tensor:
    """Deterministic small-batch k-means used only to freeze the address router."""
    if points.ndim != 2:
        raise ValueError("points must be [n, d]")
    n = points.shape[0]
    if clusters <= 0 or clusters > n:
        raise ValueError("clusters must be in [1, n]")
    x = points.detach().to(device="cpu", dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    init = torch.randperm(n, generator=g)[:clusters]
    centroids = x[init].clone()
    for _ in range(iterations):
        dist = torch.cdist(x, centroids)
        labels = dist.argmin(dim=1)
        next_centroids = []
        nearest = dist.min(dim=1).values
        for k in range(clusters):
            mask = labels == k
            if bool(mask.any()):
                next_centroids.append(x[mask].mean(dim=0))
            else:
                idx = int(nearest.argmax().item())
                next_centroids.append(x[idx].clone())
                nearest[idx] = -1
        new = torch.stack(next_centroids)
        if torch.allclose(new, centroids, atol=1e-10, rtol=0.0):
            centroids = new
            break
        centroids = new
    return centroids.to(dtype=points.dtype)


def assign_addresses(pooled: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return torch.cdist(pooled.to(dtype=torch.float32), centroids.to(dtype=torch.float32)).argmin(dim=1)


def balanced_base_assignment(
    addresses: torch.Tensor, *, num_addresses: int, base_cells: int
) -> dict[int, int]:
    """Greedy load-balanced initial address -> base Cell assignment."""
    counts = torch.bincount(addresses.to(torch.long), minlength=num_addresses).tolist()
    loads = [0 for _ in range(base_cells)]
    owners: dict[int, int] = {}
    for address in sorted(range(num_addresses), key=lambda a: (-counts[a], a)):
        owner = min(range(base_cells), key=lambda c: (loads[c], c))
        owners[address] = owner
        loads[owner] += counts[address]
    return owners


def covariance(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("z must be [tokens, d]")
    x = z.detach().to(device="cpu", dtype=torch.float64)
    return x.T @ x


def spectrum_metrics(cov: torch.Tensor, *, certificate_energy: float) -> dict[str, float | int]:
    c = cov.detach().to(dtype=torch.float64)
    evals = torch.linalg.eigvalsh(c).clamp_min(0).flip(0)
    total = float(evals.sum().item())
    if total <= _EPS:
        return {
            "rank": 0,
            "certificate_rank": 0,
            "energy_rank_99": 0,
            "participation_rank": 0.0,
            "entropy_rank": 0.0,
            "trace": 0.0,
            "top_fraction": 0.0,
        }
    tol = max(float(evals[0].item()) * 1e-10, 1e-12)
    rank = int((evals > tol).sum().item())

    def energy_rank(frac: float) -> int:
        cumulative = torch.cumsum(evals, dim=0) / total
        target = torch.tensor(frac, dtype=cumulative.dtype)
        return int(torch.searchsorted(cumulative, target).item()) + 1

    probs = evals / total
    participation = float(total * total / max(float((evals * evals).sum().item()), _EPS))
    nz = probs[probs > 0]
    entropy = float(torch.exp(-(nz * torch.log(nz)).sum()).item()) if nz.numel() else 0.0
    return {
        "rank": rank,
        "certificate_rank": energy_rank(certificate_energy),
        "energy_rank_99": energy_rank(0.99),
        "participation_rank": participation,
        "entropy_rank": entropy,
        "trace": total,
        "top_fraction": float(evals[0].item() / total),
    }


def protected_basis(cov: torch.Tensor, *, energy: float) -> torch.Tensor:
    if not (0.0 < energy <= 1.0):
        raise ValueError("energy must be in (0, 1]")
    c = cov.detach().to(device="cpu", dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(c)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    evecs = evecs[:, order]
    total = float(evals.sum().item())
    if total <= _EPS:
        return torch.zeros(c.shape[0], 0, dtype=torch.float64)
    tol = max(float(evals[0].item()) * 1e-10, 1e-12)
    valid = evals > tol
    evals = evals[valid]
    evecs = evecs[:, valid]
    cumulative = torch.cumsum(evals, dim=0) / max(float(evals.sum().item()), _EPS)
    target = torch.tensor(energy, dtype=cumulative.dtype)
    r = int(torch.searchsorted(cumulative, target).item()) + 1
    return evecs[:, :r]


def free_projector(cov: torch.Tensor, *, energy: float) -> torch.Tensor:
    q = protected_basis(cov, energy=energy)
    d = cov.shape[0]
    eye = torch.eye(d, dtype=torch.float64)
    return eye if q.numel() == 0 else eye - q @ q.T


@dataclass(frozen=True)
class FitResult:
    delta: torch.Tensor
    safe_error: float
    unrestricted_error: float
    conflict_fraction: float
    protected_rank: int
    projected_feature_fraction: float


def _ridge_solve(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    d = x.shape[1]
    lhs = x.T @ x + ridge * torch.eye(d, dtype=x.dtype)
    rhs = x.T @ y
    return torch.linalg.solve(lhs, rhs)


def fit_functional_delta(
    z: torch.Tensor,
    desired_coeff_delta: torch.Tensor,
    *,
    cov: torch.Tensor,
    certificate_energy: float,
    ridge: float,
    safe: bool,
    maximum_delta_norm: float | None = None,
) -> FitResult:
    """Fit Delta A where coefficient delta is z @ DeltaA.T.

    Safe updates satisfy DeltaA Q = 0 by parameterizing DeltaA = B P_free.
    """
    if z.ndim != 2 or desired_coeff_delta.ndim != 2:
        raise ValueError("z and desired_coeff_delta must be matrices")
    if z.shape != desired_coeff_delta.shape:
        raise ValueError("Core 006 v1 uses square d->d Cells")
    x = z.detach().to(device="cpu", dtype=torch.float64)
    y = desired_coeff_delta.detach().to(device="cpu", dtype=torch.float64)
    denom = max(float((y * y).sum().item()), _EPS)

    unrestricted_bt = _ridge_solve(x, y, ridge)
    unrestricted_pred = x @ unrestricted_bt
    unrestricted_error = float(((unrestricted_pred - y) ** 2).sum().item() / denom)

    if safe:
        p = free_projector(cov, energy=certificate_energy)
        xp = x @ p
        bt = _ridge_solve(xp, y, ridge)
        delta = bt.T @ p
        pred = x @ delta.T
        q = protected_basis(cov, energy=certificate_energy)
        protected_rank = q.shape[1]
        projected_fraction = float(
            (xp * xp).sum().item() / max(float((x * x).sum().item()), _EPS)
        )
    else:
        delta = unrestricted_bt.T
        pred = unrestricted_pred
        protected_rank = 0
        projected_fraction = 1.0

    if maximum_delta_norm is not None:
        norm = float(torch.linalg.norm(delta).item())
        if norm > maximum_delta_norm > 0:
            delta = delta * (maximum_delta_norm / norm)
            pred = x @ delta.T

    safe_error = float(((pred - y) ** 2).sum().item() / denom)
    residual_room = max(1.0 - unrestricted_error, _EPS)
    conflict = max(0.0, min(1.0, (safe_error - unrestricted_error) / residual_room))
    return FitResult(
        delta=delta,
        safe_error=safe_error,
        unrestricted_error=unrestricted_error,
        conflict_fraction=conflict,
        protected_rank=protected_rank,
        projected_feature_fraction=projected_fraction,
    )


@dataclass
class AddressStats:
    cov: torch.Tensor
    tokens: int = 0
    sequences: int = 0

    @classmethod
    def empty(cls, dim: int) -> "AddressStats":
        return cls(cov=torch.zeros(dim, dim, dtype=torch.float64))

    def register(self, z: torch.Tensor, *, sequences: int = 1) -> None:
        self.cov += covariance(z)
        self.tokens += int(z.shape[0])
        self.sequences += int(sequences)


@dataclass
class CellState:
    cell_id: int
    a: torch.Tensor
    parent_id: int | None
    birth_transaction: int
    split_address: int | None = None


@dataclass
class CellSystem:
    dim: int
    num_addresses: int
    certificate_energy: float
    address_owner: dict[int, int]
    cells: dict[int, CellState]
    address_stats: dict[int, AddressStats]
    next_cell_id: int
    splits: list[dict] = field(default_factory=list)

    @classmethod
    def initialize(
        cls,
        *,
        dim: int,
        num_addresses: int,
        base_cells: int,
        address_owner: dict[int, int],
        certificate_energy: float,
    ) -> "CellSystem":
        cells = {
            i: CellState(
                cell_id=i,
                a=torch.zeros(dim, dim, dtype=torch.float64),
                parent_id=None,
                birth_transaction=-1,
            )
            for i in range(base_cells)
        }
        return cls(
            dim=dim,
            num_addresses=num_addresses,
            certificate_energy=certificate_energy,
            address_owner=dict(address_owner),
            cells=cells,
            address_stats={a: AddressStats.empty(dim) for a in range(num_addresses)},
            next_cell_id=base_cells,
        )

    def clone(self) -> "CellSystem":
        return copy.deepcopy(self)

    def addresses_for_cell(self, cell_id: int) -> list[int]:
        return sorted(a for a, owner in self.address_owner.items() if owner == cell_id)

    def cell_covariance(self, cell_id: int) -> torch.Tensor:
        out = torch.zeros(self.dim, self.dim, dtype=torch.float64)
        for address in self.addresses_for_cell(cell_id):
            out += self.address_stats[address].cov
        return out

    def dependency_tokens(self, cell_id: int) -> int:
        return sum(self.address_stats[a].tokens for a in self.addresses_for_cell(cell_id))

    def dependency_sequences(self, cell_id: int) -> int:
        return sum(self.address_stats[a].sequences for a in self.addresses_for_cell(cell_id))

    def metrics(self, cell_id: int) -> dict[str, float | int]:
        spec = spectrum_metrics(
            self.cell_covariance(cell_id), certificate_energy=self.certificate_energy
        )
        dep_tokens = self.dependency_tokens(cell_id)
        dep_sequences = self.dependency_sequences(cell_id)
        participation = float(spec["participation_rank"])
        return {
            "cell_id": cell_id,
            "parent_id": -1 if self.cells[cell_id].parent_id is None else self.cells[cell_id].parent_id,
            "address_count": len(self.addresses_for_cell(cell_id)),
            "dependency_tokens": dep_tokens,
            "dependency_sequences": dep_sequences,
            "reuse_density": dep_tokens / max(participation, 1.0),
            **spec,
        }

    def split_address(self, address: int, *, transaction: int) -> dict:
        parent = self.address_owner[address]
        owned = self.addresses_for_cell(parent)
        if len(owned) <= 1:
            raise ValueError("cannot split the only address from a Cell in v1")
        before = self.metrics(parent)
        child_id = self.next_cell_id
        self.next_cell_id += 1
        self.cells[child_id] = CellState(
            cell_id=child_id,
            a=self.cells[parent].a.clone(),
            parent_id=parent,
            birth_transaction=transaction,
            split_address=address,
        )
        self.address_owner[address] = child_id
        parent_after = self.metrics(parent)
        child_after = self.metrics(child_id)
        record = {
            "transaction": transaction,
            "address": address,
            "parent_id": parent,
            "child_id": child_id,
            "parent_rank_before": before["certificate_rank"],
            "parent_rank_after": parent_after["certificate_rank"],
            "child_rank_after": child_after["certificate_rank"],
            "parent_dependency_before": before["dependency_tokens"],
            "parent_dependency_after": parent_after["dependency_tokens"],
            "child_dependency_after": child_after["dependency_tokens"],
        }
        self.splits.append(record)
        return record

    def register(self, address: int, z: torch.Tensor, *, sequences: int = 1) -> None:
        self.address_stats[address].register(z, sequences=sequences)


@dataclass(frozen=True)
class AddressBatch:
    address: int
    z: torch.Tensor
    desired: torch.Tensor
    sequences: int


def group_by_owner(system: CellSystem, batches: Iterable[AddressBatch]) -> dict[int, list[AddressBatch]]:
    out: dict[int, list[AddressBatch]] = {}
    for batch in batches:
        owner = system.address_owner[batch.address]
        out.setdefault(owner, []).append(batch)
    return out


def address_conflicts(
    system: CellSystem,
    batches: Iterable[AddressBatch],
    *,
    ridge: float,
    maximum_delta_norm: float | None,
) -> dict[int, FitResult]:
    out: dict[int, FitResult] = {}
    for batch in batches:
        owner = system.address_owner[batch.address]
        out[batch.address] = fit_functional_delta(
            batch.z,
            batch.desired,
            cov=system.cell_covariance(owner),
            certificate_energy=system.certificate_energy,
            ridge=ridge,
            safe=True,
            maximum_delta_norm=maximum_delta_norm,
        )
    return out


def apply_transaction_fit(
    system: CellSystem,
    batches: list[AddressBatch],
    *,
    safe: bool,
    ridge: float,
    maximum_delta_norm: float | None,
) -> dict[int, FitResult]:
    fits: dict[int, FitResult] = {}
    for owner, owner_batches in group_by_owner(system, batches).items():
        z = torch.cat([b.z for b in owner_batches], dim=0)
        desired = torch.cat([b.desired for b in owner_batches], dim=0)
        fit = fit_functional_delta(
            z,
            desired,
            cov=system.cell_covariance(owner),
            certificate_energy=system.certificate_energy,
            ridge=ridge,
            safe=safe,
            maximum_delta_norm=maximum_delta_norm,
        )
        system.cells[owner].a += fit.delta
        fits[owner] = fit
    return fits


def register_batches(system: CellSystem, batches: Iterable[AddressBatch]) -> None:
    for batch in batches:
        system.register(batch.address, batch.z, sequences=batch.sequences)
