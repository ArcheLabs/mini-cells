"""Bounded functional-boundary math for Core Validation 007.

Core 006 showed that semantic/address identity is not a sufficient mitosis
boundary.  This module keeps the frozen read representation but adds bounded
write-demand modes and three geometry families:

* activation geometry from z second moments;
* write geometry from low-rank bases of vec(u z^T);
* direct cross-write interference estimated from a mode's mean write matrix
  acting on another mode's activation covariance.

No raw historical examples are retained by these objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .real_representation_006_core import protected_basis, spectrum_metrics

_EPS = 1e-12


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    y = b.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denom = float(torch.linalg.norm(x).item() * torch.linalg.norm(y).item())
    if denom <= _EPS:
        return 0.0
    return float(torch.dot(x, y).item() / denom)


def update_low_rank_basis(
    basis: torch.Tensor,
    vector: torch.Tensor,
    *,
    maximum_rank: int,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Incremental deterministic Gram-Schmidt sketch.

    The sketch is intentionally bounded.  It is not an exact history matrix;
    it preserves at most ``maximum_rank`` independent write-demand directions.
    """
    if maximum_rank <= 0:
        return torch.zeros(vector.numel(), 0, dtype=torch.float64)
    v = vector.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    norm = float(torch.linalg.norm(v).item())
    if norm <= _EPS:
        return basis
    v = v / norm
    q = basis.detach().to(device="cpu", dtype=torch.float64)
    if q.numel():
        v = v - q @ (q.T @ v)
        # Re-orthogonalize once for numerical stability.
        v = v - q @ (q.T @ v)
    residual = float(torch.linalg.norm(v).item())
    if residual <= tolerance or q.shape[1] >= maximum_rank:
        return q
    v = v / residual
    return torch.cat([q, v[:, None]], dim=1)


@dataclass
class FunctionalMode:
    mode_id: int
    address: int
    dim: int
    maximum_write_rank: int
    z_cov: torch.Tensor = field(init=False)
    z_proto_sum: torch.Tensor = field(init=False)
    prototype_count: int = 0
    dependency_tokens: int = 0
    dependency_sequences: int = 0
    write_mean: torch.Tensor = field(init=False)
    write_count: int = 0
    write_basis: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.z_cov = torch.zeros(self.dim, self.dim, dtype=torch.float64)
        self.z_proto_sum = torch.zeros(self.dim, dtype=torch.float64)
        self.write_mean = torch.zeros(self.dim, self.dim, dtype=torch.float64)
        self.write_basis = torch.zeros(
            self.dim * self.dim, 0, dtype=torch.float64
        )

    @property
    def z_prototype(self) -> torch.Tensor:
        if self.prototype_count <= 0:
            return torch.zeros(self.dim, dtype=torch.float64)
        return self.z_proto_sum / self.prototype_count

    def observe_signature(self, pooled_z: torch.Tensor, write_matrix: torch.Tensor) -> None:
        z = pooled_z.detach().to(device="cpu", dtype=torch.float64).reshape(self.dim)
        w = write_matrix.detach().to(device="cpu", dtype=torch.float64).reshape(
            self.dim, self.dim
        )
        self.z_proto_sum += z
        self.prototype_count += 1
        self.write_count += 1
        self.write_mean += (w - self.write_mean) / self.write_count
        self.write_basis = update_low_rank_basis(
            self.write_basis,
            w,
            maximum_rank=self.maximum_write_rank,
        )

    def register_dependency(self, z: torch.Tensor, *, sequences: int = 1) -> None:
        x = z.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("dependency z must be [tokens, dim]")
        self.z_cov += x.T @ x
        self.dependency_tokens += int(x.shape[0])
        self.dependency_sequences += int(sequences)


@dataclass
class FunctionalModeCatalog:
    dim: int
    maximum_modes_per_address: int
    maximum_write_rank: int
    creation_cosine_threshold: float
    modes: dict[int, FunctionalMode] = field(default_factory=dict)
    address_modes: dict[int, list[int]] = field(default_factory=dict)
    next_mode_id: int = 0

    def modes_for_address(self, address: int) -> list[int]:
        return list(self.address_modes.get(int(address), []))

    def locate_or_create(
        self,
        *,
        address: int,
        pooled_z: torch.Tensor,
        write_matrix: torch.Tensor,
    ) -> tuple[int, bool, float]:
        ids = self.modes_for_address(address)
        best_id: int | None = None
        best = -2.0
        for mode_id in ids:
            score = cosine(write_matrix, self.modes[mode_id].write_mean)
            if score > best:
                best_id, best = mode_id, score
        create = (
            best_id is None
            or (
                best < self.creation_cosine_threshold
                and len(ids) < self.maximum_modes_per_address
            )
        )
        if create:
            best_id = self.next_mode_id
            self.next_mode_id += 1
            mode = FunctionalMode(
                mode_id=best_id,
                address=int(address),
                dim=self.dim,
                maximum_write_rank=self.maximum_write_rank,
            )
            self.modes[best_id] = mode
            self.address_modes.setdefault(int(address), []).append(best_id)
            best = 1.0
        assert best_id is not None
        self.modes[best_id].observe_signature(pooled_z, write_matrix)
        return best_id, create, float(best)

    def deploy_mode(self, *, address: int, pooled_z: torch.Tensor) -> tuple[int, float]:
        ids = self.modes_for_address(address)
        if not ids:
            raise KeyError(f"no functional modes for address {address}")
        scored = [
            (cosine(pooled_z, self.modes[mode_id].z_prototype), mode_id)
            for mode_id in ids
        ]
        score, mode_id = max(scored, key=lambda item: (item[0], -item[1]))
        return mode_id, float(score)

    def soft_top2_modes(
        self, *, address: int, pooled_z: torch.Tensor, temperature: float
    ) -> list[tuple[int, float]]:
        ids = self.modes_for_address(address)
        if not ids:
            raise KeyError(f"no functional modes for address {address}")
        scored = sorted(
            [
                (cosine(pooled_z, self.modes[mode_id].z_prototype), mode_id)
                for mode_id in ids
            ],
            key=lambda item: (-item[0], item[1]),
        )[:2]
        if len(scored) == 1:
            return [(scored[0][1], 1.0)]
        logits = torch.tensor([x[0] for x in scored], dtype=torch.float64) / max(
            float(temperature), 1e-6
        )
        weights = torch.softmax(logits, dim=0).tolist()
        return [(scored[i][1], float(weights[i])) for i in range(len(scored))]


def subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized squared principal-subspace overlap in [0, 1]."""
    qa = a.detach().to(device="cpu", dtype=torch.float64)
    qb = b.detach().to(device="cpu", dtype=torch.float64)
    if qa.numel() == 0 or qb.numel() == 0:
        return 0.0
    denom = min(qa.shape[1], qb.shape[1])
    if denom <= 0:
        return 0.0
    value = float(torch.linalg.norm(qa.T @ qb).square().item() / denom)
    return max(0.0, min(1.0, value))


def activation_overlap(a: FunctionalMode, b: FunctionalMode, *, energy: float = 0.99) -> float:
    return subspace_overlap(
        protected_basis(a.z_cov, energy=energy),
        protected_basis(b.z_cov, energy=energy),
    )


def write_overlap(a: FunctionalMode, b: FunctionalMode) -> float:
    return subspace_overlap(a.write_basis, b.write_basis)


def directed_interference(writer: FunctionalMode, protected: FunctionalMode) -> float:
    """Expected damage of writer's representative update on protected mode.

    D is the mean projected write matrix.  For historical z with second moment
    Sigma, E||D z||^2 is proportional to tr(D Sigma D^T).  Normalization by
    protected energy makes values comparable across dependency loads; load is
    reported separately.
    """
    sigma = protected.z_cov.detach().to(dtype=torch.float64)
    energy = float(torch.trace(sigma).item())
    if energy <= _EPS or writer.write_count <= 0:
        return 0.0
    d = writer.write_mean.detach().to(dtype=torch.float64)
    damage = float(torch.trace(d @ sigma @ d.T).item())
    dnorm = max(float(torch.linalg.norm(d).square().item()), _EPS)
    return max(0.0, damage / (energy * dnorm))


def symmetric_interference(a: FunctionalMode, b: FunctionalMode) -> float:
    return 0.5 * (directed_interference(a, b) + directed_interference(b, a))


def pairwise_weights(
    mode_ids: list[int],
    catalog: FunctionalModeCatalog,
    *,
    candidate: str,
) -> torch.Tensor:
    n = len(mode_ids)
    w = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = catalog.modes[mode_ids[i]], catalog.modes[mode_ids[j]]
            if candidate == "activation_community":
                value = activation_overlap(a, b)
            elif candidate == "write_community":
                value = write_overlap(a, b)
            elif candidate == "interference_cut":
                value = symmetric_interference(a, b)
            else:
                raise ValueError(f"unsupported geometry candidate: {candidate}")
            w[i, j] = w[j, i] = value
    return w


def greedy_balanced_maxcut(mode_ids: list[int], weights: torch.Tensor) -> tuple[list[int], list[int]]:
    """Deterministic small-graph max-cut heuristic with a balance tie-break."""
    if len(mode_ids) < 2:
        raise ValueError("at least two modes are required")
    n = len(mode_ids)
    best_pair = max(
        ((float(weights[i, j].item()), i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda x: (x[0], -mode_ids[x[1]], -mode_ids[x[2]]),
    )
    left = [best_pair[1]]
    right = [best_pair[2]]
    remaining = [i for i in range(n) if i not in {best_pair[1], best_pair[2]}]
    remaining.sort(
        key=lambda i: (
            -float(weights[i].sum().item()),
            mode_ids[i],
        )
    )
    for idx in remaining:
        within_left = sum(float(weights[idx, j].item()) for j in left)
        within_right = sum(float(weights[idx, j].item()) for j in right)
        # Put a node where it creates the smaller within-group interference.
        if within_left < within_right:
            left.append(idx)
        elif within_right < within_left:
            right.append(idx)
        elif len(left) <= len(right):
            left.append(idx)
        else:
            right.append(idx)
    return sorted(mode_ids[i] for i in left), sorted(mode_ids[i] for i in right)


def semantic_singleton_partition(
    mode_ids: list[int], catalog: FunctionalModeCatalog, *, trigger_mode: int
) -> tuple[list[int], list[int]]:
    address = catalog.modes[trigger_mode].address
    child = sorted(m for m in mode_ids if catalog.modes[m].address == address)
    parent = sorted(m for m in mode_ids if m not in set(child))
    if not parent or not child:
        # Degenerate address group: retain the exact 006 singleton behavior.
        child = [trigger_mode]
        parent = sorted(m for m in mode_ids if m != trigger_mode)
    if not parent:
        raise ValueError("semantic split cannot separate this Cell")
    return parent, child


def partition_modes(
    candidate: str,
    mode_ids: list[int],
    catalog: FunctionalModeCatalog,
    *,
    trigger_mode: int,
) -> tuple[list[int], list[int]]:
    ids = sorted(set(int(x) for x in mode_ids))
    if len(ids) < 2:
        raise ValueError("cannot partition fewer than two functional modes")
    if candidate == "semantic_singleton":
        return semantic_singleton_partition(ids, catalog, trigger_mode=trigger_mode)
    weights = pairwise_weights(ids, catalog, candidate=candidate)
    left, right = greedy_balanced_maxcut(ids, weights)
    # For deterministic clone-and-move semantics, the trigger side becomes child.
    if trigger_mode in left:
        return right, left
    return left, right


def cut_interference_fraction(
    mode_ids: list[int],
    left: list[int],
    right: list[int],
    catalog: FunctionalModeCatalog,
) -> float:
    ids = sorted(mode_ids)
    total = 0.0
    cross = 0.0
    left_set, right_set = set(left), set(right)
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            value = symmetric_interference(catalog.modes[a_id], catalog.modes[b_id])
            total += value
            if (a_id in left_set and b_id in right_set) or (
                a_id in right_set and b_id in left_set
            ):
                cross += value
    return 0.0 if total <= _EPS else float(cross / total)


def mode_metrics(mode: FunctionalMode, *, certificate_energy: float) -> dict[str, float | int]:
    spec = spectrum_metrics(mode.z_cov, certificate_energy=certificate_energy)
    return {
        "mode_id": mode.mode_id,
        "address": mode.address,
        "dependency_tokens": mode.dependency_tokens,
        "dependency_sequences": mode.dependency_sequences,
        "write_rank": int(mode.write_basis.shape[1]),
        **spec,
    }
