"""Core Validation 007 — Functional Boundary Discovery.

The experiment deliberately separates mechanism discovery from untouched
confirmation.  Discovery asks which bounded geometry best separates mutually
interfering write modes.  Confirmation freezes one boundary mechanism and
checks whether oracle functional routing plus a deployable z-only router can
support actual replay-free continual learning.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .real_representation_006_core import fit_functional_delta, spectrum_metrics
from .real_representation_006_experiment import (
    ProjectedSequence,
    build_transactions,
    prepare_seed,
    run_seed as run_core006_seed,
)
from .real_representation_006_io import FrozenSequence
from .real_representation_007_config import CoreValidation007Config
from .real_representation_007_core import (
    FunctionalModeCatalog,
    activation_overlap,
    cosine,
    cut_interference_fraction,
    mode_metrics,
    partition_modes,
    symmetric_interference,
    write_overlap,
)

_EPS = 1e-12


@dataclass(frozen=True)
class SequenceSignature:
    write_matrix: torch.Tensor
    pooled_z: torch.Tensor


@dataclass
class FunctionalCell:
    cell_id: int
    a: torch.Tensor
    parent_id: int | None
    birth_transaction: int


@dataclass
class FunctionalSystem:
    dim: int
    base_address_owner: dict[int, int]
    certificate_energy: float
    catalog: FunctionalModeCatalog
    cells: dict[int, FunctionalCell]
    mode_owner: dict[int, int] = field(default_factory=dict)
    next_cell_id: int = 0

    @classmethod
    def initialize(
        cls,
        *,
        dim: int,
        base_address_owner: dict[int, int],
        base_cells: int,
        certificate_energy: float,
        maximum_modes_per_address: int,
        maximum_write_rank: int,
        mode_creation_cosine_threshold: float,
    ) -> "FunctionalSystem":
        cells = {
            i: FunctionalCell(
                cell_id=i,
                a=torch.zeros(dim, dim, dtype=torch.float64),
                parent_id=None,
                birth_transaction=-1,
            )
            for i in range(base_cells)
        }
        return cls(
            dim=dim,
            base_address_owner=dict(base_address_owner),
            certificate_energy=certificate_energy,
            catalog=FunctionalModeCatalog(
                dim=dim,
                maximum_modes_per_address=maximum_modes_per_address,
                maximum_write_rank=maximum_write_rank,
                creation_cosine_threshold=mode_creation_cosine_threshold,
            ),
            cells=cells,
            next_cell_id=base_cells,
        )

    def clone(self) -> "FunctionalSystem":
        return copy.deepcopy(self)

    def ensure_mode_owner(self, mode_id: int) -> int:
        if mode_id in self.mode_owner:
            return self.mode_owner[mode_id]
        mode = self.catalog.modes[mode_id]
        other = [
            m
            for m in self.catalog.modes_for_address(mode.address)
            if m != mode_id and m in self.mode_owner
        ]
        if other:
            closest = max(
                other,
                key=lambda m: (
                    cosine(mode.write_mean, self.catalog.modes[m].write_mean),
                    -m,
                ),
            )
            owner = self.mode_owner[closest]
        else:
            owner = self.base_address_owner[mode.address]
        self.mode_owner[mode_id] = int(owner)
        return int(owner)

    def modes_for_cell(self, cell_id: int) -> list[int]:
        return sorted(m for m, owner in self.mode_owner.items() if owner == cell_id)

    def cell_covariance(self, cell_id: int) -> torch.Tensor:
        out = torch.zeros(self.dim, self.dim, dtype=torch.float64)
        for mode_id in self.modes_for_cell(cell_id):
            out += self.catalog.modes[mode_id].z_cov
        return out

    def metrics(self, cell_id: int) -> dict[str, float | int]:
        spec = spectrum_metrics(
            self.cell_covariance(cell_id), certificate_energy=self.certificate_energy
        )
        modes = self.modes_for_cell(cell_id)
        dep_tokens = sum(self.catalog.modes[m].dependency_tokens for m in modes)
        dep_sequences = sum(self.catalog.modes[m].dependency_sequences for m in modes)
        participation = float(spec["participation_rank"])
        return {
            "cell_id": cell_id,
            "parent_id": -1 if self.cells[cell_id].parent_id is None else self.cells[cell_id].parent_id,
            "mode_count": len(modes),
            "address_count": len({self.catalog.modes[m].address for m in modes}),
            "dependency_tokens": dep_tokens,
            "dependency_sequences": dep_sequences,
            "reuse_density": dep_tokens / max(participation, 1.0),
            **spec,
        }

    def split(
        self,
        *,
        parent: int,
        candidate: str,
        trigger_mode: int,
        transaction: int,
    ) -> dict[str, Any]:
        modes = self.modes_for_cell(parent)
        if len(modes) < 2:
            raise ValueError("cannot split a Cell with fewer than two modes")
        parent_group, child_group = partition_modes(
            candidate, modes, self.catalog, trigger_mode=trigger_mode
        )
        if not parent_group or not child_group:
            raise ValueError("functional partition must produce two non-empty groups")
        before = self.metrics(parent)
        child = self.next_cell_id
        self.next_cell_id += 1
        self.cells[child] = FunctionalCell(
            cell_id=child,
            a=self.cells[parent].a.clone(),
            parent_id=parent,
            birth_transaction=transaction,
        )
        for mode_id in child_group:
            self.mode_owner[mode_id] = child
        # Explicitly retain parent ownership for auditability.
        for mode_id in parent_group:
            self.mode_owner[mode_id] = parent
        return {
            "transaction": transaction,
            "candidate": candidate,
            "parent_id": parent,
            "child_id": child,
            "trigger_mode": trigger_mode,
            "parent_modes_before": len(modes),
            "parent_modes_after": len(parent_group),
            "child_modes_after": len(child_group),
            "parent_dependency_before": before["dependency_tokens"],
            "parent_dependency_after": self.metrics(parent)["dependency_tokens"],
            "child_dependency_after": self.metrics(child)["dependency_tokens"],
            "interference_cut_fraction": cut_interference_fraction(
                modes, parent_group, child_group, self.catalog
            ),
        }


@dataclass
class FunctionalHistory:
    sequence: ProjectedSequence
    mode_id: int
    reference_nll: float


@dataclass
class CandidateState:
    candidate: str
    system: FunctionalSystem
    history: list[FunctionalHistory] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    split_records: list[dict[str, Any]] = field(default_factory=list)
    rank_records: list[dict[str, Any]] = field(default_factory=list)
    routing_records: list[dict[str, Any]] = field(default_factory=list)


def _signature_batches(
    sequences: list[ProjectedSequence],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, SequenceSignature]:
    """Compute immutable foundation-path write signatures.

    These signatures use the frozen Pythia hidden state before any Cell output.
    They define the oracle functional identity and therefore do not drift with
    candidate Cell parameters.
    """
    out: dict[str, SequenceSignature] = {}
    weight = lm_head_weight.to(device=device, dtype=torch.float32)
    u_dev = u.to(device=device, dtype=torch.float32)
    for start in range(0, len(sequences), batch_size):
        rows = sequences[start : start + batch_size]
        hidden = torch.stack([s.hidden for s in rows]).to(device=device, dtype=torch.float32)
        labels = torch.stack([s.labels for s in rows]).to(device=device, dtype=torch.long)
        hidden = hidden.detach().requires_grad_(True)
        logits = F.linear(hidden, weight)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
        )
        grad_h = torch.autograd.grad(loss, hidden)[0]
        projected_grad = torch.einsum("bth,hd->btd", grad_h, u_dev)
        for i, seq in enumerate(rows):
            # G = u z^T, averaged over tokens.  Sign/step do not affect mode identity.
            write = torch.einsum(
                "to,ti->oi", projected_grad[i], seq.z.to(device=device, dtype=torch.float32)
            ) / max(seq.tokens, 1)
            out[seq.token_sha256] = SequenceSignature(
                write_matrix=write.detach().cpu().to(dtype=torch.float64),
                pooled_z=seq.pooled.detach().cpu().to(dtype=torch.float64),
            )
    return out


def _match_write_mode(
    catalog: FunctionalModeCatalog, *, address: int, write_matrix: torch.Tensor
) -> tuple[int, float]:
    ids = catalog.modes_for_address(address)
    if not ids:
        raise KeyError(f"no modes for address {address}")
    scored = [
        (cosine(write_matrix, catalog.modes[m].write_mean), m)
        for m in ids
    ]
    score, mode_id = max(scored, key=lambda x: (x[0], -x[1]))
    return mode_id, float(score)


def _build_discovery_catalog(
    sequences: list[ProjectedSequence],
    signatures: dict[str, SequenceSignature],
    cfg: CoreValidation007Config,
) -> tuple[FunctionalModeCatalog, dict[str, int], list[dict[str, Any]]]:
    catalog = FunctionalModeCatalog(
        dim=cfg.base.cell_dim,
        maximum_modes_per_address=cfg.maximum_modes_per_address,
        maximum_write_rank=cfg.maximum_write_rank,
        creation_cosine_threshold=cfg.mode_creation_cosine_threshold,
    )
    assignments: dict[str, int] = {}
    routing: list[dict[str, Any]] = []
    train = [s for s in sequences if s.partition == "train"]
    for seq in train:
        sig = signatures[seq.token_sha256]
        mode_id, created, similarity = catalog.locate_or_create(
            address=seq.address,
            pooled_z=sig.pooled_z,
            write_matrix=sig.write_matrix,
        )
        catalog.modes[mode_id].register_dependency(seq.z, sequences=1)
        assignments[seq.token_sha256] = mode_id
        routing.append(
            {
                "token_sha256": seq.token_sha256,
                "address": seq.address,
                "oracle_mode": mode_id,
                "created": created,
                "write_similarity": similarity,
            }
        )
    # Deployable routing is evaluated after the bounded catalog is formed.
    for row, seq in zip(routing, train):
        deploy, score = catalog.deploy_mode(address=seq.address, pooled_z=seq.pooled)
        top2 = catalog.soft_top2_modes(
            address=seq.address,
            pooled_z=seq.pooled,
            temperature=cfg.deploy_soft_top2_temperature,
        )
        row["deploy_mode"] = deploy
        row["deploy_score"] = score
        row["deploy_match"] = deploy == row["oracle_mode"]
        row["soft_top2_contains_oracle"] = row["oracle_mode"] in {m for m, _ in top2}
    return catalog, assignments, routing


def _discovery_candidate_rows(
    catalog: FunctionalModeCatalog,
    base_assignment: dict[int, int],
    cfg: CoreValidation007Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in cfg.boundary_candidates:
        cut_values: list[float] = []
        balances: list[float] = []
        partitions = 0
        for cell_id in range(cfg.base.base_cells):
            modes = sorted(
                mode_id
                for mode_id, mode in catalog.modes.items()
                if base_assignment[mode.address] == cell_id
            )
            if len(modes) < 2:
                continue
            triggers = modes if candidate == "semantic_singleton" else [modes[0]]
            for trigger in triggers:
                left, right = partition_modes(
                    candidate, modes, catalog, trigger_mode=trigger
                )
                cut_values.append(
                    cut_interference_fraction(modes, left, right, catalog)
                )
                balances.append(min(len(left), len(right)) / max(len(left), len(right)))
                partitions += 1
        cut_values.sort()
        balances.sort()
        median_cut = cut_values[len(cut_values) // 2] if cut_values else 0.0
        median_balance = balances[len(balances) // 2] if balances else 0.0
        rows.append(
            {
                "candidate": candidate,
                "median_interference_cut_fraction": float(median_cut),
                "median_balance": float(median_balance),
                "partitions_evaluated": partitions,
            }
        )
    return rows


def _pair_diagnostics(catalog: FunctionalModeCatalog) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = sorted(catalog.modes)
    for i, a_id in enumerate(ids):
        a = catalog.modes[a_id]
        for b_id in ids[i + 1 :]:
            b = catalog.modes[b_id]
            rows.append(
                {
                    "mode_a": a_id,
                    "mode_b": b_id,
                    "address_a": a.address,
                    "address_b": b.address,
                    "activation_overlap": activation_overlap(a, b),
                    "write_overlap": write_overlap(a, b),
                    "interference": symmetric_interference(a, b),
                }
            )
    return rows


def run_discovery_seed(
    sequences: list[FrozenSequence],
    cfg: CoreValidation007Config,
    *,
    seed: int,
    lm_head_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    u, centroids, base_assignment, projected = prepare_seed(sequences, cfg.base, seed=seed)
    signatures = _signature_batches(projected, u, lm_head_weight, device=device)
    catalog, assignments, routing = _build_discovery_catalog(projected, signatures, cfg)
    agreement = sum(bool(r["deploy_match"]) for r in routing) / max(len(routing), 1)
    top2 = sum(bool(r["soft_top2_contains_oracle"]) for r in routing) / max(len(routing), 1)
    candidate_rows = _discovery_candidate_rows(catalog, base_assignment, cfg)
    for row in candidate_rows:
        row["routing_agreement"] = float(agreement)
        row["soft_top2_coverage"] = float(top2)
        row["selection_score"] = (
            cfg.selection_conflict_weight * row["median_interference_cut_fraction"]
            + cfg.selection_routing_weight * agreement
            + cfg.selection_balance_weight * row["median_balance"]
        )
    return {
        "seed": seed,
        "router": {
            "centroids": centroids.tolist(),
            "base_assignment": {str(k): int(v) for k, v in base_assignment.items()},
        },
        "mode_count": len(catalog.modes),
        "mode_metrics": [
            mode_metrics(catalog.modes[m], certificate_energy=cfg.base.certificate_energy)
            for m in sorted(catalog.modes)
        ],
        "routing_records": routing,
        "routing_agreement": float(agreement),
        "soft_top2_coverage": float(top2),
        "candidate_rows": candidate_rows,
        "pair_diagnostics": _pair_diagnostics(catalog),
        "assignment_count": len(assignments),
    }


def summarize_discovery(runs: list[dict[str, Any]], cfg: CoreValidation007Config) -> dict[str, Any]:
    aggregate: dict[str, list[dict[str, Any]]] = {c: [] for c in cfg.boundary_candidates}
    for run in runs:
        for row in run["candidate_rows"]:
            aggregate[row["candidate"]].append(row)
    candidates = []
    for candidate in cfg.boundary_candidates:
        rows = aggregate[candidate]
        if not rows:
            continue
        mean = lambda key: sum(float(r[key]) for r in rows) / len(rows)
        candidates.append(
            {
                "candidate": candidate,
                "mean_interference_cut_fraction": mean("median_interference_cut_fraction"),
                "mean_routing_agreement": mean("routing_agreement"),
                "mean_soft_top2_coverage": mean("soft_top2_coverage"),
                "mean_balance": mean("median_balance"),
                "mean_selection_score": mean("selection_score"),
            }
        )
    winner = max(
        candidates,
        key=lambda r: (float(r["mean_selection_score"]), r["candidate"]),
    )
    return {
        "status": "FUNCTIONAL_BOUNDARY_DISCOVERY_COMPLETED",
        "scientific_decision": False,
        "candidate_summary": candidates,
        "provisional_winner": winner["candidate"],
        "winner_meets_routing_floor": (
            float(winner["mean_routing_agreement"])
            >= cfg.minimum_discovery_routing_agreement
        ),
    }


def _adjusted_batch(
    sequences: list[ProjectedSequence],
    mode_ids: list[int],
    system: FunctionalSystem,
    u: torch.Tensor,
    *,
    device: torch.device,
    ablate_cell: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.stack([s.hidden for s in sequences]).to(device=device, dtype=torch.float32)
    z = torch.stack([s.z for s in sequences]).to(device=device, dtype=torch.float32)
    labels = torch.stack([s.labels for s in sequences]).to(device=device, dtype=torch.long)
    matrices = []
    for mode_id in mode_ids:
        owner = system.mode_owner[mode_id]
        if ablate_cell is not None and owner == ablate_cell:
            matrices.append(torch.zeros(system.dim, system.dim, dtype=torch.float64))
        else:
            matrices.append(system.cells[owner].a)
    a = torch.stack(matrices).to(device=device, dtype=torch.float32)
    coeff = torch.einsum("bti,boi->bto", z, a)
    delta_h = torch.einsum(
        "btd,hd->bth", coeff, u.to(device=device, dtype=torch.float32)
    )
    return hidden + delta_h, labels, z


def _nll_targets(
    sequences: list[ProjectedSequence],
    mode_ids: list[int],
    system: FunctionalSystem,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    functional_step: float,
    device: torch.device,
    need_gradient: bool,
    ablate_cell: int | None = None,
) -> tuple[float, list[float], list[torch.Tensor]]:
    adjusted, labels, _ = _adjusted_batch(
        sequences, mode_ids, system, u, device=device, ablate_cell=ablate_cell
    )
    adjusted = adjusted.detach().requires_grad_(need_gradient)
    logits = F.linear(
        adjusted,
        lm_head_weight.to(device=device, dtype=adjusted.dtype),
    )
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    per = token_loss.mean(dim=1)
    mean = float(per.mean().detach().cpu().item())
    if not need_gradient:
        return mean, [float(x) for x in per.detach().cpu().tolist()], []
    grad_h = torch.autograd.grad(token_loss.sum(), adjusted)[0]
    desired = -functional_step * torch.einsum(
        "bth,hd->btd",
        grad_h.detach(),
        u.to(device=device, dtype=grad_h.dtype),
    )
    return (
        mean,
        [float(x) for x in per.detach().cpu().tolist()],
        [desired[i].detach().cpu().to(dtype=torch.float64) for i in range(len(sequences))],
    )


def _fit_mode_subset(
    state: CandidateState,
    sequences: list[ProjectedSequence],
    mode_ids: list[int],
    desired: list[torch.Tensor],
    cfg: CoreValidation007Config,
    *,
    apply: bool,
) -> dict[int, Any]:
    groups: dict[int, list[int]] = {}
    for i, mode_id in enumerate(mode_ids):
        owner = state.system.mode_owner[mode_id]
        groups.setdefault(owner, []).append(i)
    fits = {}
    for owner, indices in groups.items():
        z = torch.cat([sequences[i].z.to(torch.float64) for i in indices], dim=0)
        y = torch.cat([desired[i] for i in indices], dim=0)
        fit = fit_functional_delta(
            z,
            y,
            cov=state.system.cell_covariance(owner),
            certificate_energy=cfg.base.certificate_energy,
            ridge=cfg.base.ridge,
            safe=True,
            maximum_delta_norm=cfg.base.maximum_delta_norm,
        )
        if apply:
            state.system.cells[owner].a += fit.delta
        fits[owner] = fit
    return fits


def _checkpoint_regression(
    state: CandidateState,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    if not state.history:
        return 0.0
    now: list[float] = []
    refs: list[float] = []
    for start in range(0, len(state.history), 16):
        chunk = state.history[start : start + 16]
        _, per, _ = _nll_targets(
            [x.sequence for x in chunk],
            [x.mode_id for x in chunk],
            state.system,
            u,
            lm_head_weight,
            functional_step=0.0,
            device=device,
            need_gradient=False,
        )
        now.extend(per)
        refs.extend(x.reference_nll for x in chunk)
    rel = [max(0.0, (a - b) / max(b, 1e-8)) for a, b in zip(now, refs)]
    return float(sum(rel) / len(rel))


def _record_rank(state: CandidateState, transaction: int) -> None:
    for cell_id in sorted(state.system.cells):
        if state.system.modes_for_cell(cell_id):
            state.rank_records.append(
                {"transaction": transaction, **state.system.metrics(cell_id)}
            )


def _train_candidate_transaction(
    state: CandidateState,
    current: list[ProjectedSequence],
    signatures: dict[str, SequenceSignature],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    cfg: CoreValidation007Config,
    *,
    transaction: int,
    device: torch.device,
) -> tuple[float, float, int, int]:
    mode_ids: list[int] = []
    child_reuse = 0
    for seq in current:
        sig = signatures[seq.token_sha256]
        mode_id, created, similarity = state.system.catalog.locate_or_create(
            address=seq.address,
            pooled_z=sig.pooled_z,
            write_matrix=sig.write_matrix,
        )
        owner = state.system.ensure_mode_owner(mode_id)
        deploy, deploy_score = state.system.catalog.deploy_mode(
            address=seq.address, pooled_z=seq.pooled
        )
        state.routing_records.append(
            {
                "transaction": transaction,
                "address": seq.address,
                "oracle_mode": mode_id,
                "deploy_mode": deploy,
                "agreement": deploy == mode_id,
                "created": created,
                "write_similarity": similarity,
                "deploy_score": deploy_score,
            }
        )
        mode_ids.append(mode_id)
        if state.system.cells[owner].parent_id is not None:
            child_reuse += 1

    pre, _, desired = _nll_targets(
        current,
        mode_ids,
        state.system,
        u,
        lm_head_weight,
        functional_step=cfg.base.functional_step,
        device=device,
        need_gradient=True,
    )

    # Measure per-mode conflict and split the worst conflicts first.
    conflict_rows = []
    for i, mode_id in enumerate(mode_ids):
        owner = state.system.mode_owner[mode_id]
        fit = fit_functional_delta(
            current[i].z.to(torch.float64),
            desired[i],
            cov=state.system.cell_covariance(owner),
            certificate_energy=cfg.base.certificate_energy,
            ridge=cfg.base.ridge,
            safe=True,
            maximum_delta_norm=None,
        )
        conflict_rows.append((fit.conflict_fraction, i, mode_id, owner))
    conflict_rows.sort(key=lambda x: (-x[0], x[2]))
    splits = 0
    for conflict, i, mode_id, parent in conflict_rows:
        if conflict <= cfg.split_conflict_threshold:
            continue
        parent = state.system.mode_owner[mode_id]
        if len(state.system.modes_for_cell(parent)) < 2:
            continue
        if splits >= cfg.maximum_splits_per_transaction:
            break
        record = state.system.split(
            parent=parent,
            candidate=state.candidate,
            trigger_mode=mode_id,
            transaction=transaction,
        )
        after_owner = state.system.mode_owner[mode_id]
        after = fit_functional_delta(
            current[i].z.to(torch.float64),
            desired[i],
            cov=state.system.cell_covariance(after_owner),
            certificate_energy=cfg.base.certificate_energy,
            ridge=cfg.base.ridge,
            safe=True,
            maximum_delta_norm=None,
        )
        record["conflict_before"] = float(conflict)
        record["conflict_after"] = float(after.conflict_fraction)
        record["conflict_reduction"] = float(conflict - after.conflict_fraction)
        state.split_records.append(record)
        splits += 1

    _fit_mode_subset(state, current, mode_ids, desired, cfg, apply=True)
    post, per, _ = _nll_targets(
        current,
        mode_ids,
        state.system,
        u,
        lm_head_weight,
        functional_step=0.0,
        device=device,
        need_gradient=False,
    )
    for seq, mode_id, ref in zip(current, mode_ids, per):
        state.system.catalog.modes[mode_id].register_dependency(seq.z, sequences=1)
        state.history.append(
            FunctionalHistory(sequence=seq, mode_id=mode_id, reference_nll=float(ref))
        )
    return pre, post, splits, child_reuse


def _eval_mode_ids(
    sequences: list[ProjectedSequence],
    signatures: dict[str, SequenceSignature],
    system: FunctionalSystem,
    *,
    deploy: bool,
) -> tuple[list[int], float]:
    modes = []
    agreements = []
    for seq in sequences:
        sig = signatures[seq.token_sha256]
        oracle, _ = _match_write_mode(
            system.catalog, address=seq.address, write_matrix=sig.write_matrix
        )
        deploy_id, _ = system.catalog.deploy_mode(address=seq.address, pooled_z=seq.pooled)
        agreements.append(oracle == deploy_id)
        modes.append(deploy_id if deploy else oracle)
    agreement = sum(agreements) / max(len(agreements), 1)
    return modes, float(agreement)


def _soft_top2_eval_nll(
    sequences: list[ProjectedSequence],
    system: FunctionalSystem,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    cfg: CoreValidation007Config,
    *,
    device: torch.device,
) -> float:
    if not sequences:
        return 0.0
    hidden = torch.stack([s.hidden for s in sequences]).to(device=device, dtype=torch.float32)
    z = torch.stack([s.z for s in sequences]).to(device=device, dtype=torch.float32)
    labels = torch.stack([s.labels for s in sequences]).to(device=device, dtype=torch.long)
    coeff_rows = []
    for i, seq in enumerate(sequences):
        pairs = system.catalog.soft_top2_modes(
            address=seq.address,
            pooled_z=seq.pooled,
            temperature=cfg.deploy_soft_top2_temperature,
        )
        coeff = torch.zeros_like(z[i])
        for mode_id, weight in pairs:
            owner = system.mode_owner[mode_id]
            a = system.cells[owner].a.to(device=device, dtype=torch.float32)
            coeff += float(weight) * torch.einsum("ti,oi->to", z[i], a)
        coeff_rows.append(coeff)
    coeff = torch.stack(coeff_rows)
    adjusted = hidden + torch.einsum(
        "btd,hd->bth", coeff, u.to(device=device, dtype=torch.float32)
    )
    logits = F.linear(adjusted, lm_head_weight.to(device=device, dtype=torch.float32))
    return float(
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        .detach()
        .cpu()
        .item()
    )


def _causal_rows(
    state: CandidateState,
    eval_sequences: list[ProjectedSequence],
    signatures: dict[str, SequenceSignature],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    oracle_modes, _ = _eval_mode_ids(
        eval_sequences, signatures, state.system, deploy=False
    )
    rows = []
    for cell_id in sorted(state.system.cells):
        indices = [
            i
            for i, mode_id in enumerate(oracle_modes)
            if state.system.mode_owner[mode_id] == cell_id
        ]
        metrics = state.system.metrics(cell_id)
        if not indices:
            rows.append({**metrics, "eval_sequences": 0, "causal_delta_nll": 0.0})
            continue
        subset = [eval_sequences[i] for i in indices]
        modes = [oracle_modes[i] for i in indices]
        full, _, _ = _nll_targets(
            subset,
            modes,
            state.system,
            u,
            lm_head_weight,
            functional_step=0.0,
            device=device,
            need_gradient=False,
        )
        ablated, _, _ = _nll_targets(
            subset,
            modes,
            state.system,
            u,
            lm_head_weight,
            functional_step=0.0,
            device=device,
            need_gradient=False,
            ablate_cell=cell_id,
        )
        rows.append(
            {
                **metrics,
                "eval_sequences": len(subset),
                "causal_delta_nll": float(ablated - full),
            }
        )
    return rows


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    x = sorted(float(v) for v in values)
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def run_confirmation_seed(
    sequences: list[FrozenSequence],
    cfg: CoreValidation007Config,
    *,
    seed: int,
    winner: str,
    lm_head_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    if winner not in cfg.boundary_candidates:
        raise ValueError(f"winner {winner!r} is not a frozen boundary candidate")
    # Exact 006 run supplies untouched unsafe/replay/no-growth/semantic baselines.
    baseline = run_core006_seed(
        sequences,
        cfg.base,
        seed=seed,
        lm_head_weight=lm_head_weight,
        device=device,
    )
    u, centroids, base_assignment, projected = prepare_seed(sequences, cfg.base, seed=seed)
    signatures = _signature_batches(projected, u, lm_head_weight, device=device)
    transactions = build_transactions(projected, cfg.base)
    eval_sequences = [s for s in projected if s.partition == "eval"]
    system = FunctionalSystem.initialize(
        dim=cfg.base.cell_dim,
        base_address_owner=base_assignment,
        base_cells=cfg.base.base_cells,
        certificate_energy=cfg.base.certificate_energy,
        maximum_modes_per_address=cfg.maximum_modes_per_address,
        maximum_write_rank=cfg.maximum_write_rank,
        mode_creation_cosine_threshold=cfg.mode_creation_cosine_threshold,
    )
    state = CandidateState(candidate=winner, system=system)
    checkpoint_rows = []
    for tx, current in enumerate(transactions):
        pre, post, splits, child_reuse = _train_candidate_transaction(
            state,
            current,
            signatures,
            u,
            lm_head_weight,
            cfg,
            transaction=tx,
            device=device,
        )
        checkpoint = 0.0
        if (
            (tx + 1) % cfg.base.retention_checkpoint_every_transactions == 0
            or tx + 1 == len(transactions)
        ):
            checkpoint = _checkpoint_regression(
                state, u, lm_head_weight, device=device
            )
            checkpoint_rows.append(
                {"transaction": tx, "positive_registered_regression": checkpoint}
            )
        state.records.append(
            {
                "transaction": tx,
                "source": current[0].source,
                "pre_nll": pre,
                "post_nll": post,
                "relative_new_gain": (pre - post) / max(pre, 1e-8),
                "splits": splits,
                "child_reuse": child_reuse,
                "checkpoint_positive_regression": checkpoint,
            }
        )
        _record_rank(state, tx)

    oracle_modes, eval_agreement = _eval_mode_ids(
        eval_sequences, signatures, state.system, deploy=False
    )
    deploy_modes, _ = _eval_mode_ids(
        eval_sequences, signatures, state.system, deploy=True
    )
    oracle_nll, _, _ = _nll_targets(
        eval_sequences,
        oracle_modes,
        state.system,
        u,
        lm_head_weight,
        functional_step=0.0,
        device=device,
        need_gradient=False,
    )
    deploy_nll, _, _ = _nll_targets(
        eval_sequences,
        deploy_modes,
        state.system,
        u,
        lm_head_weight,
        functional_step=0.0,
        device=device,
        need_gradient=False,
    )
    soft_nll = _soft_top2_eval_nll(
        eval_sequences,
        state.system,
        u,
        lm_head_weight,
        cfg,
        device=device,
    )
    causal = _causal_rows(
        state,
        eval_sequences,
        signatures,
        u,
        lm_head_weight,
        device=device,
    )

    base_summary = baseline["gate_summary"]["variant_summaries"]
    unsafe = base_summary["unsafe"]
    replay = base_summary["replay"]
    no_growth = base_summary["certificate_no_growth"]
    semantic = base_summary["certificate_mitosis"]
    gain = sum(max(0.0, float(r["relative_new_gain"])) for r in state.records)
    regression = float(state.records[-1]["checkpoint_positive_regression"]) if state.records else 0.0
    gain_ratio = gain / max(float(replay["cumulative_new_gain"]), _EPS)
    regression_ratio = regression / max(float(unsafe["final_positive_registered_regression"]), _EPS)
    reductions = [float(r["conflict_reduction"]) for r in state.split_records]
    median_reduction = _median(reductions)
    spawned = len(state.system.cells) - cfg.base.base_cells
    spawned_fraction = spawned / cfg.base.addresses
    child_reuse_transactions = sum(int(r["child_reuse"] > 0) for r in state.records)
    train_agreement = sum(bool(r["agreement"]) for r in state.routing_records) / max(
        len(state.routing_records), 1
    )
    routing_agreement = min(float(train_agreement), float(eval_agreement))
    deploy_gap = max(0.0, (deploy_nll - oracle_nll) / max(oracle_nll, _EPS))
    causal_nonzero = sum(abs(float(r["causal_delta_nll"])) > 1e-8 for r in causal)
    baseline_split = float(baseline["gate_summary"]["median_split_conflict_reduction"])

    gates = {
        "no_replay_candidate": True,
        "registered_retention": regression_ratio <= cfg.maximum_confirmation_regression_ratio_vs_unsafe,
        "plasticity_vs_replay": gain_ratio >= cfg.minimum_confirmation_gain_ratio_vs_replay,
        "improves_no_growth": gain > float(no_growth["cumulative_new_gain"]),
        "split_reduces_conflict": median_reduction >= cfg.minimum_confirmation_split_conflict_reduction,
        "beats_core006_split_geometry": median_reduction > baseline_split,
        "bounded_growth": spawned_fraction <= cfg.maximum_confirmation_spawned_fraction_of_addresses,
        "child_reuse": child_reuse_transactions >= cfg.minimum_confirmation_child_reuse_transactions,
        "routing_identifiable": routing_agreement >= cfg.minimum_confirmation_routing_agreement,
        "deploy_nll_close_to_oracle": deploy_gap <= cfg.maximum_confirmation_deploy_nll_gap,
        "causal_signal_present": causal_nonzero > 0,
    }
    return {
        "seed": seed,
        "winner": winner,
        "pass": bool(all(gates.values())),
        "gates": gates,
        "candidate": {
            "cumulative_new_gain": gain,
            "final_positive_registered_regression": regression,
            "gain_ratio_vs_replay": float(gain_ratio),
            "regression_ratio_vs_unsafe": float(regression_ratio),
            "median_split_conflict_reduction": float(median_reduction),
            "core006_median_split_conflict_reduction": baseline_split,
            "spawned_cells": spawned,
            "spawned_fraction_of_addresses": float(spawned_fraction),
            "child_reuse_transactions": child_reuse_transactions,
            "train_routing_agreement": float(train_agreement),
            "eval_routing_agreement": float(eval_agreement),
            "routing_agreement": float(routing_agreement),
            "oracle_eval_nll": float(oracle_nll),
            "deploy_eval_nll": float(deploy_nll),
            "soft_top2_eval_nll": float(soft_nll),
            "deploy_relative_nll_gap": float(deploy_gap),
            "causal_nonzero_cells": causal_nonzero,
        },
        "baseline_006": {
            "unsafe": unsafe,
            "replay": replay,
            "certificate_no_growth": no_growth,
            "semantic_singleton": semantic,
            "gate_summary": baseline["gate_summary"],
        },
        "records": state.records,
        "split_records": state.split_records,
        "rank_records": state.rank_records,
        "routing_records": state.routing_records,
        "checkpoint_records": checkpoint_rows,
        "causal_records": causal,
        "mode_metrics": [
            mode_metrics(state.system.catalog.modes[m], certificate_energy=cfg.base.certificate_energy)
            for m in sorted(state.system.catalog.modes)
        ],
        "router": {
            "centroids": centroids.tolist(),
            "base_assignment": {str(k): int(v) for k, v in base_assignment.items()},
        },
    }


def summarize_confirmation(
    runs: list[dict[str, Any]], *, winner: str, positive_status: str, negative_status: str
) -> dict[str, Any]:
    passed = bool(runs) and all(bool(r["pass"]) for r in runs)
    return {
        "status": positive_status if passed else negative_status,
        "scientific_decision": True,
        "pass": passed,
        "winner": winner,
        "passed_seeds": sum(bool(r["pass"]) for r in runs),
        "total_seeds": len(runs),
        "hypotheses": {
            "functional_geometry_provides_better_mitosis_boundary_than_semantic_address": all(
                r["gates"]["split_reduces_conflict"]
                and r["gates"]["beats_core006_split_geometry"]
                for r in runs
            ),
            "functional_boundary_restores_plasticity_with_bounded_growth": all(
                r["gates"]["plasticity_vs_replay"]
                and r["gates"]["bounded_growth"]
                and r["gates"]["child_reuse"]
                for r in runs
            ),
            "functional_identity_is_predictable_from_inference_visible_z": all(
                r["gates"]["routing_identifiable"]
                and r["gates"]["deploy_nll_close_to_oracle"]
                for r in runs
            ),
            "replay_free_certificate_retention_survives_functional_mitosis": all(
                r["gates"]["no_replay_candidate"] and r["gates"]["registered_retention"]
                for r in runs
            ),
        },
    }
