"""Real-representation continual-learning experiment for Core Validation 006."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .real_representation_006_config import CoreValidation006Config
from .real_representation_006_core import (
    AddressBatch,
    CellSystem,
    address_conflicts,
    apply_transaction_fit,
    assign_addresses,
    balanced_base_assignment,
    kmeans,
    orthonormal_columns,
    register_batches,
)
from .real_representation_006_io import FrozenSequence

VARIANTS = ("unsafe", "certificate_no_growth", "certificate_mitosis", "replay")


@dataclass(frozen=True)
class ProjectedSequence:
    partition: str
    source: str
    hidden: torch.Tensor
    labels: torch.Tensor
    z: torch.Tensor
    pooled: torch.Tensor
    address: int
    document_sha256: str
    token_sha256: str

    @property
    def tokens(self) -> int:
        return int(self.labels.numel())


@dataclass
class HistoryEntry:
    sequence: ProjectedSequence
    reference_nll: float


@dataclass
class VariantState:
    name: str
    system: CellSystem
    replay: list[ProjectedSequence]
    history: list[HistoryEntry]
    records: list[dict[str, Any]]
    rank_records: list[dict[str, Any]]
    split_records: list[dict[str, Any]]
    learner_old_sample_accesses: int = 0
    learner_old_label_accesses: int = 0


def prepare_seed(
    sequences: list[FrozenSequence], cfg: CoreValidation006Config, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int], list[ProjectedSequence]]:
    if not sequences:
        raise ValueError("no frozen sequences")
    hidden_dim = int(sequences[0].hidden.shape[-1])
    u = orthonormal_columns(hidden_dim, cfg.cell_dim, seed=seed + 11)
    interim: list[tuple[FrozenSequence, torch.Tensor, torch.Tensor]] = []
    router_points = []
    for seq in sequences:
        z = seq.hidden.to(dtype=torch.float32) @ u
        pooled = z.mean(dim=0)
        interim.append((seq, z, pooled))
        if seq.partition == "router":
            router_points.append(pooled)
    router_matrix = torch.stack(router_points)
    centroids = kmeans(
        router_matrix, cfg.addresses, seed=seed + 23, iterations=cfg.kmeans_iterations
    )
    router_addresses = assign_addresses(router_matrix, centroids)
    base_assignment = balanced_base_assignment(
        router_addresses, num_addresses=cfg.addresses, base_cells=cfg.base_cells
    )
    projected: list[ProjectedSequence] = []
    for seq, z, pooled in interim:
        address = int(assign_addresses(pooled[None, :], centroids)[0].item())
        projected.append(
            ProjectedSequence(
                partition=seq.partition,
                source=seq.source,
                hidden=seq.hidden,
                labels=seq.labels,
                z=z,
                pooled=pooled,
                address=address,
                document_sha256=seq.document_sha256,
                token_sha256=seq.token_sha256,
            )
        )
    return u, centroids, base_assignment, projected


def build_transactions(
    sequences: list[ProjectedSequence], cfg: CoreValidation006Config
) -> list[list[ProjectedSequence]]:
    by_source = {
        source: [s for s in sequences if s.partition == "train" and s.source == source]
        for source in cfg.sources
    }
    transactions: list[list[ProjectedSequence]] = []
    for source in cfg.sources:
        rows = by_source[source]
        if len(rows) != cfg.train_sequences_per_source:
            raise ValueError(f"unexpected train count for {source}: {len(rows)}")
        for start in range(0, len(rows), cfg.sequences_per_transaction):
            transactions.append(rows[start : start + cfg.sequences_per_transaction])
    return transactions


def _stack_batch(
    sequences: list[ProjectedSequence],
    system: CellSystem,
    u: torch.Tensor,
    *,
    device: torch.device,
    ablate_cell: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.stack([s.hidden for s in sequences]).to(device=device, dtype=torch.float32)
    z = torch.stack([s.z for s in sequences]).to(device=device, dtype=torch.float32)
    labels = torch.stack([s.labels for s in sequences]).to(device=device, dtype=torch.long)
    matrices = []
    for seq in sequences:
        owner = system.address_owner[seq.address]
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


def nll_and_gradient_targets(
    sequences: list[ProjectedSequence],
    system: CellSystem,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    functional_step: float,
    device: torch.device,
    need_gradient: bool,
    ablate_cell: int | None = None,
) -> tuple[float, list[float], dict[int, AddressBatch]]:
    if not sequences:
        return 0.0, [], {}
    adjusted, labels, z = _stack_batch(
        sequences, system, u, device=device, ablate_cell=ablate_cell
    )
    adjusted = adjusted.detach().requires_grad_(need_gradient)
    weight = lm_head_weight.to(device=device, dtype=adjusted.dtype)
    logits = F.linear(adjusted, weight)
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    per_seq = flat_loss.mean(dim=1)
    mean_nll = float(per_seq.mean().detach().cpu().item())
    if not need_gradient:
        return mean_nll, [float(x) for x in per_seq.detach().cpu().tolist()], {}

    grad_h = torch.autograd.grad(flat_loss.sum(), adjusted)[0]
    desired = -functional_step * torch.einsum(
        "bth,hd->btd", grad_h.detach(), u.to(device=device, dtype=grad_h.dtype)
    )
    grouped: dict[int, list[int]] = {}
    for idx, seq in enumerate(sequences):
        grouped.setdefault(seq.address, []).append(idx)
    batches: dict[int, AddressBatch] = {}
    z_cpu = z.detach().cpu().to(dtype=torch.float64)
    desired_cpu = desired.detach().cpu().to(dtype=torch.float64)
    for address, indices in grouped.items():
        batches[address] = AddressBatch(
            address=address,
            z=torch.cat([z_cpu[i] for i in indices], dim=0),
            desired=torch.cat([desired_cpu[i] for i in indices], dim=0),
            sequences=len(indices),
        )
    return mean_nll, [float(x) for x in per_seq.detach().cpu().tolist()], batches


def _record_rank_state(state: VariantState, transaction: int) -> None:
    for cell_id in sorted(state.system.cells):
        if state.system.addresses_for_cell(cell_id):
            state.rank_records.append(
                {"transaction": transaction, "variant": state.name, **state.system.metrics(cell_id)}
            )


def _update_replay(
    replay: list[ProjectedSequence], current: list[ProjectedSequence], *, maximum: int
) -> None:
    replay.extend(current)
    if len(replay) > maximum:
        del replay[: len(replay) - maximum]


def _sample_replay(
    replay: list[ProjectedSequence], *, count: int, rng: random.Random
) -> list[ProjectedSequence]:
    if not replay or count <= 0:
        return []
    if len(replay) <= count:
        return list(replay)
    return rng.sample(replay, count)


def _checkpoint_regression(
    state: VariantState,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_sequences: int = 16,
) -> dict[str, float]:
    if not state.history:
        return {"mean_positive_relative_regression": 0.0, "mean_relative_regression": 0.0}
    now: list[float] = []
    refs: list[float] = []
    for start in range(0, len(state.history), batch_sequences):
        chunk = state.history[start : start + batch_sequences]
        _, per, _ = nll_and_gradient_targets(
            [x.sequence for x in chunk],
            state.system,
            u,
            lm_head_weight,
            functional_step=0.0,
            device=device,
            need_gradient=False,
        )
        now.extend(per)
        refs.extend(x.reference_nll for x in chunk)
    rel = [(a - b) / max(b, 1e-8) for a, b in zip(now, refs)]
    positive = [max(0.0, x) for x in rel]
    return {
        "mean_positive_relative_regression": float(sum(positive) / len(positive)),
        "mean_relative_regression": float(sum(rel) / len(rel)),
    }


def _fit_current(
    state: VariantState,
    current: list[ProjectedSequence],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    cfg: CoreValidation006Config,
    *,
    transaction: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[float, float, int, int, int]:
    learner_batch = list(current)
    if state.name == "replay":
        replay = _sample_replay(
            state.replay, count=cfg.replay_sequences_per_transaction, rng=rng
        )
        learner_batch += replay
        state.learner_old_sample_accesses += len(replay)
        state.learner_old_label_accesses += sum(s.tokens for s in replay)

    _, _, address_batches = nll_and_gradient_targets(
        learner_batch,
        state.system,
        u,
        lm_head_weight,
        functional_step=cfg.functional_step,
        device=device,
        need_gradient=True,
    )
    current_pre, _, _ = nll_and_gradient_targets(
        current,
        state.system,
        u,
        lm_head_weight,
        functional_step=0.0,
        device=device,
        need_gradient=False,
    )

    split_count = 0
    blocked = 0
    child_reuse = sum(
        1
        for seq in current
        if state.system.cells[state.system.address_owner[seq.address]].parent_id is not None
    )
    current_addresses = {s.address for s in current}

    if state.name == "certificate_mitosis":
        current_batches = [b for a, b in address_batches.items() if a in current_addresses]
        conflicts = address_conflicts(
            state.system, current_batches, ridge=cfg.ridge, maximum_delta_norm=None
        )
        ordered = sorted(
            conflicts.items(), key=lambda item: (-item[1].conflict_fraction, item[0])
        )
        for address, fit in ordered:
            if fit.conflict_fraction <= cfg.split_conflict_threshold:
                continue
            parent = state.system.address_owner[address]
            if len(state.system.addresses_for_cell(parent)) <= 1:
                blocked += 1
                continue
            if split_count >= cfg.maximum_splits_per_transaction:
                blocked += 1
                continue
            split = state.system.split_address(address, transaction=transaction)
            after = address_conflicts(
                state.system,
                [address_batches[address]],
                ridge=cfg.ridge,
                maximum_delta_norm=None,
            )[address]
            split["conflict_before"] = fit.conflict_fraction
            split["conflict_after"] = after.conflict_fraction
            split["conflict_reduction"] = fit.conflict_fraction - after.conflict_fraction
            state.split_records.append(split)
            split_count += 1

    safe = state.name in {"certificate_no_growth", "certificate_mitosis"}
    apply_transaction_fit(
        state.system,
        list(address_batches.values()),
        safe=safe,
        ridge=cfg.ridge,
        maximum_delta_norm=cfg.maximum_delta_norm,
    )

    post_nll, current_post_per, _ = nll_and_gradient_targets(
        current,
        state.system,
        u,
        lm_head_weight,
        functional_step=0.0,
        device=device,
        need_gradient=False,
    )
    current_group: dict[int, list[ProjectedSequence]] = {}
    for seq in current:
        current_group.setdefault(seq.address, []).append(seq)
    to_register = []
    for address, rows in current_group.items():
        to_register.append(
            AddressBatch(
                address=address,
                z=torch.cat([x.z.to(torch.float64) for x in rows], dim=0),
                desired=torch.zeros(
                    sum(x.tokens for x in rows), cfg.cell_dim, dtype=torch.float64
                ),
                sequences=len(rows),
            )
        )
    register_batches(state.system, to_register)

    for seq, ref in zip(current, current_post_per):
        state.history.append(HistoryEntry(sequence=seq, reference_nll=float(ref)))
    if state.name == "replay":
        _update_replay(state.replay, current, maximum=cfg.replay_buffer_sequences)
    return current_pre, post_nll, split_count, child_reuse, blocked


def final_causal_metrics(
    state: VariantState,
    eval_sequences: list[ProjectedSequence],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for cell_id in sorted(state.system.cells):
        owned = set(state.system.addresses_for_cell(cell_id))
        if not owned:
            continue
        subset = [s for s in eval_sequences if s.address in owned]
        metrics = state.system.metrics(cell_id)
        if not subset:
            rows.append({**metrics, "eval_sequences": 0, "causal_delta_nll": 0.0})
            continue
        full, _, _ = nll_and_gradient_targets(
            subset,
            state.system,
            u,
            lm_head_weight,
            functional_step=0.0,
            device=device,
            need_gradient=False,
        )
        ablated, _, _ = nll_and_gradient_targets(
            subset,
            state.system,
            u,
            lm_head_weight,
            functional_step=0.0,
            device=device,
            need_gradient=False,
            ablate_cell=cell_id,
        )
        rows.append(
            {**metrics, "eval_sequences": len(subset), "causal_delta_nll": float(ablated - full)}
        )
    return rows


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(x) for x in values)
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def summarize_seed(
    states: dict[str, VariantState],
    cfg: CoreValidation006Config,
    *,
    causal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for name, state in states.items():
        rec = state.records
        summaries[name] = {
            "transactions": len(rec),
            "cumulative_new_gain": sum(max(0.0, float(r["relative_new_gain"])) for r in rec),
            "final_positive_registered_regression": (
                float(rec[-1]["checkpoint_positive_regression"]) if rec else 0.0
            ),
            "spawned_cells": len(state.system.cells) - cfg.base_cells,
            "child_reuse_transactions": sum(int(r["child_reuse"] > 0) for r in rec),
            "blocked_saturation_events": sum(int(r["blocked_saturation"]) for r in rec),
            "learner_old_sample_accesses": state.learner_old_sample_accesses,
            "learner_old_label_accesses": state.learner_old_label_accesses,
        }

    growth = summaries["certificate_mitosis"]
    unsafe = summaries["unsafe"]
    replay = summaries["replay"]
    no_growth = summaries["certificate_no_growth"]
    regression_ratio = growth["final_positive_registered_regression"] / max(
        unsafe["final_positive_registered_regression"], 1e-8
    )
    gain_ratio = growth["cumulative_new_gain"] / max(replay["cumulative_new_gain"], 1e-8)

    growth_rank = states["certificate_mitosis"].rank_records
    midpoint = max(1, cfg.transactions // 2)
    first_point = max(1, cfg.transactions // 4)
    mid_rows = [r for r in growth_rank if r["transaction"] == midpoint - 1]
    first_rows = [r for r in growth_rank if r["transaction"] == first_point - 1]
    mid_energy_fraction = _median(
        [float(r["energy_rank_99"]) / cfg.cell_dim for r in mid_rows]
    )
    first_reuse = _median([float(r["reuse_density"]) for r in first_rows])
    mid_reuse = _median([float(r["reuse_density"]) for r in mid_rows])
    reuse_ratio = mid_reuse / max(first_reuse, 1e-8)

    split_reductions = [
        float(r["conflict_reduction"])
        for r in states["certificate_mitosis"].split_records
    ]
    median_split_reduction = _median(split_reductions)
    spawned_fraction = growth["spawned_cells"] / cfg.addresses
    causal_growth = [r for r in causal_rows if r["variant"] == "certificate_mitosis"]
    causal_nonzero = sum(abs(float(r["causal_delta_nll"])) > 1e-8 for r in causal_growth)

    gates = {
        "no_replay_candidate": (
            growth["learner_old_sample_accesses"] == 0
            and growth["learner_old_label_accesses"] == 0
        ),
        "real_representation_not_immediately_full": (
            mid_energy_fraction <= cfg.maximum_midstream_energy_rank_fraction
        ),
        "functional_reuse_grows": reuse_ratio >= cfg.minimum_midstream_reuse_ratio,
        "registered_retention": (
            regression_ratio <= cfg.maximum_registered_regression_ratio_vs_unsafe
        ),
        "plasticity_vs_replay": gain_ratio >= cfg.minimum_gain_ratio_vs_replay,
        "mitosis_improves_plasticity": (
            growth["cumulative_new_gain"] > no_growth["cumulative_new_gain"]
        ),
        "split_reduces_conflict": (
            bool(split_reductions)
            and median_split_reduction >= cfg.minimum_split_conflict_reduction
        ),
        "bounded_growth": spawned_fraction <= cfg.maximum_spawned_fraction_of_addresses,
        "child_reuse": growth["child_reuse_transactions"] >= cfg.minimum_child_reuse_transactions,
        "causal_signal_present": causal_nonzero > 0,
    }
    return {
        "pass": bool(all(gates.values())),
        "gates": gates,
        "variant_summaries": summaries,
        "registered_regression_ratio_vs_unsafe": float(regression_ratio),
        "gain_ratio_vs_replay": float(gain_ratio),
        "midstream_energy_rank_fraction": float(mid_energy_fraction),
        "midstream_reuse_ratio": float(reuse_ratio),
        "median_split_conflict_reduction": float(median_split_reduction),
        "spawned_fraction_of_addresses": float(spawned_fraction),
        "causal_nonzero_cells": int(causal_nonzero),
    }


def run_seed(
    sequences: list[FrozenSequence],
    cfg: CoreValidation006Config,
    *,
    seed: int,
    lm_head_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    u, centroids, base_assignment, projected = prepare_seed(sequences, cfg, seed=seed)
    transactions = build_transactions(projected, cfg)
    eval_sequences = [s for s in projected if s.partition == "eval"]
    initial = CellSystem.initialize(
        dim=cfg.cell_dim,
        num_addresses=cfg.addresses,
        base_cells=cfg.base_cells,
        address_owner=base_assignment,
        certificate_energy=cfg.certificate_energy,
    )
    states = {
        name: VariantState(
            name=name,
            system=initial.clone(),
            replay=[],
            history=[],
            records=[],
            rank_records=[],
            split_records=[],
        )
        for name in VARIANTS
    }
    rngs = {name: random.Random(seed + 1000 + i) for i, name in enumerate(VARIANTS)}
    checkpoint_rows: list[dict[str, Any]] = []

    for tx_index, current in enumerate(transactions):
        for name in VARIANTS:
            state = states[name]
            pre, post, splits, child_reuse, blocked = _fit_current(
                state,
                current,
                u,
                lm_head_weight,
                cfg,
                transaction=tx_index,
                device=device,
                rng=rngs[name],
            )
            relative_gain = (pre - post) / max(pre, 1e-8)
            checkpoint = {"mean_positive_relative_regression": 0.0}
            if (
                (tx_index + 1) % cfg.retention_checkpoint_every_transactions == 0
                or tx_index + 1 == len(transactions)
            ):
                checkpoint = _checkpoint_regression(
                    state, u, lm_head_weight, device=device
                )
                checkpoint_rows.append(
                    {"seed": seed, "transaction": tx_index, "variant": name, **checkpoint}
                )
            state.records.append(
                {
                    "transaction": tx_index,
                    "source": current[0].source,
                    "pre_nll": pre,
                    "post_nll": post,
                    "relative_new_gain": float(relative_gain),
                    "splits": splits,
                    "child_reuse": child_reuse,
                    "blocked_saturation": blocked,
                    "checkpoint_positive_regression": float(
                        checkpoint["mean_positive_relative_regression"]
                    ),
                }
            )
            _record_rank_state(state, tx_index)

    eval_records: list[dict[str, Any]] = []
    for source in cfg.sources:
        subset = [s for s in eval_sequences if s.source == source]
        foundation_nll, _, _ = nll_and_gradient_targets(
            subset,
            initial,
            u,
            lm_head_weight,
            functional_step=0.0,
            device=device,
            need_gradient=False,
        )
        for name, state in states.items():
            variant_nll, _, _ = nll_and_gradient_targets(
                subset,
                state.system,
                u,
                lm_head_weight,
                functional_step=0.0,
                device=device,
                need_gradient=False,
            )
            eval_records.append(
                {
                    "seed": seed,
                    "source": source,
                    "variant": name,
                    "foundation_nll": foundation_nll,
                    "final_nll": variant_nll,
                    "delta_vs_foundation": variant_nll - foundation_nll,
                }
            )

    causal_rows: list[dict[str, Any]] = []
    for name, state in states.items():
        for row in final_causal_metrics(
            state, eval_sequences, u, lm_head_weight, device=device
        ):
            causal_rows.append({"seed": seed, "variant": name, **row})

    gate = summarize_seed(states, cfg, causal_rows=causal_rows)
    return {
        "seed": seed,
        "router": {
            "centroids": centroids.tolist(),
            "base_assignment": {str(k): int(v) for k, v in base_assignment.items()},
        },
        "variants": {
            name: {
                "summary": gate["variant_summaries"][name],
                "records": state.records,
                "rank_records": state.rank_records,
                "split_records": state.split_records,
            }
            for name, state in states.items()
        },
        "checkpoint_records": checkpoint_rows,
        "eval_records": eval_records,
        "causal_records": causal_rows,
        "gate_summary": gate,
    }


def summarize_experiment(
    runs: list[dict[str, Any]], *, positive_status: str, negative_status: str
) -> dict[str, Any]:
    passed = bool(runs) and all(bool(run["gate_summary"]["pass"]) for run in runs)
    return {
        "status": positive_status if passed else negative_status,
        "pass": passed,
        "scientific_decision": True,
        "passed_seeds": sum(bool(r["gate_summary"]["pass"]) for r in runs),
        "total_seeds": len(runs),
        "hypotheses": {
            "real_hidden_states_have_reusable_functional_geometry": all(
                r["gate_summary"]["gates"]["real_representation_not_immediately_full"]
                and r["gate_summary"]["gates"]["functional_reuse_grows"]
                for r in runs
            ),
            "bounded_certificate_reduces_registered_forgetting_without_replay": all(
                r["gate_summary"]["gates"]["no_replay_candidate"]
                and r["gate_summary"]["gates"]["registered_retention"]
                for r in runs
            ),
            "dependency_partitioned_mitosis_restores_plasticity": all(
                r["gate_summary"]["gates"]["mitosis_improves_plasticity"]
                and r["gate_summary"]["gates"]["split_reduces_conflict"]
                for r in runs
            ),
            "growth_remains_bounded_and_reused": all(
                r["gate_summary"]["gates"]["bounded_growth"]
                and r["gate_summary"]["gates"]["child_reuse"]
                for r in runs
            ),
        },
    }
