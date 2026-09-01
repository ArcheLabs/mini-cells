"""Replay-free subspace-certified mitosis for Core Validation 005.

The learner receives only current transaction data plus Cell-local state (W, Q, route).
All historical examples live exclusively inside HiddenEvaluator and are never passed to
the learner decision functions.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

_DTYPE = torch.float64
VARIANTS = (
    "unsafe_always",
    "certificate_no_growth",
    "certificate_growth",
    "wrong_certificate",
)


@dataclass(frozen=True)
class CoreValidation005Config:
    base_cells: int
    addresses_per_base: int
    feature_dim: int
    output_dim: int
    examples_per_transaction: int
    transactions_per_base: int
    feasibility_threshold: float
    numerical_rank_tolerance: float
    regression_tolerance: float
    transaction_template: tuple[dict[str, Any], ...]
    maximum_false_safe_count: int
    maximum_decision_mismatch_count: int
    maximum_regression_damage_ratio_vs_unsafe: float
    minimum_committed_gain_ratio_vs_unsafe: float
    minimum_growth_rescue_rate: float
    minimum_child_reuse_acceptance_rate: float
    maximum_spawned_cells_per_effective_commit: float
    minimum_wrong_certificate_false_safe_count: int
    minimum_wrong_certificate_regression: float

    @classmethod
    def from_protocol(cls, path: str | Path) -> "CoreValidation005Config":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        world = p["world"]
        mechanism = p["mechanism"]
        gates = p["gates"]
        template = tuple(dict(item) for item in p["curriculum"]["transaction_template"])
        cfg = cls(
            base_cells=int(world["base_cells"]),
            addresses_per_base=int(world["addresses_per_base"]),
            feature_dim=int(world["feature_dim"]),
            output_dim=int(world["output_dim"]),
            examples_per_transaction=int(world["examples_per_transaction"]),
            transactions_per_base=int(world["transactions_per_base"]),
            feasibility_threshold=float(mechanism["feasibility_threshold"]),
            numerical_rank_tolerance=float(mechanism["numerical_rank_tolerance"]),
            regression_tolerance=float(mechanism["regression_tolerance"]),
            transaction_template=template,
            maximum_false_safe_count=int(gates["maximum_false_safe_count"]),
            maximum_decision_mismatch_count=int(gates["maximum_decision_mismatch_count"]),
            maximum_regression_damage_ratio_vs_unsafe=float(
                gates["maximum_regression_damage_ratio_vs_unsafe"]
            ),
            minimum_committed_gain_ratio_vs_unsafe=float(
                gates["minimum_committed_gain_ratio_vs_unsafe"]
            ),
            minimum_growth_rescue_rate=float(gates["minimum_growth_rescue_rate"]),
            minimum_child_reuse_acceptance_rate=float(
                gates["minimum_child_reuse_acceptance_rate"]
            ),
            maximum_spawned_cells_per_effective_commit=float(
                gates["maximum_spawned_cells_per_effective_commit"]
            ),
            minimum_wrong_certificate_false_safe_count=int(
                gates["minimum_wrong_certificate_false_safe_count"]
            ),
            minimum_wrong_certificate_regression=float(
                gates["minimum_wrong_certificate_regression"]
            ),
        )
        if cfg.base_cells <= 0:
            raise ValueError("base_cells must be positive")
        if cfg.addresses_per_base <= 0:
            raise ValueError("addresses_per_base must be positive")
        if cfg.feature_dim <= 0 or cfg.output_dim <= 0:
            raise ValueError("feature/output dimensions must be positive")
        if len(template) != cfg.transactions_per_base:
            raise ValueError("transaction template length must equal transactions_per_base")
        for item in template:
            if int(item["address_offset"]) >= cfg.addresses_per_base:
                raise ValueError("transaction address_offset outside address range")
            dims = [int(x) for x in item["activation_dims"]]
            residual = [int(x) for x in item["residual_dims"]]
            if not dims or not residual:
                raise ValueError("every transaction requires activation and residual dimensions")
            if max(dims + residual) >= cfg.feature_dim:
                raise ValueError("transaction dimension outside feature_dim")
            if not set(residual).issubset(dims):
                raise ValueError("residual_dims must be a subset of activation_dims")
        return cfg


def smoke_config(config: CoreValidation005Config) -> CoreValidation005Config:
    return replace(config, base_cells=1)


@dataclass
class LearnerAudit:
    old_sample_accesses: int = 0
    old_label_accesses: int = 0
    replay_items_retained: int = 0


@dataclass
class LinearCell:
    weight: torch.Tensor
    basis: torch.Tensor

    @classmethod
    def empty(cls, config: CoreValidation005Config) -> "LinearCell":
        return cls(
            weight=torch.zeros(config.output_dim, config.feature_dim, dtype=_DTYPE),
            basis=torch.zeros(config.feature_dim, 0, dtype=_DTYPE),
        )

    def clone(self) -> "LinearCell":
        return LinearCell(self.weight.clone(), self.basis.clone())


class SubspaceCellModel:
    """Fixed base routing plus monotonic exact-address private growth routes."""

    def __init__(self, config: CoreValidation005Config):
        self.config = config
        self.base_cells = [LinearCell.empty(config) for _ in range(config.base_cells)]
        self.private_cells: dict[int, LinearCell] = {}
        self.audit = LearnerAudit()

    def clone(self) -> "SubspaceCellModel":
        other = SubspaceCellModel(self.config)
        other.base_cells = [cell.clone() for cell in self.base_cells]
        other.private_cells = {
            address: cell.clone() for address, cell in self.private_cells.items()
        }
        other.audit = copy.deepcopy(self.audit)
        return other

    def base_id(self, address: int) -> int:
        base_id = int(address) // self.config.addresses_per_base
        if not 0 <= base_id < self.config.base_cells:
            raise ValueError(f"address {address} has no base Cell")
        return base_id

    def write_cell(self, address: int) -> tuple[str, LinearCell]:
        if int(address) in self.private_cells:
            return f"private:{int(address)}", self.private_cells[int(address)]
        base_id = self.base_id(address)
        return f"base:{base_id}", self.base_cells[base_id]

    def forward(self, z: torch.Tensor, address: int) -> torch.Tensor:
        base = self.base_cells[self.base_id(address)]
        out = z @ base.weight.T
        private = self.private_cells.get(int(address))
        if private is not None:
            out = out + z @ private.weight.T
        return out

    def spawn_private(self, address: int) -> LinearCell:
        address = int(address)
        if address in self.private_cells:
            raise ValueError("address already owns a private Cell")
        cell = LinearCell.empty(self.config)
        self.private_cells[address] = cell
        return cell

    def total_certificate_scalars(self) -> int:
        cells = [*self.base_cells, *self.private_cells.values()]
        return int(sum(cell.basis.numel() for cell in cells))

    def maximum_certificate_rank(self) -> int:
        cells = [*self.base_cells, *self.private_cells.values()]
        return max((cell.basis.shape[1] for cell in cells), default=0)


@dataclass(frozen=True)
class Transaction:
    index: int
    base_id: int
    address: int
    kind: str
    z: torch.Tensor
    residual: torch.Tensor


def _orthonormal_coefficients(
    examples: int,
    width: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    raw = torch.randn(examples, width, generator=generator, dtype=_DTYPE)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q[:, :width] * math.sqrt(examples)


def make_transaction(
    config: CoreValidation005Config,
    *,
    base_id: int,
    local_index: int,
    seed: int,
) -> Transaction:
    spec = config.transaction_template[local_index]
    active = [int(x) for x in spec["activation_dims"]]
    residual_dims = [int(x) for x in spec["residual_dims"]]
    scales = {int(k): float(v) for k, v in spec.get("dimension_scales", {}).items()}
    generator = torch.Generator().manual_seed(seed)
    coeff = _orthonormal_coefficients(
        config.examples_per_transaction,
        len(active),
        generator=generator,
    )
    z = torch.zeros(
        config.examples_per_transaction,
        config.feature_dim,
        dtype=_DTYPE,
    )
    for slot, dim in enumerate(active):
        z[:, dim] = coeff[:, slot] * scales.get(dim, 1.0)
    decoder = torch.randn(
        len(residual_dims),
        config.output_dim,
        generator=generator,
        dtype=_DTYPE,
    ) / math.sqrt(len(residual_dims))
    residual = z[:, residual_dims] @ decoder
    address = base_id * config.addresses_per_base + int(spec["address_offset"])
    return Transaction(
        index=base_id * config.transactions_per_base + local_index,
        base_id=base_id,
        address=address,
        kind=str(spec["kind"]),
        z=z,
        residual=residual,
    )


def transaction_stream(config: CoreValidation005Config, *, seed: int) -> list[Transaction]:
    stream: list[Transaction] = []
    for base_id in range(config.base_cells):
        for local_index in range(config.transactions_per_base):
            stream.append(
                make_transaction(
                    config,
                    base_id=base_id,
                    local_index=local_index,
                    seed=seed + base_id * 1000 + local_index * 17,
                )
            )
    return stream


def extend_basis(
    basis: torch.Tensor,
    z: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    """Add the row-span of current activations without retaining those activations."""
    residual = z.T
    if basis.numel():
        residual = residual - basis @ (basis.T @ residual)
    u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    keep = singular > tolerance
    if not bool(keep.any()):
        return basis
    new_directions = u[:, keep]
    combined = new_directions if not basis.numel() else torch.cat([basis, new_directions], dim=1)
    # SVD re-orthogonalization prevents drift after many incremental additions.
    u2, singular2, _ = torch.linalg.svd(combined, full_matrices=False)
    return u2[:, singular2 > tolerance]


def projector_from_basis(feature_dim: int, basis: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(feature_dim, dtype=_DTYPE)
    if not basis.numel():
        return eye
    return eye - basis @ basis.T


def _truncated_solve(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable least-squares solve with an absolute singular-value threshold."""
    u, singular, vh = torch.linalg.svd(design, full_matrices=False)
    keep = singular > tolerance
    solution = torch.zeros(design.shape[1], target.shape[1], dtype=_DTYPE)
    if bool(keep.any()):
        projected = (u[:, keep].T @ target) / singular[keep, None]
        solution = vh[keep, :].T @ projected
    return solution, singular


def constrained_update(
    z: torch.Tensor,
    residual: torch.Tensor,
    basis: torch.Tensor,
    *,
    numerical_rank_tolerance: float,
) -> dict[str, Any]:
    """Solve min ||Z dW^T - R|| with dW Q = 0."""
    feature_dim = z.shape[1]
    projector = projector_from_basis(feature_dim, basis)
    free_design = z @ projector
    solution, singular = _truncated_solve(
        free_design,
        residual,
        tolerance=numerical_rank_tolerance,
    )
    predicted = free_design @ solution
    denom = torch.sum(residual.square()).clamp_min(torch.tensor(1e-30, dtype=_DTYPE))
    fit_error = float(torch.sum((predicted - residual).square()) / denom)
    delta_weight = solution.T @ projector
    protected_change = (
        float(torch.max(torch.abs(delta_weight @ basis))) if basis.numel() else 0.0
    )
    free_rank = int((singular > numerical_rank_tolerance).sum())
    return {
        "fit_error": fit_error,
        "delta_weight": delta_weight,
        "protected_change": protected_change,
        "free_design_rank": free_rank,
    }


def wrong_basis(basis: torch.Tensor) -> torch.Tensor:
    """Same rank and norm as Q, but deliberately wrong geometry."""
    if not basis.numel():
        return basis
    return torch.roll(basis, shifts=1, dims=0)


class HiddenEvaluator:
    """Full historical oracle. It is evaluator-only and never passed to learner decisions."""

    def __init__(self, config: CoreValidation005Config):
        self.config = config
        self.behavior_history: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.cell_history: dict[str, list[torch.Tensor]] = {
            f"base:{i}": [] for i in range(config.base_cells)
        }

    def behavior_mse(self, model: SubspaceCellModel) -> float:
        if not self.behavior_history:
            return 0.0
        values = [
            float(torch.mean((model.forward(z, address) - target).square()))
            for address, z, target in self.behavior_history
        ]
        return float(sum(values) / len(values))

    def full_basis(self, cell_key: str) -> torch.Tensor:
        items = self.cell_history.get(cell_key, [])
        if not items:
            return torch.zeros(self.config.feature_dim, 0, dtype=_DTYPE)
        return extend_basis(
            torch.zeros(self.config.feature_dim, 0, dtype=_DTYPE),
            torch.cat(items, dim=0),
            tolerance=self.config.numerical_rank_tolerance,
        )

    def record_commit(
        self,
        *,
        model: SubspaceCellModel,
        transaction: Transaction,
        private_active: bool,
    ) -> None:
        z = transaction.z.detach().clone()
        target = model.forward(z, transaction.address).detach().clone()
        self.behavior_history.append((transaction.address, z, target))
        self.cell_history[f"base:{transaction.base_id}"].append(z)
        if private_active:
            self.cell_history.setdefault(f"private:{transaction.address}", []).append(z)


def _learner_decision(
    cell: LinearCell,
    transaction: Transaction,
    config: CoreValidation005Config,
    *,
    use_wrong_certificate: bool,
) -> dict[str, Any]:
    """Learner-visible decision: only current data and Cell-local Q are accepted."""
    basis = wrong_basis(cell.basis) if use_wrong_certificate else cell.basis
    solved = constrained_update(
        transaction.z,
        transaction.residual,
        basis,
        numerical_rank_tolerance=config.numerical_rank_tolerance,
    )
    solved["feasible"] = bool(solved["fit_error"] <= config.feasibility_threshold)
    solved["certificate_rank"] = int(basis.shape[1])
    solved["true_state_rank"] = int(cell.basis.shape[1])
    return solved


def _new_gain(
    model: SubspaceCellModel,
    transaction: Transaction,
    target: torch.Tensor,
) -> float:
    after = model.forward(transaction.z, transaction.address)
    denom = torch.sum(transaction.residual.square()).clamp_min(
        torch.tensor(1e-30, dtype=_DTYPE)
    )
    error = float(torch.sum((after - target).square()) / denom)
    return max(0.0, 1.0 - error)


def run_variant(
    config: CoreValidation005Config,
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    model = SubspaceCellModel(config)
    evaluator = HiddenEvaluator(config)
    records: list[dict[str, Any]] = []
    cumulative_gain = 0.0
    cumulative_damage = 0.0
    false_safe_count = 0
    safe_decisions = 0
    decision_mismatch_count = 0
    certificate_decisions = 0
    growth_attempts = 0
    growth_commits = 0
    child_reuse_attempts = 0
    child_reuse_commits = 0
    max_certificate_oracle_gap = 0.0

    for transaction in transaction_stream(config, seed=seed + 1009):
        before_output = model.forward(transaction.z, transaction.address).detach().clone()
        target = before_output + transaction.residual
        regression_before = evaluator.behavior_mse(model)

        private_before = transaction.address in model.private_cells
        write_key, write_cell = model.write_cell(transaction.address)
        oracle_basis = evaluator.full_basis(write_key)
        oracle = constrained_update(
            transaction.z,
            transaction.residual,
            oracle_basis,
            numerical_rank_tolerance=config.numerical_rank_tolerance,
        )
        oracle_feasible = bool(oracle["fit_error"] <= config.feasibility_threshold)

        committed = False
        growth_attempted = False
        growth_committed = False
        safe_direct_commit = False
        direct_fit_error: float | None = None
        certificate_rank: int | None = None
        direct_feasible: bool | None = None

        if variant == "unsafe_always":
            direct = constrained_update(
                transaction.z,
                transaction.residual,
                torch.zeros(config.feature_dim, 0, dtype=_DTYPE),
                numerical_rank_tolerance=config.numerical_rank_tolerance,
            )
            write_cell.weight = write_cell.weight + direct["delta_weight"]
            direct_fit_error = float(direct["fit_error"])
            direct_feasible = True
            committed = True
        else:
            decision = _learner_decision(
                write_cell,
                transaction,
                config,
                use_wrong_certificate=(variant == "wrong_certificate"),
            )
            direct_fit_error = float(decision["fit_error"])
            direct_feasible = bool(decision["feasible"])
            certificate_rank = int(decision["certificate_rank"])
            certificate_decisions += 1
            mismatch = direct_feasible != oracle_feasible
            decision_mismatch_count += int(mismatch)
            max_certificate_oracle_gap = max(
                max_certificate_oracle_gap,
                abs(direct_fit_error - float(oracle["fit_error"])),
            )

            if direct_feasible:
                write_cell.weight = write_cell.weight + decision["delta_weight"]
                committed = True
                safe_direct_commit = True
                safe_decisions += 1
            elif variant in {"certificate_growth", "wrong_certificate"} and not private_before:
                growth_attempted = True
                growth_attempts += 1
                child = LinearCell.empty(config)
                growth = _learner_decision(
                    child,
                    transaction,
                    config,
                    use_wrong_certificate=False,
                )
                if bool(growth["feasible"]):
                    child.weight = child.weight + growth["delta_weight"]
                    model.private_cells[transaction.address] = child
                    committed = True
                    growth_committed = True
                    growth_commits += 1

        if private_before:
            child_reuse_attempts += 1
            if committed:
                child_reuse_commits += 1

        regression_after = evaluator.behavior_mse(model)
        step_damage = max(0.0, regression_after - regression_before)
        cumulative_damage += step_damage
        if safe_direct_commit and step_damage > config.regression_tolerance:
            false_safe_count += 1

        gain = _new_gain(model, transaction, target) if committed else 0.0
        cumulative_gain += gain

        if committed:
            # Every committed input remains dependent on its base Cell, even when a private
            # additive Cell owns the write. Therefore the base certificate always grows.
            base = model.base_cells[transaction.base_id]
            base.basis = extend_basis(
                base.basis,
                transaction.z,
                tolerance=config.numerical_rank_tolerance,
            )
            if transaction.address in model.private_cells:
                private = model.private_cells[transaction.address]
                private.basis = extend_basis(
                    private.basis,
                    transaction.z,
                    tolerance=config.numerical_rank_tolerance,
                )
            evaluator.record_commit(
                model=model,
                transaction=transaction,
                private_active=(transaction.address in model.private_cells),
            )

        records.append(
            {
                "transaction": transaction.index,
                "base_id": transaction.base_id,
                "address": transaction.address,
                "kind": transaction.kind,
                "variant": variant,
                "private_before": private_before,
                "committed": committed,
                "growth_attempted": growth_attempted,
                "growth_committed": growth_committed,
                "direct_feasible": direct_feasible,
                "oracle_feasible": oracle_feasible,
                "direct_fit_error": direct_fit_error,
                "oracle_fit_error": float(oracle["fit_error"]),
                "certificate_rank": certificate_rank,
                "oracle_rank": int(oracle_basis.shape[1]),
                "new_gain": gain,
                "incremental_global_regression": step_damage,
                "historical_items_visible_to_evaluator": len(evaluator.behavior_history),
                "learner_old_sample_accesses": model.audit.old_sample_accesses,
            }
        )

    effective_commits = sum(bool(r["committed"]) for r in records)
    summary = {
        "transactions": len(records),
        "effective_commits": effective_commits,
        "effective_acceptance_rate": effective_commits / max(len(records), 1),
        "cumulative_committed_new_gain": cumulative_gain,
        "cumulative_positive_global_regression": cumulative_damage,
        "false_safe_count": false_safe_count,
        "false_safe_rate": false_safe_count / max(safe_decisions, 1),
        "safe_decisions": safe_decisions,
        "decision_mismatch_count": decision_mismatch_count,
        "decision_mismatch_rate": decision_mismatch_count / max(certificate_decisions, 1),
        "growth_attempts": growth_attempts,
        "growth_commits": growth_commits,
        "growth_rescue_rate": growth_commits / max(growth_attempts, 1),
        "child_reuse_attempts": child_reuse_attempts,
        "child_reuse_commits": child_reuse_commits,
        "child_reuse_acceptance_rate": child_reuse_commits / max(child_reuse_attempts, 1),
        "spawned_cells": len(model.private_cells),
        "spawned_cells_per_effective_commit": len(model.private_cells) / max(effective_commits, 1),
        "maximum_certificate_rank": model.maximum_certificate_rank(),
        "certificate_state_scalars": model.total_certificate_scalars(),
        "learner_old_sample_accesses": model.audit.old_sample_accesses,
        "learner_old_label_accesses": model.audit.old_label_accesses,
        "learner_replay_items_retained": model.audit.replay_items_retained,
        "max_certificate_oracle_fit_error_gap": max_certificate_oracle_gap,
        "final_hidden_history_items": len(evaluator.behavior_history),
    }
    return {"summary": summary, "records": records}


def summarize_seed(
    config: CoreValidation005Config,
    *,
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    unsafe = variants["unsafe_always"]["summary"]
    no_growth = variants["certificate_no_growth"]["summary"]
    growth = variants["certificate_growth"]["summary"]
    wrong = variants["wrong_certificate"]["summary"]
    damage_ratio = growth["cumulative_positive_global_regression"] / max(
        unsafe["cumulative_positive_global_regression"], 1e-30
    )
    gain_ratio = growth["cumulative_committed_new_gain"] / max(
        unsafe["cumulative_committed_new_gain"], 1e-30
    )
    gates = {
        "no_replay": (
            growth["learner_old_sample_accesses"] == 0
            and growth["learner_old_label_accesses"] == 0
            and growth["learner_replay_items_retained"] == 0
        ),
        "zero_false_safe": growth["false_safe_count"] <= config.maximum_false_safe_count,
        "certificate_matches_full_history": (
            growth["decision_mismatch_count"] <= config.maximum_decision_mismatch_count
        ),
        "stability": damage_ratio <= config.maximum_regression_damage_ratio_vs_unsafe,
        "plasticity": gain_ratio >= config.minimum_committed_gain_ratio_vs_unsafe,
        "growth_rescue": growth["growth_rescue_rate"] >= config.minimum_growth_rescue_rate,
        "child_reuse": (
            growth["child_reuse_acceptance_rate"]
            >= config.minimum_child_reuse_acceptance_rate
        ),
        "bounded_growth": (
            growth["spawned_cells_per_effective_commit"]
            <= config.maximum_spawned_cells_per_effective_commit
        ),
        "growth_improves_no_growth": (
            growth["cumulative_committed_new_gain"]
            > no_growth["cumulative_committed_new_gain"]
        ),
        "wrong_certificate_causal_failure": (
            wrong["false_safe_count"] >= config.minimum_wrong_certificate_false_safe_count
            and wrong["cumulative_positive_global_regression"]
            >= config.minimum_wrong_certificate_regression
        ),
    }
    return {
        "pass": bool(all(gates.values())),
        "gates": gates,
        "regression_damage_ratio_vs_unsafe": float(damage_ratio),
        "committed_gain_ratio_vs_unsafe": float(gain_ratio),
        "variant_summaries": {name: run["summary"] for name, run in variants.items()},
    }


def run_primary_seed(
    config: CoreValidation005Config,
    *,
    seed: int,
) -> dict[str, Any]:
    variants = {name: run_variant(config, variant=name, seed=seed) for name in VARIANTS}
    return {
        "seed": int(seed),
        "variants": variants,
        "gate_summary": summarize_seed(config, variants=variants),
    }


def summarize_experiment(
    runs: list[dict[str, Any]],
    *,
    positive_status: str,
    negative_status: str,
) -> dict[str, Any]:
    passed = bool(runs) and all(bool(run["gate_summary"]["pass"]) for run in runs)
    return {
        "status": positive_status if passed else negative_status,
        "pass": passed,
        "scientific_decision": True,
        "passed_seeds": sum(bool(run["gate_summary"]["pass"]) for run in runs),
        "total_seeds": len(runs),
        "hypotheses": {
            "finite_certificate_replaces_replay_for_registered_subspace": all(
                run["gate_summary"]["gates"]["no_replay"]
                and run["gate_summary"]["gates"]["certificate_matches_full_history"]
                and run["gate_summary"]["gates"]["zero_false_safe"]
                for run in runs
            ),
            "free_subspace_preserves_stability_and_plasticity": all(
                run["gate_summary"]["gates"]["stability"]
                and run["gate_summary"]["gates"]["plasticity"]
                for run in runs
            ),
            "saturation_triggered_growth_restores_plasticity": all(
                run["gate_summary"]["gates"]["growth_rescue"]
                and run["gate_summary"]["gates"]["child_reuse"]
                and run["gate_summary"]["gates"]["bounded_growth"]
                for run in runs
            ),
            "certificate_geometry_is_causal": all(
                run["gate_summary"]["gates"]["wrong_certificate_causal_failure"]
                for run in runs
            ),
        },
    }
