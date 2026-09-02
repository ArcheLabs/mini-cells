from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from minicells.constructive_clm_003 import StructuralBridgeConfig, learn_structural_roots
from minicells.subspace_mitosis_005 import constrained_update, extend_basis, projector_from_basis

_DTYPE = torch.float64
MODES = ("simultaneous", "sequential")


@dataclass(frozen=True)
class ComputationConfig:
    root_count: int = 12
    hidden_dim: int = 16
    operator_spectral_norm: float = 0.40
    acquisition_batches_per_cell: int = 4
    acquisition_batch_size: int = 32
    acquisition_noise: float = 0.002
    route_noise: float = 0.010
    ridge: float = 1e-8
    evaluation_cases_per_mode: int = 64
    evaluation_batch_size: int = 24
    minimum_active_cells: int = 2
    maximum_active_cells: int = 4
    certificate_history_cases: int = 6
    mutation_examples: int = 6
    numerical_rank_tolerance: float = 1e-10


@dataclass
class OperatorCell:
    cell_id: int
    route_key: torch.Tensor
    weight: torch.Tensor
    gram: torch.Tensor
    cross: torch.Tensor
    basis: torch.Tensor
    observations: int = 0

    @classmethod
    def empty(cls, cell_id: int, route_key: torch.Tensor, hidden_dim: int) -> "OperatorCell":
        return cls(
            cell_id=int(cell_id),
            route_key=F.normalize(route_key.to(dtype=_DTYPE), dim=0),
            weight=torch.zeros(hidden_dim, hidden_dim, dtype=_DTYPE),
            gram=torch.zeros(hidden_dim, hidden_dim, dtype=_DTYPE),
            cross=torch.zeros(hidden_dim, hidden_dim, dtype=_DTYPE),
            basis=torch.zeros(hidden_dim, 0, dtype=_DTYPE),
        )

    def clone(self) -> "OperatorCell":
        return OperatorCell(
            self.cell_id,
            self.route_key.clone(),
            self.weight.clone(),
            self.gram.clone(),
            self.cross.clone(),
            self.basis.clone(),
            self.observations,
        )


class ComputationalCellModel:
    """Sparse route-addressed residual operators with Cell-local sufficient statistics."""

    def __init__(self, route_keys: torch.Tensor, hidden_dim: int, *, ridge: float) -> None:
        keys = F.normalize(route_keys.to(dtype=_DTYPE), dim=1)
        self.route_keys = keys.clone()
        self.hidden_dim = int(hidden_dim)
        self.ridge = float(ridge)
        self.cells = [OperatorCell.empty(i, keys[i], self.hidden_dim) for i in range(len(keys))]
        self.raw_examples_retained = 0
        self.replay_accesses = 0

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def clone(self) -> "ComputationalCellModel":
        other = object.__new__(ComputationalCellModel)
        other.route_keys = self.route_keys.clone()
        other.hidden_dim = self.hidden_dim
        other.ridge = self.ridge
        other.cells = [cell.clone() for cell in self.cells]
        other.raw_examples_retained = self.raw_examples_retained
        other.replay_accesses = self.replay_accesses
        return other

    def route_tokens(self, tokens: torch.Tensor) -> list[int]:
        query = tokens.to(dtype=_DTYPE)
        if query.ndim == 1:
            query = query[None, :]
        query = F.normalize(query, dim=1)
        return [int(v) for v in torch.argmax(query @ self.route_keys.T, dim=1).tolist()]

    def observe_operator(self, route_context: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor) -> int:
        token = route_context.mean(dim=0) if route_context.ndim == 2 else route_context
        cell_id = self.route_tokens(token)[0]
        z = hidden.to(dtype=_DTYPE)
        r = residual.to(dtype=_DTYPE)
        cell = self.cells[cell_id]
        cell.gram += z.T @ z
        cell.cross += r.T @ z
        cell.observations += int(z.shape[0])
        return cell_id

    def fit_operators(self) -> None:
        eye = torch.eye(self.hidden_dim, dtype=_DTYPE)
        for cell in self.cells:
            if cell.observations:
                cell.weight = cell.cross @ torch.linalg.pinv(cell.gram + self.ridge * eye)

    def execute_ids(self, hidden: torch.Tensor, cell_ids: Iterable[int], *, mode: str, return_trace: bool = False) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        ids = [int(v) for v in cell_ids]
        if mode not in MODES:
            raise ValueError(f"unknown composition mode {mode!r}")
        h = hidden.to(dtype=_DTYPE)
        trace: list[dict[str, Any]] = []
        if mode == "simultaneous":
            base = h
            total = torch.zeros_like(base)
            for cell_id in ids:
                residual = base @ self.cells[cell_id].weight.T
                total += residual
                if return_trace:
                    trace.append({"cell_id": cell_id, "incoming": base.detach().clone(), "residual": residual.detach().clone()})
            h = base + total
        else:
            for cell_id in ids:
                incoming = h
                residual = incoming @ self.cells[cell_id].weight.T
                h = incoming + residual
                if return_trace:
                    trace.append({"cell_id": cell_id, "incoming": incoming.detach().clone(), "residual": residual.detach().clone()})
        return h, trace

    def execute(self, hidden: torch.Tensor, route_tokens: torch.Tensor, *, mode: str, return_trace: bool = False) -> tuple[torch.Tensor, list[int], list[dict[str, Any]]]:
        ids = self.route_tokens(route_tokens)
        output, trace = self.execute_ids(hidden, ids, mode=mode, return_trace=return_trace)
        return output, ids, trace


def _mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left - right).square()).item())


def _relative_frobenius(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.norm(right).clamp_min(torch.tensor(1e-30, dtype=_DTYPE))
    return float((torch.linalg.norm(left - right) / denom).item())


def _scaled_random_operator(hidden_dim: int, scale: float, *, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(hidden_dim, hidden_dim, generator=generator, dtype=_DTYPE)
    largest = torch.linalg.svdvals(raw)[0].clamp_min(torch.tensor(1e-12, dtype=_DTYPE))
    return raw * (float(scale) / largest)


def _true_execute(hidden: torch.Tensor, operators: torch.Tensor, cell_ids: Iterable[int], *, mode: str) -> torch.Tensor:
    ids = [int(v) for v in cell_ids]
    h = hidden.to(dtype=_DTYPE)
    if mode == "simultaneous":
        base = h
        residual = torch.zeros_like(base)
        for cell_id in ids:
            residual += base @ operators[cell_id].T
        return base + residual
    if mode == "sequential":
        for cell_id in ids:
            h = h + h @ operators[cell_id].T
        return h
    raise ValueError(f"unknown mode {mode!r}")


def _route_tokens(route_keys: torch.Tensor, cell_ids: Iterable[int], *, noise: float, generator: torch.Generator) -> torch.Tensor:
    rows = []
    for cell_id in cell_ids:
        key = route_keys[int(cell_id)]
        perturbed = key + float(noise) * torch.randn(key.shape, generator=generator, dtype=_DTYPE)
        rows.append(F.normalize(perturbed, dim=0))
    return torch.stack(rows)


def _acquire_operators(model: ComputationalCellModel, true_operators: torch.Tensor, cfg: ComputationConfig, *, generator: torch.Generator, rng: random.Random) -> dict[str, Any]:
    schedule = [(cell_id, batch) for cell_id in range(cfg.root_count) for batch in range(cfg.acquisition_batches_per_cell)]
    rng.shuffle(schedule)
    correct = 0
    for cell_id, _batch in schedule:
        context = _route_tokens(model.route_keys, [cell_id], noise=cfg.route_noise, generator=generator)[0]
        hidden = torch.randn(cfg.acquisition_batch_size, cfg.hidden_dim, generator=generator, dtype=_DTYPE)
        residual = hidden @ true_operators[cell_id].T
        residual += cfg.acquisition_noise * torch.randn(residual.shape, generator=generator, dtype=_DTYPE)
        correct += int(model.observe_operator(context, hidden, residual) == cell_id)
    model.fit_operators()
    errors = [_relative_frobenius(model.cells[i].weight, true_operators[i]) for i in range(cfg.root_count)]
    distances = [float(torch.linalg.norm(model.cells[i].weight - model.cells[j].weight).item()) for i in range(cfg.root_count) for j in range(i + 1, cfg.root_count)]
    return {
        "transactions": len(schedule),
        "route_accuracy": correct / max(len(schedule), 1),
        "mean_operator_relative_error": _mean(errors),
        "max_operator_relative_error": max(errors, default=0.0),
        "minimum_pairwise_operator_distance": min(distances, default=0.0),
        "raw_examples_retained": model.raw_examples_retained,
    }


def _evaluate_mode(model: ComputationalCellModel, true_operators: torch.Tensor, cfg: ComputationConfig, *, mode: str, generator: torch.Generator, rng: random.Random) -> dict[str, Any]:
    main_mse, single_mse, dense_mse, wrong_mse = [], [], [], []
    active_counts, route_exact, order_effect, permutation_error = [], [], [], []
    for _ in range(cfg.evaluation_cases_per_mode):
        active = rng.randint(cfg.minimum_active_cells, cfg.maximum_active_cells)
        ids = rng.sample(range(model.cell_count), active)
        hidden = torch.randn(cfg.evaluation_batch_size, cfg.hidden_dim, generator=generator, dtype=_DTYPE)
        tokens = _route_tokens(model.route_keys, ids, noise=cfg.route_noise, generator=generator)
        target = _true_execute(hidden, true_operators, ids, mode=mode)
        predicted, routed, _ = model.execute(hidden, tokens, mode=mode)
        main_mse.append(_mse(predicted, target))
        route_exact.append(float(routed == ids))
        active_counts.append(active)
        single, _ = model.execute_ids(hidden, [ids[0]], mode=mode)
        single_mse.append(_mse(single, target))
        dense, _ = model.execute_ids(hidden, range(model.cell_count), mode=mode)
        dense_mse.append(_mse(dense, target))
        other_mode = "simultaneous" if mode == "sequential" else "sequential"
        wrong, _ = model.execute_ids(hidden, ids, mode=other_mode)
        wrong_mse.append(_mse(wrong, target))
        if mode == "sequential":
            reversed_target = _true_execute(hidden, true_operators, reversed(ids), mode="sequential")
            order_effect.append(_mse(reversed_target, target))
        else:
            permuted, _ = model.execute_ids(hidden, reversed(ids), mode="simultaneous")
            permutation_error.append(_mse(permuted, predicted))
    mean_active = _mean(active_counts)
    fraction = mean_active / max(model.cell_count, 1)
    return {
        "mode": mode,
        "cases": cfg.evaluation_cases_per_mode,
        "all_compositions_unseen_during_training": True,
        "mean_mse": _mean(main_mse),
        "max_mse": max(main_mse, default=0.0),
        "single_cell_baseline_mse": _mean(single_mse),
        "dense_all_cells_baseline_mse": _mean(dense_mse),
        "wrong_semantics_baseline_mse": _mean(wrong_mse),
        "exact_route_sequence_accuracy": _mean(route_exact),
        "mean_active_cells": mean_active,
        "maximum_active_cells": max(active_counts, default=0),
        "total_cells": model.cell_count,
        "cell_execution_fraction_vs_dense": fraction,
        "cell_execution_savings_vs_dense": 1.0 - fraction,
        "mean_true_order_effect_mse": _mean(order_effect),
        "mean_simultaneous_permutation_mse": _mean(permutation_error),
    }


def _target_incoming(trace: list[dict[str, Any]], target_cell: int) -> torch.Tensor:
    for item in trace:
        if int(item["cell_id"]) == int(target_cell):
            incoming = item["incoming"]
            if int(incoming.shape[0]) != 1:
                raise ValueError("certificate trace expects batch size one")
            return incoming[0].clone()
    raise ValueError("target cell missing from history trace")


def _protected_mutation_probe(model: ComputationalCellModel, cfg: ComputationConfig, *, generator: torch.Generator, rng: random.Random) -> dict[str, Any]:
    target = rng.randrange(model.cell_count)
    history_rows = []
    history_records = []
    for _ in range(cfg.certificate_history_cases):
        active = rng.randint(cfg.minimum_active_cells, cfg.maximum_active_cells)
        others = rng.sample([i for i in range(model.cell_count) if i != target], active - 1)
        ids = list(others)
        ids.insert(rng.randrange(active), target)
        hidden = torch.randn(1, cfg.hidden_dim, generator=generator, dtype=_DTYPE)
        before, trace = model.execute_ids(hidden, ids, mode="sequential", return_trace=True)
        history_rows.append(_target_incoming(trace, target))
        history_records.append((hidden.clone(), ids, before.clone()))
    basis = extend_basis(torch.zeros(cfg.hidden_dim, 0, dtype=_DTYPE), torch.stack(history_rows), tolerance=cfg.numerical_rank_tolerance)
    model.cells[target].basis = basis.clone()
    z = torch.randn(cfg.mutation_examples, cfg.hidden_dim, generator=generator, dtype=_DTYPE)
    projector = projector_from_basis(cfg.hidden_dim, basis)
    desired_map = 0.20 * torch.randn(cfg.hidden_dim, cfg.hidden_dim, generator=generator, dtype=_DTYPE)
    residual = (z @ projector) @ desired_map.T
    safe = constrained_update(z, residual, basis, numerical_rank_tolerance=cfg.numerical_rank_tolerance)
    unsafe = constrained_update(z, residual, torch.zeros(cfg.hidden_dim, 0, dtype=_DTYPE), numerical_rank_tolerance=cfg.numerical_rank_tolerance)
    safe_model, unsafe_model = model.clone(), model.clone()
    safe_model.cells[target].weight += safe["delta_weight"]
    unsafe_model.cells[target].weight += unsafe["delta_weight"]
    safe_hist, unsafe_hist = [], []
    for hidden, ids, before in history_records:
        safe_after, _ = safe_model.execute_ids(hidden, ids, mode="sequential")
        unsafe_after, _ = unsafe_model.execute_ids(hidden, ids, mode="sequential")
        safe_hist.append(_mse(safe_after, before))
        unsafe_hist.append(_mse(unsafe_after, before))
    unrelated = 0.0
    for i in range(model.cell_count):
        if i != target:
            unrelated = max(unrelated, float(torch.max(torch.abs(safe_model.cells[i].weight - model.cells[i].weight)).item()))
    safe_pred = z @ safe["delta_weight"].T
    unsafe_pred = z @ unsafe["delta_weight"].T
    unsafe_protected = float(torch.max(torch.abs(unsafe["delta_weight"] @ basis)).item()) if basis.numel() else 0.0
    return {
        "target_cell": int(target),
        "certificate_rank": int(basis.shape[1]),
        "learner_replay_accesses": 0,
        "learner_raw_history_retained": 0,
        "safe_fit_error": _mse(safe_pred, residual),
        "unsafe_fit_error": _mse(unsafe_pred, residual),
        "safe_reported_relative_fit_error": float(safe["fit_error"]),
        "safe_protected_change": float(safe["protected_change"]),
        "unsafe_protected_change": unsafe_protected,
        "safe_historical_composition_mse": _mean(safe_hist),
        "unsafe_historical_composition_mse": _mean(unsafe_hist),
        "unrelated_cell_parameter_drift": unrelated,
        "route_key_drift": float(torch.max(torch.abs(safe_model.route_keys - model.route_keys)).item()),
    }


def run_seed(seed: int, cfg: ComputationConfig = ComputationConfig()) -> dict[str, Any]:
    seed = int(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0xC1A004)
    rng = random.Random(seed ^ 0x51A004)
    route_keys, bridge = learn_structural_roots(
        seed,
        StructuralBridgeConfig(
            max_transactions=1024,
            initial_factors=6,
            final_factors=cfg.root_count,
            context_dim=48,
            effect_dim=40,
            bootstrap_cycles=6,
            introduction_repeats=4,
            growth_alpha=0.60,
            stabilization_tail=64,
            samples_per_transaction=16,
            train_noise=0.02,
        ),
    )
    model = ComputationalCellModel(route_keys, cfg.hidden_dim, ridge=cfg.ridge)
    true_operators = torch.stack([_scaled_random_operator(cfg.hidden_dim, cfg.operator_spectral_norm, generator=generator) for _ in range(cfg.root_count)])
    acquisition = _acquire_operators(model, true_operators, cfg, generator=generator, rng=rng)
    simultaneous = _evaluate_mode(model, true_operators, cfg, mode="simultaneous", generator=generator, rng=rng)
    sequential = _evaluate_mode(model, true_operators, cfg, mode="sequential", generator=generator, rng=rng)
    mutation = _protected_mutation_probe(model, cfg, generator=generator, rng=rng)
    main = max(simultaneous["mean_mse"], sequential["mean_mse"])
    single = min(simultaneous["single_cell_baseline_mse"], sequential["single_cell_baseline_mse"])
    wrong = min(simultaneous["wrong_semantics_baseline_mse"], sequential["wrong_semantics_baseline_mse"])
    dense = min(simultaneous["dense_all_cells_baseline_mse"], sequential["dense_all_cells_baseline_mse"])
    gates = {
        "structural_bridge_valid": bool(int(bridge.get("root_cells", -1)) == cfg.root_count and int(bridge.get("covered_factors", -1)) == cfg.root_count and int(bridge.get("duplicate_assignments", 1)) == 0 and float(bridge.get("mean_matched_root_key_cosine", 0.0)) >= 0.985),
        "operator_acquisition_routes": acquisition["route_accuracy"] >= 0.99,
        "operator_learning_quality": acquisition["mean_operator_relative_error"] <= 0.01 and acquisition["max_operator_relative_error"] <= 0.02,
        "distinct_cell_operators": acquisition["minimum_pairwise_operator_distance"] >= 0.05,
        "no_raw_acquisition_replay_state": acquisition["raw_examples_retained"] == 0,
        "simultaneous_composition_quality": simultaneous["mean_mse"] <= 1e-4,
        "sequential_composition_quality": sequential["mean_mse"] <= 1e-4,
        "route_support_and_order_recovery": simultaneous["exact_route_sequence_accuracy"] >= 0.995 and sequential["exact_route_sequence_accuracy"] >= 0.995,
        "unseen_composition_generalization": simultaneous["all_compositions_unseen_during_training"] and sequential["all_compositions_unseen_during_training"] and main <= 0.05 * max(single, 1e-12),
        "composition_semantics_are_nontrivial": sequential["mean_true_order_effect_mse"] >= 1e-3 and simultaneous["mean_simultaneous_permutation_mse"] <= 1e-12 and main <= 0.05 * max(wrong, 1e-12),
        "dense_all_cells_control_fails": dense >= 1e-2 and main <= 0.05 * max(dense, 1e-12),
        "sparse_active_compute": simultaneous["cell_execution_fraction_vs_dense"] <= 0.30 and sequential["cell_execution_fraction_vs_dense"] <= 0.30 and simultaneous["maximum_active_cells"] <= cfg.maximum_active_cells and sequential["maximum_active_cells"] <= cfg.maximum_active_cells,
        "protected_composition_retention": mutation["safe_historical_composition_mse"] <= 1e-10 and mutation["safe_protected_change"] <= 1e-10,
        "protected_mutation_plasticity": mutation["safe_fit_error"] <= 1e-10 and mutation["safe_reported_relative_fit_error"] <= 1e-10,
        "protected_mutation_zero_replay": mutation["learner_replay_accesses"] == 0 and mutation["learner_raw_history_retained"] == 0,
        "unsafe_mutation_exposes_interference": mutation["unsafe_historical_composition_mse"] >= 1e-4 and mutation["unsafe_protected_change"] >= 1e-3,
        "cell_local_mutation_isolation": mutation["unrelated_cell_parameter_drift"] <= 1e-15 and mutation["route_key_drift"] <= 1e-15,
    }
    return {
        "seed": seed,
        "pass": all(gates.values()),
        "config": asdict(cfg),
        "structural_bridge": bridge,
        "acquisition": acquisition,
        "simultaneous": simultaneous,
        "sequential": sequential,
        "protected_mutation": mutation,
        "gates": gates,
    }


def model_level_smoke(seed: int = 501) -> dict[str, Any]:
    cfg = ComputationConfig(root_count=4, hidden_dim=8, acquisition_batches_per_cell=3, acquisition_batch_size=24, evaluation_cases_per_mode=16, evaluation_batch_size=12, minimum_active_cells=2, maximum_active_cells=3, certificate_history_cases=4, mutation_examples=3)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0xC1A004)
    rng = random.Random(int(seed) ^ 0x51A004)
    route_keys = F.normalize(torch.randn(cfg.root_count, 12, generator=generator, dtype=_DTYPE), dim=1)
    model = ComputationalCellModel(route_keys, cfg.hidden_dim, ridge=cfg.ridge)
    true_operators = torch.stack([_scaled_random_operator(cfg.hidden_dim, cfg.operator_spectral_norm, generator=generator) for _ in range(cfg.root_count)])
    acquisition = _acquire_operators(model, true_operators, cfg, generator=generator, rng=rng)
    simultaneous = _evaluate_mode(model, true_operators, cfg, mode="simultaneous", generator=generator, rng=rng)
    sequential = _evaluate_mode(model, true_operators, cfg, mode="sequential", generator=generator, rng=rng)
    mutation = _protected_mutation_probe(model, cfg, generator=generator, rng=rng)
    return {"seed": int(seed), "acquisition": acquisition, "simultaneous": simultaneous, "sequential": sequential, "protected_mutation": mutation}
