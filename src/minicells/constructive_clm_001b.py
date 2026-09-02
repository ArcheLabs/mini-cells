from __future__ import annotations

import itertools
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WorldConfig:
    factor_count: int = 6
    context_dim: int = 16
    effect_dim: int = 12
    context_common_rho: float = 0.32
    effect_common_rho: float = 0.22
    atom_jitter: float = 0.10
    discovery_cycles: int = 8
    samples_per_transaction: int = 24
    train_noise: float = 0.03
    eval_noise: float = 0.015
    heldout_pair_repeats: int = 8
    heldout_triple_repeats: int = 2


@dataclass(frozen=True)
class LearnerConfig:
    prototype_radius: float = 0.08
    minimum_clique_size: int = 3
    max_inference_support: int = 3
    support_complexity_penalty: float = 2e-4


@dataclass(frozen=True)
class Transaction:
    factors: tuple[int, ...]
    weights: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    phase: str


@dataclass(frozen=True)
class World:
    context_atoms: torch.Tensor
    effect_atoms: torch.Tensor
    train_pairs: tuple[tuple[int, int], ...]
    heldout_pair_types: tuple[tuple[int, int], ...]
    train_transactions: tuple[Transaction, ...]
    heldout_pairs: tuple[Transaction, ...]
    heldout_triples: tuple[Transaction, ...]


def _mean(xs: Iterable[float]) -> float:
    values = [float(x) for x in xs]
    return float(statistics.fmean(values)) if values else 0.0


def _correlated_atoms(
    count: int,
    dim: int,
    rho: float,
    jitter: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if dim <= count:
        raise ValueError("dimension must exceed factor count")
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    common = q[:, count]
    atoms: list[torch.Tensor] = []
    for index in range(count):
        value = math.sqrt(1.0 - rho) * q[:, index] + math.sqrt(rho) * common
        if jitter > 0.0:
            direction = torch.randn(dim, generator=generator, dtype=torch.float64)
            direction = direction - torch.dot(direction, value) * value
            direction = direction / max(float(torch.linalg.norm(direction).item()), 1e-12)
            value = value + jitter * direction
        atoms.append(F.normalize(value, dim=0))
    return torch.stack(atoms)


def _sample_mixture(
    context_atoms: torch.Tensor,
    effect_atoms: torch.Tensor,
    factors: Iterable[int],
    weights: torch.Tensor,
    *,
    samples: int,
    context_noise: float,
    effect_noise: float,
    generator: torch.Generator,
    phase: str,
) -> Transaction:
    support = tuple(sorted(int(i) for i in factors))
    if not support:
        raise ValueError("mixture support cannot be empty")
    if len(support) != int(weights.numel()):
        raise ValueError("support and weights disagree")
    w = weights.to(dtype=torch.float64)
    w = w / w.sum()
    base_x = (w[:, None] * context_atoms[list(support)]).sum(dim=0)
    base_y = (w[:, None] * effect_atoms[list(support)]).sum(dim=0)
    x = base_x.repeat(samples, 1)
    y = base_y.repeat(samples, 1)
    if context_noise > 0.0:
        x = x + context_noise * torch.randn(
            x.shape, generator=generator, dtype=x.dtype
        )
    if effect_noise > 0.0:
        y = y + effect_noise * torch.randn(
            y.shape, generator=generator, dtype=y.dtype
        )
    return Transaction(support, w, x, y, phase)


def _heldout_matching(count: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    if count % 2 != 0:
        raise ValueError("registered 001B world requires an even factor count")
    permutation = list(range(count))
    rng.shuffle(permutation)
    pairs = []
    for offset in range(0, count, 2):
        pairs.append(tuple(sorted((permutation[offset], permutation[offset + 1]))))
    return tuple(sorted(pairs))


def build_world(seed: int, cfg: WorldConfig = WorldConfig()) -> World:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rng = random.Random(int(seed) ^ 0xBADC0DE)
    context_atoms = _correlated_atoms(
        cfg.factor_count,
        cfg.context_dim,
        cfg.context_common_rho,
        cfg.atom_jitter,
        generator=generator,
    )
    effect_atoms = _correlated_atoms(
        cfg.factor_count,
        cfg.effect_dim,
        cfg.effect_common_rho,
        cfg.atom_jitter,
        generator=generator,
    )

    heldout = set(_heldout_matching(cfg.factor_count, rng))
    all_pairs = list(itertools.combinations(range(cfg.factor_count), 2))
    train_pairs = [pair for pair in all_pairs if tuple(sorted(pair)) not in heldout]

    train: list[Transaction] = []
    for cycle in range(cfg.discovery_cycles):
        order = list(train_pairs)
        rng.shuffle(order)
        for pair in order:
            train.append(
                _sample_mixture(
                    context_atoms,
                    effect_atoms,
                    pair,
                    torch.tensor([0.5, 0.5], dtype=torch.float64),
                    samples=cfg.samples_per_transaction,
                    context_noise=cfg.train_noise,
                    effect_noise=0.30 * cfg.train_noise,
                    generator=generator,
                    phase=f"discovery-{cycle}",
                )
            )

    heldout_pairs: list[Transaction] = []
    for pair in sorted(heldout):
        for _ in range(cfg.heldout_pair_repeats):
            first = float(torch.rand((), generator=generator).item()) * 0.50 + 0.25
            heldout_pairs.append(
                _sample_mixture(
                    context_atoms,
                    effect_atoms,
                    pair,
                    torch.tensor([first, 1.0 - first], dtype=torch.float64),
                    samples=64,
                    context_noise=cfg.eval_noise,
                    effect_noise=0.30 * cfg.eval_noise,
                    generator=generator,
                    phase="heldout-pair",
                )
            )

    heldout_triples: list[Transaction] = []
    for triple in itertools.combinations(range(cfg.factor_count), 3):
        for _ in range(cfg.heldout_triple_repeats):
            weights = torch.rand(3, generator=generator, dtype=torch.float64) + 0.35
            heldout_triples.append(
                _sample_mixture(
                    context_atoms,
                    effect_atoms,
                    triple,
                    weights,
                    samples=64,
                    context_noise=cfg.eval_noise,
                    effect_noise=0.30 * cfg.eval_noise,
                    generator=generator,
                    phase="heldout-triple",
                )
            )

    return World(
        context_atoms=context_atoms,
        effect_atoms=effect_atoms,
        train_pairs=tuple(sorted(tuple(sorted(pair)) for pair in train_pairs)),
        heldout_pair_types=tuple(sorted(heldout)),
        train_transactions=tuple(train),
        heldout_pairs=tuple(heldout_pairs),
        heldout_triples=tuple(heldout_triples),
    )


def _kmeans_two_threshold(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("at least two distances are required")
    vector = torch.tensor(values, dtype=torch.float64)
    first = float(vector.min().item())
    second = float(vector.max().item())
    for _ in range(100):
        left = torch.abs(vector - first) <= torch.abs(vector - second)
        if not bool(left.any()) or not bool((~left).any()):
            raise ValueError("distance clustering collapsed")
        new_first = float(vector[left].mean().item())
        new_second = float(vector[~left].mean().item())
        if abs(new_first - first) + abs(new_second - second) < 1e-12:
            first, second = new_first, new_second
            break
        first, second = new_first, new_second
    low, high = sorted((first, second))
    return low, high, 0.5 * (low + high)


def _maximal_cliques(adjacency: list[list[bool]], minimum_size: int) -> list[tuple[int, ...]]:
    count = len(adjacency)
    cliques: list[tuple[int, ...]] = []
    for size in range(minimum_size, count + 1):
        for candidate in itertools.combinations(range(count), size):
            if not all(adjacency[i][j] for i, j in itertools.combinations(candidate, 2)):
                continue
            extendable = False
            for node in range(count):
                if node in candidate:
                    continue
                if all(adjacency[node][member] for member in candidate):
                    extendable = True
                    break
            if not extendable:
                cliques.append(candidate)
    return cliques


def _best_permutation(score: torch.Tensor) -> tuple[int, ...] | None:
    rows, columns = score.shape
    if rows != columns or rows > 8:
        return None
    best_score = float("-inf")
    best: tuple[int, ...] | None = None
    for permutation in itertools.permutations(range(columns)):
        total = sum(float(score[row, permutation[row]].item()) for row in range(rows))
        if total > best_score:
            best_score = total
            best = tuple(int(i) for i in permutation)
    return best


class RelationalCellLearner:
    """Recover latent Cells from repeated pair superpositions without singleton labels.

    The learner receives only current x/y samples. It clusters stable mixture prototypes,
    infers the overlap graph from prototype geometry, recovers maximal star cliques of
    the line graph, then solves the learned incidence system for Cell keys/effects.
    Hidden factor identities and the hidden factor count are never passed to learner
    methods.
    """

    def __init__(self, context_dim: int, effect_dim: int, cfg: LearnerConfig = LearnerConfig()):
        self.context_dim = int(context_dim)
        self.effect_dim = int(effect_dim)
        self.cfg = cfg
        self.prototypes: list[torch.Tensor] = []
        self.prototype_counts: list[int] = []
        self.cells: torch.Tensor | None = None
        self.last_fit: dict[str, Any] | None = None

    def observe(self, x: torch.Tensor, y: torch.Tensor) -> None:
        joint = torch.cat([x.mean(dim=0), y.mean(dim=0)]).to(dtype=torch.float64)
        if not self.prototypes:
            self.prototypes.append(joint.clone())
            self.prototype_counts.append(1)
            return
        distances = torch.stack([torch.linalg.norm(joint - p) for p in self.prototypes])
        index = int(torch.argmin(distances).item())
        if float(distances[index].item()) < self.cfg.prototype_radius:
            self.prototype_counts[index] += 1
            count = self.prototype_counts[index]
            self.prototypes[index] = self.prototypes[index] + (
                joint - self.prototypes[index]
            ) / float(count)
            return
        self.prototypes.append(joint.clone())
        self.prototype_counts.append(1)

    def fit_cells(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "valid": False,
            "prototype_count": len(self.prototypes),
            "reason": None,
        }
        if len(self.prototypes) < self.cfg.minimum_clique_size:
            diagnostics["reason"] = "too_few_prototypes"
            self.last_fit = diagnostics
            return diagnostics

        prototype_matrix = torch.stack(self.prototypes)
        distances = [
            float(torch.linalg.norm(prototype_matrix[i] - prototype_matrix[j]).item())
            for i, j in itertools.combinations(range(len(self.prototypes)), 2)
        ]
        try:
            low, high, threshold = _kmeans_two_threshold(distances)
        except ValueError as exc:
            diagnostics["reason"] = str(exc)
            self.last_fit = diagnostics
            return diagnostics

        count = len(self.prototypes)
        adjacency = [[False] * count for _ in range(count)]
        for i, j in itertools.combinations(range(count), 2):
            if float(torch.linalg.norm(prototype_matrix[i] - prototype_matrix[j]).item()) < threshold:
                adjacency[i][j] = True
                adjacency[j][i] = True

        cliques = _maximal_cliques(adjacency, self.cfg.minimum_clique_size)
        if not cliques:
            diagnostics["reason"] = "no_maximal_cliques"
            self.last_fit = diagnostics
            return diagnostics
        largest = max(len(clique) for clique in cliques)
        stars = [clique for clique in cliques if len(clique) == largest]

        incidence = torch.zeros(count, len(stars), dtype=torch.float64)
        memberships: list[list[int]] = []
        for prototype_index in range(count):
            member_of = [
                star_index
                for star_index, clique in enumerate(stars)
                if prototype_index in clique
            ]
            memberships.append(member_of)
            if len(member_of) == 2:
                incidence[prototype_index, member_of] = 0.5

        all_two = all(len(member_of) == 2 for member_of in memberships)
        rank = int(torch.linalg.matrix_rank(incidence).item()) if incidence.numel() else 0
        diagnostics.update(
            {
                "distance_low_mean": low,
                "distance_high_mean": high,
                "distance_threshold": threshold,
                "distance_separation_ratio": high / max(low, 1e-12),
                "maximal_clique_sizes": [len(clique) for clique in cliques],
                "largest_clique_size": largest,
                "star_count": len(stars),
                "membership_all_two": all_two,
                "incidence_rank": rank,
            }
        )
        if not all_two:
            diagnostics["reason"] = "prototype_not_in_exactly_two_stars"
            self.last_fit = diagnostics
            return diagnostics
        if rank != len(stars):
            diagnostics["reason"] = "incidence_rank_deficient"
            self.last_fit = diagnostics
            return diagnostics

        cells = torch.linalg.lstsq(incidence, prototype_matrix).solution
        self.cells = cells.contiguous()
        diagnostics["valid"] = True
        diagnostics["reason"] = None
        diagnostics["cell_count"] = int(cells.shape[0])
        self.last_fit = diagnostics
        return diagnostics

    def keys_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cells is None:
            raise RuntimeError("latent Cells have not been recovered")
        return self.cells[:, : self.context_dim], self.cells[:, self.context_dim :]

    def infer(self, x: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor, float]:
        keys, _ = self.keys_values()
        query = x.to(dtype=torch.float64)
        best: tuple[float, float, tuple[int, ...], torch.Tensor] | None = None
        max_support = min(self.cfg.max_inference_support, int(keys.shape[0]))
        for size in range(1, max_support + 1):
            for subset in itertools.combinations(range(int(keys.shape[0])), size):
                selected = keys[list(subset)]
                if size == 1:
                    weights = torch.ones(1, dtype=torch.float64)
                else:
                    gram = selected @ selected.T
                    target = selected @ query
                    kkt = torch.zeros(size + 1, size + 1, dtype=torch.float64)
                    kkt[:size, :size] = gram + 1e-8 * torch.eye(size, dtype=torch.float64)
                    kkt[:size, size] = 1.0
                    kkt[size, :size] = 1.0
                    rhs = torch.cat([target, torch.ones(1, dtype=torch.float64)])
                    try:
                        solution = torch.linalg.solve(kkt, rhs)
                    except RuntimeError:
                        continue
                    weights = solution[:size]
                    if float(weights.min().item()) < -0.03:
                        continue
                    weights = torch.clamp(weights, min=0.0)
                    if float(weights.sum().item()) <= 1e-12:
                        continue
                    weights = weights / weights.sum()
                reconstruction = weights @ selected
                residual = float(torch.mean((reconstruction - query) ** 2).item())
                score = residual + self.cfg.support_complexity_penalty * float(size)
                row = (score, residual, tuple(int(i) for i in subset), weights)
                if best is None or score < best[0]:
                    best = row
        if best is None:
            raise RuntimeError("no feasible inference support")
        return best[2], best[3], best[1]

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], torch.Tensor, float]:
        subset, weights, residual = self.infer(x)
        _, values = self.keys_values()
        prediction = weights @ values[list(subset)]
        return prediction, subset, weights, residual


def _cell_similarity(first: torch.Tensor, second: torch.Tensor, context_dim: int) -> float:
    if first.shape != second.shape or int(first.shape[0]) > 8:
        return 0.0
    first_keys = F.normalize(first[:, :context_dim], dim=1)
    first_values = F.normalize(first[:, context_dim:], dim=1)
    second_keys = F.normalize(second[:, :context_dim], dim=1)
    second_values = F.normalize(second[:, context_dim:], dim=1)
    score = 0.5 * (first_keys @ second_keys.T + first_values @ second_values.T)
    permutation = _best_permutation(score)
    if permutation is None:
        return 0.0
    return _mean(score[row, permutation[row]].item() for row in range(len(permutation)))


def _posthoc_alignment(learner: RelationalCellLearner, world: World) -> dict[str, Any]:
    keys, values = learner.keys_values()
    normalized_keys = F.normalize(keys, dim=1)
    normalized_values = F.normalize(values, dim=1)
    key_cosine = normalized_keys @ F.normalize(world.context_atoms, dim=1).T
    value_cosine = normalized_values @ F.normalize(world.effect_atoms, dim=1).T
    joint = 0.5 * (key_cosine + value_cosine)
    permutation = _best_permutation(joint)
    if permutation is None:
        best_values, best_indices = joint.max(dim=1)
        mapping = tuple(int(i) for i in best_indices.tolist())
        return {
            "mapping": list(mapping),
            "covered_factors": len(set(mapping)),
            "mean_matched_key_cosine": _mean(
                key_cosine[row, mapping[row]].item() for row in range(len(mapping))
            ),
            "mean_matched_effect_cosine": _mean(
                value_cosine[row, mapping[row]].item() for row in range(len(mapping))
            ),
            "mean_joint_cosine": float(best_values.mean().item()),
            "exact_permutation": False,
        }
    return {
        "mapping": list(permutation),
        "covered_factors": len(set(permutation)),
        "mean_matched_key_cosine": _mean(
            key_cosine[row, permutation[row]].item() for row in range(len(permutation))
        ),
        "mean_matched_effect_cosine": _mean(
            value_cosine[row, permutation[row]].item() for row in range(len(permutation))
        ),
        "mean_joint_cosine": _mean(
            joint[row, permutation[row]].item() for row in range(len(permutation))
        ),
        "exact_permutation": True,
    }


def _evaluate_transactions(
    learner: RelationalCellLearner,
    transactions: Iterable[Transaction],
    mapping: list[int],
) -> dict[str, float]:
    mse: list[float] = []
    recall: list[float] = []
    exact: list[float] = []
    support_sizes: list[float] = []
    context_residuals: list[float] = []
    for transaction in transactions:
        query = transaction.x.mean(dim=0)
        target = transaction.y.mean(dim=0)
        prediction, subset, _, context_residual = learner.predict(query)
        mse.append(float(torch.mean((prediction - target) ** 2).item()))
        predicted = {int(mapping[index]) for index in subset}
        expected = set(transaction.factors)
        recall.append(len(predicted & expected) / max(len(expected), 1))
        exact.append(float(predicted == expected))
        support_sizes.append(float(len(subset)))
        context_residuals.append(float(context_residual))
    return {
        "mse": _mean(mse),
        "route_recall": _mean(recall),
        "exact_support_rate": _mean(exact),
        "mean_inferred_support": _mean(support_sizes),
        "mean_context_residual": _mean(context_residuals),
    }


def _nearest_prototype_baseline(
    learner: RelationalCellLearner,
    transactions: Iterable[Transaction],
    *,
    shuffled_effect: bool,
    seed: int,
) -> dict[str, float]:
    prototypes = torch.stack(learner.prototypes)
    context = prototypes[:, : learner.context_dim]
    effects = prototypes[:, learner.context_dim :].clone()
    if shuffled_effect:
        rng = random.Random(int(seed) ^ 0x51F7)
        permutation = list(range(len(effects)))
        for _ in range(128):
            rng.shuffle(permutation)
            if all(index != permutation[index] for index in range(len(permutation))):
                break
        if any(index == permutation[index] for index in range(len(permutation))):
            permutation = permutation[1:] + permutation[:1]
        effects = effects[permutation]

    mse: list[float] = []
    for transaction in transactions:
        query = transaction.x.mean(dim=0)
        target = transaction.y.mean(dim=0)
        index = int(torch.argmin(torch.linalg.norm(context - query, dim=1)).item())
        mse.append(float(torch.mean((effects[index] - target) ** 2).item()))
    return {"mse": _mean(mse)}


def evaluate_gates(result: dict[str, Any]) -> dict[str, bool]:
    fit = result["fit"]
    alignment = result["alignment"]
    pair_eval = result["heldout_pair"]
    triple_eval = result["heldout_triple"]
    return {
        "no_singleton_training": bool(result["no_singleton_training"]),
        "expected_prototype_count": int(result["prototype_count"]) == 12,
        "latent_cell_count": int(result["active_cells"]) == 6,
        "line_graph_structure": bool(fit["valid"])
        and int(fit.get("largest_clique_size", 0)) == 4
        and int(fit.get("star_count", 0)) == 6
        and bool(fit.get("membership_all_two", False)),
        "distance_class_separation": float(fit.get("distance_separation_ratio", 0.0)) >= 1.25,
        "coordinate_recovery": float(alignment["mean_matched_key_cosine"]) >= 0.98
        and float(alignment["mean_matched_effect_cosine"]) >= 0.98
        and int(alignment["covered_factors"]) == 6,
        "heldout_pair_quality": float(pair_eval["mse"]) <= 0.001,
        "heldout_triple_quality": float(triple_eval["mse"]) <= 0.001,
        "heldout_pair_addressability": float(pair_eval["route_recall"]) >= 0.95
        and float(pair_eval["exact_support_rate"]) >= 0.90,
        "heldout_triple_addressability": float(triple_eval["route_recall"]) >= 0.95
        and float(triple_eval["exact_support_rate"]) >= 0.90,
        "beats_transaction_memory": float(pair_eval["mse"])
        <= 0.10 * max(float(result["transaction_memory_pair_mse"]), 1e-12)
        and float(triple_eval["mse"])
        <= 0.10 * max(float(result["transaction_memory_triple_mse"]), 1e-12),
        "beats_shuffled_effect_address": float(pair_eval["mse"])
        <= 0.10 * max(float(result["shuffled_effect_pair_mse"]), 1e-12),
        "late_coordinate_stability": float(result["late_checkpoint_similarity_min"]) >= 0.995,
        "state_compression": float(result["transaction_to_cell_compression"]) >= 12.0
        and float(result["prototype_to_cell_compression"]) >= 1.5,
    }


def run_seed(
    seed: int,
    world_cfg: WorldConfig = WorldConfig(),
    learner_cfg: LearnerConfig = LearnerConfig(),
) -> dict[str, Any]:
    world = build_world(seed, world_cfg)
    learner = RelationalCellLearner(
        world_cfg.context_dim,
        world_cfg.effect_dim,
        learner_cfg,
    )

    checkpoint_cells: list[torch.Tensor | None] = []
    checkpoint_counts: list[int] = []
    per_cycle = len(world.train_pairs)
    for step, transaction in enumerate(world.train_transactions):
        learner.observe(transaction.x, transaction.y)
        if (step + 1) % per_cycle == 0:
            fit = learner.fit_cells()
            if fit["valid"] and learner.cells is not None:
                checkpoint_cells.append(learner.cells.clone())
                checkpoint_counts.append(int(learner.cells.shape[0]))
            else:
                checkpoint_cells.append(None)
                checkpoint_counts.append(0)

    fit = learner.fit_cells()
    if not fit["valid"] or learner.cells is None:
        result = {
            "seed": int(seed),
            "world": asdict(world_cfg),
            "learner": asdict(learner_cfg),
            "transactions": len(world.train_transactions),
            "prototype_count": len(learner.prototypes),
            "active_cells": 0,
            "no_singleton_training": all(
                len(transaction.factors) >= 2 for transaction in world.train_transactions
            ),
            "fit": fit,
            "checkpoint_cell_counts": checkpoint_counts,
            "alignment": {
                "mapping": [],
                "covered_factors": 0,
                "mean_matched_key_cosine": 0.0,
                "mean_matched_effect_cosine": 0.0,
                "mean_joint_cosine": 0.0,
                "exact_permutation": False,
            },
            "heldout_pair": {"mse": float("inf"), "route_recall": 0.0, "exact_support_rate": 0.0},
            "heldout_triple": {"mse": float("inf"), "route_recall": 0.0, "exact_support_rate": 0.0},
            "transaction_memory_pair_mse": 0.0,
            "transaction_memory_triple_mse": 0.0,
            "shuffled_effect_pair_mse": 0.0,
            "late_checkpoint_similarity_min": 0.0,
            "transaction_to_cell_compression": 0.0,
            "prototype_to_cell_compression": 0.0,
        }
        result["gates"] = evaluate_gates(result)
        result["pass"] = False
        return result

    alignment = _posthoc_alignment(learner, world)
    pair_eval = _evaluate_transactions(learner, world.heldout_pairs, alignment["mapping"])
    triple_eval = _evaluate_transactions(learner, world.heldout_triples, alignment["mapping"])
    pair_baseline = _nearest_prototype_baseline(
        learner, world.heldout_pairs, shuffled_effect=False, seed=seed
    )
    triple_baseline = _nearest_prototype_baseline(
        learner, world.heldout_triples, shuffled_effect=False, seed=seed
    )
    shuffled_pair = _nearest_prototype_baseline(
        learner, world.heldout_pairs, shuffled_effect=True, seed=seed
    )

    final_cells = learner.cells.clone()
    late_cells = checkpoint_cells[-4:]
    similarities = [
        _cell_similarity(cells, final_cells, world_cfg.context_dim)
        if cells is not None
        else 0.0
        for cells in late_cells
    ]
    result = {
        "seed": int(seed),
        "world": asdict(world_cfg),
        "learner": asdict(learner_cfg),
        "transactions": len(world.train_transactions),
        "distinct_train_pair_types": len(world.train_pairs),
        "heldout_pair_types": [list(pair) for pair in world.heldout_pair_types],
        "prototype_count": len(learner.prototypes),
        "active_cells": int(final_cells.shape[0]),
        "no_singleton_training": all(
            len(transaction.factors) >= 2 for transaction in world.train_transactions
        ),
        "minimum_training_support": min(
            len(transaction.factors) for transaction in world.train_transactions
        ),
        "maximum_training_support": max(
            len(transaction.factors) for transaction in world.train_transactions
        ),
        "fit": fit,
        "checkpoint_cell_counts": checkpoint_counts,
        "late_checkpoint_similarity": similarities,
        "late_checkpoint_similarity_min": min(similarities) if similarities else 0.0,
        "alignment": alignment,
        "heldout_pair": pair_eval,
        "heldout_triple": triple_eval,
        "transaction_memory_pair_mse": pair_baseline["mse"],
        "transaction_memory_triple_mse": triple_baseline["mse"],
        "shuffled_effect_pair_mse": shuffled_pair["mse"],
        "transaction_to_cell_compression": len(world.train_transactions)
        / max(int(final_cells.shape[0]), 1),
        "prototype_to_cell_compression": len(learner.prototypes)
        / max(int(final_cells.shape[0]), 1),
    }
    result["gates"] = evaluate_gates(result)
    result["pass"] = all(result["gates"].values())
    return result
