from __future__ import annotations

import itertools
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from minicells.constructive_clm_001b import LearnerConfig as BootstrapLearnerConfig
from minicells.constructive_clm_001b import RelationalCellLearner


@dataclass(frozen=True)
class WorldConfig:
    max_transactions: int = 4096
    initial_factors: int = 6
    max_factors: int = 30
    context_dim: int = 48
    effect_dim: int = 40
    context_common_rho: float = 0.18
    effect_common_rho: float = 0.12
    atom_jitter: float = 0.05
    bootstrap_cycles: int = 6
    introduction_repeats: int = 4
    growth_alpha: float = 0.60
    stabilization_tail: int = 128
    samples_per_transaction: int = 16
    train_noise: float = 0.02
    eval_noise: float = 0.01
    checkpoints: tuple[int, ...] = (256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class GrowthConfig:
    residual_mse_threshold: float = 3e-4
    inference_support: int = 3
    support_complexity_penalty: float = 2e-5
    proposal_anchor_count: int = 6
    proposal_min_anchor_weight: float = 0.15
    proposal_max_anchor_weight: float = 0.85
    proposal_key_norm_min: float = 0.70
    proposal_key_norm_max: float = 1.30
    proposal_value_norm_min: float = 0.65
    proposal_value_norm_max: float = 1.35
    candidate_similarity_threshold: float = 0.985
    candidate_confirmations: int = 2
    candidate_distinct_anchors: int = 2
    candidate_expiry: int = 10
    existing_key_similarity_reject: float = 0.95


@dataclass(frozen=True)
class Transaction:
    factors: tuple[int, ...]
    weights: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    phase: str


@dataclass
class CandidateCell:
    key: torch.Tensor
    value: torch.Tensor
    count: int
    anchors: set[int]
    born_at: int
    last_seen: int


@dataclass(frozen=True)
class World:
    context_atoms: torch.Tensor
    effect_atoms: torch.Tensor
    bootstrap_pairs: tuple[tuple[int, int], ...]
    introduction_starts: dict[int, int]


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def _correlated_atoms(
    count: int,
    dim: int,
    rho: float,
    jitter: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if dim <= count:
        raise ValueError("registered CLM-002 world requires dimension > factor count")
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


def _bootstrap_pairs(initial_factors: int) -> tuple[tuple[int, int], ...]:
    if initial_factors != 6:
        raise ValueError("registered CLM-002 bootstrap reuses the six-factor 001B scaffold")
    heldout = {(0, 1), (2, 3), (4, 5)}
    return tuple(
        pair
        for pair in itertools.combinations(range(initial_factors), 2)
        if tuple(sorted(pair)) not in heldout
    )


def _introduction_schedule(cfg: WorldConfig) -> dict[int, int]:
    bootstrap_transactions = len(_bootstrap_pairs(cfg.initial_factors)) * cfg.bootstrap_cycles
    span = cfg.max_transactions - bootstrap_transactions - cfg.stabilization_tail
    if span <= 0:
        raise ValueError("stream is too short for the registered growth schedule")
    extra = cfg.max_factors - cfg.initial_factors
    if extra <= 0:
        raise ValueError("max_factors must exceed initial_factors")
    starts: dict[int, int] = {}
    previous = -1
    for offset in range(extra):
        fraction = float(offset + 1) / float(extra)
        step = bootstrap_transactions + int((fraction ** (1.0 / cfg.growth_alpha)) * span)
        if step <= previous + cfg.introduction_repeats:
            raise ValueError("registered factor introductions overlap")
        starts[step] = cfg.initial_factors + offset
        previous = step
    return starts


def build_world(seed: int, cfg: WorldConfig = WorldConfig()) -> World:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    context_atoms = _correlated_atoms(
        cfg.max_factors,
        cfg.context_dim,
        cfg.context_common_rho,
        cfg.atom_jitter,
        generator=generator,
    )
    effect_atoms = _correlated_atoms(
        cfg.max_factors,
        cfg.effect_dim,
        cfg.effect_common_rho,
        cfg.atom_jitter,
        generator=generator,
    )
    return World(
        context_atoms=context_atoms,
        effect_atoms=effect_atoms,
        bootstrap_pairs=_bootstrap_pairs(cfg.initial_factors),
        introduction_starts=_introduction_schedule(cfg),
    )


def _sample_mixture(
    world: World,
    factors: Iterable[int],
    weights: Iterable[float],
    *,
    samples: int,
    context_noise: float,
    effect_noise: float,
    generator: torch.Generator,
    phase: str,
) -> Transaction:
    support = tuple(int(index) for index in factors)
    w = torch.tensor(list(weights), dtype=torch.float64)
    if len(support) != int(w.numel()) or not support:
        raise ValueError("support and weights must be non-empty and aligned")
    w = w / w.sum()
    base_x = (w[:, None] * world.context_atoms[list(support)]).sum(dim=0)
    base_y = (w[:, None] * world.effect_atoms[list(support)]).sum(dim=0)
    x = base_x.repeat(samples, 1)
    y = base_y.repeat(samples, 1)
    if context_noise > 0.0:
        x = x + context_noise * torch.randn(x.shape, generator=generator, dtype=x.dtype)
    if effect_noise > 0.0:
        y = y + effect_noise * torch.randn(y.shape, generator=generator, dtype=y.dtype)
    return Transaction(support, w, x, y, phase)


def _joint_similarity(
    first_key: torch.Tensor,
    first_value: torch.Tensor,
    second_key: torch.Tensor,
    second_value: torch.Tensor,
) -> float:
    key = float(torch.dot(F.normalize(first_key, dim=0), F.normalize(second_key, dim=0)).item())
    value = float(
        torch.dot(F.normalize(first_value, dim=0), F.normalize(second_value, dim=0)).item()
    )
    return 0.5 * (key + value)


class StreamingGrowthLearner:
    """Track a growing latent vocabulary without a hard Cell cap.

    Existing Cells explain normal pair/triple reuse transactions from x only. If the
    registered context residual is too large, the learner generates unlabeled
    one-missing-factor proposals against several possible known anchors. A proposal is
    committed only when the same latent residual is recovered through multiple distinct
    anchors. Hidden factor IDs and novelty labels are never inputs to this class.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        born_at: int,
        cfg: GrowthConfig = GrowthConfig(),
    ) -> None:
        if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must be aligned matrices")
        self.cfg = cfg
        self.keys = F.normalize(keys.to(dtype=torch.float64).clone(), dim=1)
        self.values = values.to(dtype=torch.float64).clone()
        self.birth_steps = [int(born_at)] * int(keys.shape[0])
        self.last_used_steps = [int(born_at)] * int(keys.shape[0])
        self.usage_counts = [0] * int(keys.shape[0])
        self.candidates: list[CandidateCell] = []
        self.spawn_steps: list[int] = []
        self.action_counts = {"reuse": 0, "candidate": 0, "spawn": 0}
        self.action_log: list[dict[str, Any]] = []

    @property
    def cell_count(self) -> int:
        return int(self.keys.shape[0])

    def _infer(self, query: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor, float]:
        query = query.to(dtype=torch.float64)
        norms = torch.linalg.norm(self.keys, dim=1) * max(float(torch.linalg.norm(query).item()), 1e-12)
        similarities = (self.keys @ query) / norms
        order = torch.argsort(similarities, descending=True)[: min(self.cfg.inference_support, self.cell_count)]
        ordered = [int(index) for index in order.tolist()]
        best: tuple[float, float, tuple[int, ...], torch.Tensor] | None = None
        for size in range(1, len(ordered) + 1):
            subset = tuple(ordered[:size])
            selected = self.keys[list(subset)]
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
            row = (score, residual, subset, weights)
            if best is None or score < best[0]:
                best = row
        if best is None:
            raise RuntimeError("no feasible Cell support")
        return best[2], best[3], best[1]

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], torch.Tensor, float]:
        query = x.mean(dim=0) if x.ndim == 2 else x
        subset, weights, residual = self._infer(query)
        prediction = weights @ self.values[list(subset)]
        return prediction, subset, weights, residual

    def _proposals(self, query: torch.Tensor, target: torch.Tensor) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        norms = torch.linalg.norm(self.keys, dim=1) * max(float(torch.linalg.norm(query).item()), 1e-12)
        similarities = (self.keys @ query) / norms
        anchors = torch.topk(
            similarities,
            k=min(self.cfg.proposal_anchor_count, self.cell_count),
        ).indices.tolist()
        query_norm_sq = float(torch.dot(query, query).item())
        rows: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for anchor in anchors:
            anchor = int(anchor)
            key = self.keys[anchor]
            value = self.values[anchor]
            denominator = 2.0 * (1.0 - float(torch.dot(query, key).item()))
            if denominator <= 1e-8:
                continue
            anchor_weight = (1.0 - query_norm_sq) / denominator
            if not (
                self.cfg.proposal_min_anchor_weight
                <= anchor_weight
                <= self.cfg.proposal_max_anchor_weight
            ):
                continue
            missing_weight = 1.0 - anchor_weight
            raw_key = (query - anchor_weight * key) / missing_weight
            raw_norm = float(torch.linalg.norm(raw_key).item())
            if not self.cfg.proposal_key_norm_min <= raw_norm <= self.cfg.proposal_key_norm_max:
                continue
            candidate_key = raw_key / raw_norm
            effective_weight = missing_weight * raw_norm
            candidate_value = (target - anchor_weight * value) / effective_weight
            value_norm = float(torch.linalg.norm(candidate_value).item())
            if not self.cfg.proposal_value_norm_min <= value_norm <= self.cfg.proposal_value_norm_max:
                continue
            rows.append((anchor, candidate_key, candidate_value))
        return rows

    def _register_proposals(
        self,
        proposals: list[tuple[int, torch.Tensor, torch.Tensor]],
        step: int,
    ) -> None:
        self.candidates = [
            candidate
            for candidate in self.candidates
            if step - candidate.last_seen <= self.cfg.candidate_expiry
        ]
        for anchor, key, value in proposals:
            scores = [
                _joint_similarity(candidate.key, candidate.value, key, value)
                for candidate in self.candidates
            ]
            if scores and max(scores) >= self.cfg.candidate_similarity_threshold:
                index = max(range(len(scores)), key=lambda row: scores[row])
                candidate = self.candidates[index]
                candidate.count += 1
                candidate.anchors.add(int(anchor))
                candidate.last_seen = int(step)
                rate = 1.0 / float(candidate.count)
                candidate.key = F.normalize((1.0 - rate) * candidate.key + rate * key, dim=0)
                candidate.value = (1.0 - rate) * candidate.value + rate * value
            else:
                self.candidates.append(
                    CandidateCell(
                        key=key.clone(),
                        value=value.clone(),
                        count=1,
                        anchors={int(anchor)},
                        born_at=int(step),
                        last_seen=int(step),
                    )
                )

    def _commit_candidate(self, step: int) -> bool:
        eligible: list[tuple[int, int]] = []
        for index, candidate in enumerate(self.candidates):
            if candidate.count < self.cfg.candidate_confirmations:
                continue
            if len(candidate.anchors) < self.cfg.candidate_distinct_anchors:
                continue
            key_similarity = torch.max(self.keys @ F.normalize(candidate.key, dim=0))
            if float(key_similarity.item()) >= self.cfg.existing_key_similarity_reject:
                continue
            eligible.append((candidate.count, index))
        if not eligible:
            return False
        _, index = max(eligible)
        candidate = self.candidates.pop(index)
        committed_key = F.normalize(candidate.key, dim=0)
        self.keys = torch.cat([self.keys, committed_key[None, :]], dim=0)
        self.values = torch.cat([self.values, candidate.value[None, :]], dim=0)
        self.birth_steps.append(int(step))
        self.last_used_steps.append(int(step))
        self.usage_counts.append(1)
        self.spawn_steps.append(int(step))
        self.candidates = [
            row
            for row in self.candidates
            if _joint_similarity(row.key, row.value, committed_key, candidate.value) < 0.95
        ]
        return True

    def observe(self, x: torch.Tensor, y: torch.Tensor, step: int) -> dict[str, Any]:
        query = x.mean(dim=0).to(dtype=torch.float64)
        target = y.mean(dim=0).to(dtype=torch.float64)
        subset, _, residual = self._infer(query)
        if residual <= self.cfg.residual_mse_threshold:
            for index in subset:
                self.usage_counts[index] += 1
                self.last_used_steps[index] = int(step)
            action = "reuse"
        else:
            self._register_proposals(self._proposals(query, target), int(step))
            if self._commit_candidate(int(step)):
                action = "spawn"
            else:
                action = "candidate"
        self.action_counts[action] += 1
        row = {
            "step": int(step),
            "action": action,
            "context_residual_mse": float(residual),
            "cells": self.cell_count,
            "candidate_clusters": len(self.candidates),
        }
        self.action_log.append(row)
        return row


def _bootstrap(
    world: World,
    cfg: WorldConfig,
    *,
    generator: torch.Generator,
    rng: random.Random,
) -> tuple[StreamingGrowthLearner, dict[str, Any], int]:
    bootstrap = RelationalCellLearner(
        cfg.context_dim,
        cfg.effect_dim,
        BootstrapLearnerConfig(
            prototype_radius=0.08,
            minimum_clique_size=3,
            max_inference_support=3,
            support_complexity_penalty=2e-4,
        ),
    )
    step = 0
    for cycle in range(cfg.bootstrap_cycles):
        order = list(world.bootstrap_pairs)
        rng.shuffle(order)
        for pair in order:
            transaction = _sample_mixture(
                world,
                pair,
                (0.5, 0.5),
                samples=cfg.samples_per_transaction,
                context_noise=cfg.train_noise,
                effect_noise=0.30 * cfg.train_noise,
                generator=generator,
                phase=f"bootstrap-{cycle}",
            )
            bootstrap.observe(transaction.x, transaction.y)
            step += 1
    fit = bootstrap.fit_cells()
    if not fit.get("valid") or bootstrap.cells is None:
        raise RuntimeError(f"001B bootstrap failed: {fit}")
    keys, values = bootstrap.keys_values()
    learner = StreamingGrowthLearner(keys, values, born_at=step - 1)
    return learner, fit, step


def _posthoc_alignment(
    learner: StreamingGrowthLearner,
    world: World,
    true_factor_count: int,
) -> dict[str, Any]:
    normalized_keys = F.normalize(learner.keys, dim=1)
    normalized_values = F.normalize(learner.values, dim=1)
    key_cosine = normalized_keys @ world.context_atoms[:true_factor_count].T
    value_cosine = normalized_values @ world.effect_atoms[:true_factor_count].T
    joint = 0.5 * (key_cosine + value_cosine)
    best_values, best_indices = joint.max(dim=1)
    mapping = [int(index) for index in best_indices.tolist()]
    key_values = [float(key_cosine[row, mapping[row]].item()) for row in range(len(mapping))]
    effect_values = [float(value_cosine[row, mapping[row]].item()) for row in range(len(mapping))]
    return {
        "mapping": mapping,
        "covered_factors": len(set(mapping)),
        "duplicate_assignments": len(mapping) - len(set(mapping)),
        "mean_joint_cosine": float(best_values.mean().item()),
        "mean_matched_key_cosine": _mean(key_values),
        "mean_matched_effect_cosine": _mean(effect_values),
        "min_matched_key_cosine": min(key_values) if key_values else 0.0,
        "min_matched_effect_cosine": min(effect_values) if effect_values else 0.0,
    }


def _evaluate(
    learner: StreamingGrowthLearner,
    world: World,
    true_factor_count: int,
    *,
    seed: int,
    pair_examples: int = 64,
    triple_examples: int = 32,
    restrict_to_initial: bool = False,
    eval_noise: float = 0.01,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rng = random.Random(int(seed) ^ 0xE11A)
    alignment = _posthoc_alignment(learner, world, true_factor_count)
    mapping = alignment["mapping"]
    limit = min(true_factor_count, 6) if restrict_to_initial else true_factor_count

    def run_examples(count: int, support_size: int) -> dict[str, float]:
        mse: list[float] = []
        recall: list[float] = []
        exact: list[float] = []
        residuals: list[float] = []
        for _ in range(count):
            support = tuple(rng.sample(range(limit), support_size))
            weights = [0.25 + rng.random() for _ in support]
            transaction = _sample_mixture(
                world,
                support,
                weights,
                samples=64,
                context_noise=eval_noise,
                effect_noise=0.30 * eval_noise,
                generator=generator,
                phase="eval",
            )
            prediction, subset, _, residual = learner.predict(transaction.x)
            target = transaction.y.mean(dim=0)
            mse.append(float(torch.mean((prediction - target) ** 2).item()))
            predicted_factors = {mapping[index] for index in subset}
            expected = set(support)
            recall.append(len(predicted_factors & expected) / float(len(expected)))
            exact.append(float(predicted_factors == expected))
            residuals.append(float(residual))
        return {
            "mse": _mean(mse),
            "route_recall": _mean(recall),
            "exact_support_rate": _mean(exact),
            "context_residual_mse": _mean(residuals),
        }

    return {
        "alignment": alignment,
        "pair": run_examples(pair_examples, 2),
        "triple": run_examples(triple_examples, 3),
    }


def _log_slope(xs: list[float], ys: list[float]) -> float:
    rows = [(math.log(float(x)), math.log(float(y))) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(rows) < 2:
        return 0.0
    mean_x = _mean(row[0] for row in rows)
    mean_y = _mean(row[1] for row in rows)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in rows)
    denominator = sum((x - mean_x) ** 2 for x, _ in rows)
    return float(numerator / max(denominator, 1e-12))


def _window_metrics(
    action_log: list[dict[str, Any]],
    boundaries: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        actions = [row for row in action_log if start <= int(row["step"]) < end]
        counts = {
            action: sum(row["action"] == action for row in actions)
            for action in ("reuse", "candidate", "spawn")
        }
        width = max(end - start, 1)
        rows.append(
            {
                "start": int(start),
                "end": int(end),
                "transactions": len(actions),
                "spawn_count": int(counts["spawn"]),
                "spawn_rate": float(counts["spawn"] / width),
                "reuse_rate": float(counts["reuse"] / max(len(actions), 1)),
                "candidate_rate": float(counts["candidate"] / max(len(actions), 1)),
            }
        )
    return rows


def evaluate_gates(result: dict[str, Any]) -> dict[str, bool]:
    checkpoints = result["checkpoints"]
    final = checkpoints[-1]
    windows = result["window_metrics"]
    spawn_rates = [float(row["spawn_rate"]) for row in windows]
    return {
        "bootstrap_reused_component_valid": bool(result["bootstrap_fit"]["valid"])
        and int(result["bootstrap_fit"]["cell_count"]) == 6,
        "no_hard_cell_cap": result["hard_cell_cap"] is None,
        "latent_vocabulary_tracking": max(abs(int(row["cells"]) - int(row["true_factors"])) for row in checkpoints) <= 1,
        "final_no_false_or_missed_growth": int(final["cells"]) == int(final["true_factors"]) == 30,
        "sublinear_like_finite_horizon_exponent": 0.40 <= float(result["cell_growth_exponent"]) <= 0.80,
        "growth_exponent_tracks_oracle": abs(float(result["cell_growth_exponent"]) - float(result["latent_growth_exponent"])) <= 0.05,
        "beats_transaction_memory_growth": float(result["cell_growth_exponent"]) <= 0.85 * float(result["transaction_memory_exponent"]),
        "state_ratio_declines": float(final["cell_to_transaction_ratio"]) <= 0.01
        and float(final["cell_to_transaction_ratio"]) <= 0.30 * float(checkpoints[0]["cell_to_transaction_ratio"]),
        "spawn_rate_declines": all(later <= 1.15 * earlier for earlier, later in zip(spawn_rates[:-1], spawn_rates[1:])),
        "late_spawn_rate_low": float(windows[-1]["spawn_rate"]) <= 0.006,
        "late_reuse_high": float(windows[-1]["reuse_rate"]) >= 0.985,
        "growth_remains_enabled_late": int(result["last_spawn_step"]) >= int(0.90 * result["world"]["max_transactions"]),
        "coordinate_quality": float(final["evaluation"]["alignment"]["mean_matched_key_cosine"]) >= 0.985
        and float(final["evaluation"]["alignment"]["mean_matched_effect_cosine"]) >= 0.995
        and int(final["evaluation"]["alignment"]["duplicate_assignments"]) == 0,
        "composition_quality": float(final["evaluation"]["pair"]["mse"]) <= 5e-4
        and float(final["evaluation"]["triple"]["mse"]) <= 5e-4
        and float(final["evaluation"]["pair"]["route_recall"]) >= 0.98
        and float(final["evaluation"]["triple"]["route_recall"]) >= 0.98,
        "early_factor_retention": float(result["early_retention"]["pair"]["mse"]) <= 5e-4
        and float(result["early_retention"]["triple"]["mse"]) <= 5e-4
        and float(result["early_retention"]["pair"]["route_recall"]) >= 0.98
        and float(result["early_retention"]["triple"]["route_recall"]) >= 0.98,
        "state_compression": float(result["transaction_to_cell_compression"]) >= 100.0,
    }


def run_seed(
    seed: int,
    world_cfg: WorldConfig = WorldConfig(),
    growth_cfg: GrowthConfig = GrowthConfig(),
) -> dict[str, Any]:
    if tuple(world_cfg.checkpoints)[-1] != world_cfg.max_transactions:
        raise ValueError("final checkpoint must equal max_transactions")
    world = build_world(seed, world_cfg)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0xC1A002)
    rng = random.Random(int(seed) ^ 0x51A002)

    bootstrap_learner = RelationalCellLearner(
        world_cfg.context_dim,
        world_cfg.effect_dim,
        BootstrapLearnerConfig(
            prototype_radius=0.08,
            minimum_clique_size=3,
            max_inference_support=3,
            support_complexity_penalty=2e-4,
        ),
    )
    step = 0
    for cycle in range(world_cfg.bootstrap_cycles):
        order = list(world.bootstrap_pairs)
        rng.shuffle(order)
        for pair in order:
            transaction = _sample_mixture(
                world,
                pair,
                (0.5, 0.5),
                samples=world_cfg.samples_per_transaction,
                context_noise=world_cfg.train_noise,
                effect_noise=0.30 * world_cfg.train_noise,
                generator=generator,
                phase=f"bootstrap-{cycle}",
            )
            bootstrap_learner.observe(transaction.x, transaction.y)
            step += 1
    bootstrap_fit = bootstrap_learner.fit_cells()
    if not bootstrap_fit.get("valid") or bootstrap_learner.cells is None:
        result = {
            "seed": int(seed),
            "world": asdict(world_cfg),
            "growth": asdict(growth_cfg),
            "hard_cell_cap": None,
            "bootstrap_fit": bootstrap_fit,
            "checkpoints": [],
            "window_metrics": [],
            "cell_growth_exponent": 0.0,
            "latent_growth_exponent": 0.0,
            "transaction_memory_exponent": 1.0,
            "last_spawn_step": -1,
            "early_retention": {
                "pair": {"mse": 1.0, "route_recall": 0.0},
                "triple": {"mse": 1.0, "route_recall": 0.0},
            },
            "transaction_to_cell_compression": 0.0,
            "pass": False,
        }
        result["gates"] = {"bootstrap_reused_component_valid": False}
        return result

    bootstrap_keys, bootstrap_values = bootstrap_learner.keys_values()
    learner = StreamingGrowthLearner(
        bootstrap_keys,
        bootstrap_values,
        born_at=step - 1,
        cfg=growth_cfg,
    )
    active_true_factors = world_cfg.initial_factors
    introduction_queue: list[tuple[int, int]] = []
    checkpoints: list[dict[str, Any]] = []
    checkpoint_set = set(int(row) for row in world_cfg.checkpoints)

    while step < world_cfg.max_transactions:
        if step in world.introduction_starts:
            novel_factor = int(world.introduction_starts[step])
            anchors = rng.sample(range(novel_factor), world_cfg.introduction_repeats)
            introduction_queue = [(novel_factor, int(anchor)) for anchor in anchors]

        if introduction_queue:
            novel_factor, anchor = introduction_queue.pop(0)
            anchor_weight = 0.30 + 0.40 * rng.random()
            transaction = _sample_mixture(
                world,
                (anchor, novel_factor),
                (anchor_weight, 1.0 - anchor_weight),
                samples=world_cfg.samples_per_transaction,
                context_noise=world_cfg.train_noise,
                effect_noise=0.30 * world_cfg.train_noise,
                generator=generator,
                phase="introduction",
            )
            if not introduction_queue:
                active_true_factors = max(active_true_factors, novel_factor + 1)
        else:
            support_size = 2 if rng.random() < 0.65 else 3
            support = tuple(rng.sample(range(active_true_factors), support_size))
            weights = [0.25 + rng.random() for _ in support]
            transaction = _sample_mixture(
                world,
                support,
                weights,
                samples=world_cfg.samples_per_transaction,
                context_noise=world_cfg.train_noise,
                effect_noise=0.30 * world_cfg.train_noise,
                generator=generator,
                phase="reuse",
            )

        learner.observe(transaction.x, transaction.y, step)
        step += 1

        if step in checkpoint_set:
            evaluation = _evaluate(
                learner,
                world,
                active_true_factors,
                seed=int(seed) ^ (step * 1009),
                eval_noise=world_cfg.eval_noise,
            )
            checkpoints.append(
                {
                    "transactions": int(step),
                    "true_factors": int(active_true_factors),
                    "cells": int(learner.cell_count),
                    "tracking_error": int(learner.cell_count - active_true_factors),
                    "cell_to_transaction_ratio": float(learner.cell_count / step),
                    "true_factor_to_transaction_ratio": float(active_true_factors / step),
                    "spawn_count": len(learner.spawn_steps),
                    "evaluation": evaluation,
                }
            )

    bootstrap_transactions = len(world.bootstrap_pairs) * world_cfg.bootstrap_cycles
    effective_transactions = [
        int(row["transactions"]) - bootstrap_transactions for row in checkpoints
    ]
    learned_growth = [int(row["cells"]) - world_cfg.initial_factors for row in checkpoints]
    latent_growth = [int(row["true_factors"]) - world_cfg.initial_factors for row in checkpoints]
    cell_growth_exponent = _log_slope(effective_transactions, learned_growth)
    latent_growth_exponent = _log_slope(effective_transactions, latent_growth)
    boundaries = (bootstrap_transactions, 512, 1024, 2048, world_cfg.max_transactions)
    window_metrics = _window_metrics(learner.action_log, boundaries)
    early_retention = _evaluate(
        learner,
        world,
        active_true_factors,
        seed=int(seed) ^ 0xEA21,
        pair_examples=64,
        triple_examples=32,
        restrict_to_initial=True,
        eval_noise=world_cfg.eval_noise,
    )

    result = {
        "seed": int(seed),
        "world": asdict(world_cfg),
        "growth": asdict(growth_cfg),
        "hard_cell_cap": None,
        "bootstrap_fit": bootstrap_fit,
        "bootstrap_transactions": int(bootstrap_transactions),
        "introduction_starts": [int(step) for step in sorted(world.introduction_starts)],
        "checkpoints": checkpoints,
        "window_metrics": window_metrics,
        "cell_growth_exponent": float(cell_growth_exponent),
        "latent_growth_exponent": float(latent_growth_exponent),
        "transaction_memory_exponent": 1.0,
        "spawn_steps": [int(row) for row in learner.spawn_steps],
        "last_spawn_step": int(learner.spawn_steps[-1]) if learner.spawn_steps else -1,
        "action_counts": dict(learner.action_counts),
        "usage_counts": [int(row) for row in learner.usage_counts],
        "birth_steps": [int(row) for row in learner.birth_steps],
        "cell_lifetimes": [int(world_cfg.max_transactions - row) for row in learner.birth_steps],
        "median_cell_lifetime": float(statistics.median(world_cfg.max_transactions - row for row in learner.birth_steps)),
        "early_retention": early_retention,
        "transaction_to_cell_compression": float(world_cfg.max_transactions / learner.cell_count),
    }
    result["gates"] = evaluate_gates(result)
    result["pass"] = all(result["gates"].values())
    return result
