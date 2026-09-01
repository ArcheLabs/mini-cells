from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

DEVELOPMENT_SEEDS = (1001, 1002, 1003)
FORMAL_SEEDS = (90111, 90112, 90113)


@dataclass(frozen=True)
class WorldConfig:
    factor_count: int = 6
    context_dim: int = 16
    effect_dim: int = 12
    train_noise: float = 0.025
    eval_noise: float = 0.015
    samples_per_transaction: int = 24
    warmup_repeats: int = 3
    intro_repeats: int = 3
    consolidation_transactions: int = 24


@dataclass(frozen=True)
class ModelConfig:
    max_cells: int = 24
    top_k: int = 2
    temperature: float = 0.08
    spawn_loss_threshold: float = 0.035
    spawn_margin_threshold: float = 0.18
    key_update_rate: float = 0.15
    value_update_rate: float = 0.15
    reuse_update_loss_threshold: float = 0.020


@dataclass(frozen=True)
class Transaction:
    factors: tuple[int, ...]  # evaluator-only hidden labels
    x: torch.Tensor
    y: torch.Tensor
    phase: str


@dataclass
class CellState:
    key: torch.Tensor
    value: torch.Tensor
    usage: int = 0
    born_at: int = 0


@dataclass(frozen=True)
class World:
    context_atoms: torch.Tensor
    effect_atoms: torch.Tensor
    transactions: tuple[Transaction, ...]
    heldout_singletons: tuple[Transaction, ...]
    heldout_pairs: tuple[Transaction, ...]


def _orthonormal_rows(count: int, dim: int, gen: torch.Generator) -> torch.Tensor:
    if count > dim:
        raise ValueError("count must not exceed dim")
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen, dtype=torch.float64))
    return q[:, :count].T.contiguous()


def _tx(
    context_atoms: torch.Tensor,
    effect_atoms: torch.Tensor,
    factors: Iterable[int],
    *,
    samples: int,
    noise: float,
    gen: torch.Generator,
    phase: str,
) -> Transaction:
    ids = tuple(sorted(int(i) for i in factors))
    if not ids or len(ids) > 2:
        raise ValueError("transactions must contain one or two factors")
    x0 = context_atoms[list(ids)].mean(dim=0)
    y0 = effect_atoms[list(ids)].mean(dim=0)
    x = x0.repeat(samples, 1)
    y = y0.repeat(samples, 1)
    if noise:
        x = x + noise * torch.randn(x.shape, generator=gen, dtype=x.dtype)
        y = y + 0.20 * noise * torch.randn(y.shape, generator=gen, dtype=y.dtype)
    return Transaction(ids, x, y, phase)


def build_world(seed: int, cfg: WorldConfig = WorldConfig()) -> World:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    context_atoms = _orthonormal_rows(cfg.factor_count, cfg.context_dim, gen)
    effect_atoms = _orthonormal_rows(cfg.factor_count, cfg.effect_dim, gen)
    transactions: list[Transaction] = []
    seen: list[int] = []

    for factor in range(cfg.factor_count):
        repeats = cfg.warmup_repeats if factor < 2 else cfg.intro_repeats
        phase = "warmup" if factor < 2 else "introduction"
        transactions.append(
            _tx(
                context_atoms,
                effect_atoms,
                [factor],
                samples=cfg.samples_per_transaction,
                noise=cfg.train_noise,
                gen=gen,
                phase=phase,
            )
        )
        seen.append(factor)
        for repeat in range(repeats - 1):
            if len(seen) == 1 or repeat == 0:
                factors = [factor]
            else:
                partner = seen[(factor + repeat) % (len(seen) - 1)]
                factors = [factor, partner]
            transactions.append(
                _tx(
                    context_atoms,
                    effect_atoms,
                    factors,
                    samples=cfg.samples_per_transaction,
                    noise=cfg.train_noise,
                    gen=gen,
                    phase=phase,
                )
            )

    pairs = [(i, j) for i in range(cfg.factor_count) for j in range(i + 1, cfg.factor_count)]
    rng = random.Random(seed ^ 0xC1A0)
    for step in range(cfg.consolidation_transactions):
        factors = (
            [step % cfg.factor_count]
            if step % 4 == 0
            else list(pairs[rng.randrange(len(pairs))])
        )
        transactions.append(
            _tx(
                context_atoms,
                effect_atoms,
                factors,
                samples=cfg.samples_per_transaction,
                noise=cfg.train_noise,
                gen=gen,
                phase="consolidation",
            )
        )

    heldout_singletons = tuple(
        _tx(
            context_atoms,
            effect_atoms,
            [i],
            samples=64,
            noise=cfg.eval_noise,
            gen=gen,
            phase="heldout-single",
        )
        for i in range(cfg.factor_count)
    )
    heldout_pairs = tuple(
        _tx(
            context_atoms,
            effect_atoms,
            [i, j],
            samples=64,
            noise=cfg.eval_noise,
            gen=gen,
            phase="heldout-pair",
        )
        for i in range(cfg.factor_count)
        for j in range(i + 1, cfg.factor_count)
    )
    return World(
        context_atoms,
        effect_atoms,
        tuple(transactions),
        heldout_singletons,
        heldout_pairs,
    )


class GrowingCellMemory:
    """Scaffolded constructive Cell memory; learner decisions consume x/y only."""

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        self.cfg = cfg
        self.cells: list[CellState] = []
        self.spawn_events: list[int] = []

    @property
    def active_cells(self) -> int:
        return len(self.cells)

    def _stack(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cells:
            raise RuntimeError("memory has no active cells")
        keys = F.normalize(torch.stack([cell.key for cell in self.cells]), dim=1)
        values = torch.stack([cell.value for cell in self.cells])
        return keys, values

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys, values = self._stack()
        k = min(self.cfg.top_k, self.active_cells)
        scores, indices = torch.topk(x @ keys.T, k=k, dim=1)
        weights = torch.softmax(scores / self.cfg.temperature, dim=1)
        pred = (weights.unsqueeze(-1) * values[indices]).sum(dim=1)
        return pred, weights, indices

    def _loss_margin(self, tx: Transaction) -> tuple[float, float]:
        if not self.cells:
            return float(torch.mean(tx.y.square()).item()), 0.0
        pred, weights, _ = self.predict(tx.x)
        loss = float(torch.mean((pred - tx.y).square()).item())
        margin = (
            0.0
            if weights.shape[1] == 1
            else float((weights[:, 0] - weights[:, 1]).abs().mean().item())
        )
        return loss, margin

    def _spawn(self, tx: Transaction, step: int) -> None:
        if self.active_cells >= self.cfg.max_cells:
            raise RuntimeError("maximum Cell count reached")
        key = F.normalize(tx.x.mean(dim=0), dim=0)
        value = tx.y.mean(dim=0).clone()
        self.cells.append(CellState(key=key, value=value, usage=1, born_at=step))
        self.spawn_events.append(step)

    def _reuse_update(self, tx: Transaction) -> None:
        _, _, indices = self.predict(tx.x)
        dominant = indices[:, 0]
        for cell_idx in dominant.unique(sorted=True).tolist():
            mask = dominant == int(cell_idx)
            cell = self.cells[int(cell_idx)]
            x_mean = F.normalize(tx.x[mask].mean(dim=0), dim=0)
            y_mean = tx.y[mask].mean(dim=0)
            cell.key = F.normalize(
                (1.0 - self.cfg.key_update_rate) * cell.key
                + self.cfg.key_update_rate * x_mean,
                dim=0,
            )
            cell.value = (
                (1.0 - self.cfg.value_update_rate) * cell.value
                + self.cfg.value_update_rate * y_mean
            )
            cell.usage += int(mask.sum().item())

    def observe(self, tx: Transaction, step: int) -> dict[str, Any]:
        loss_before, margin = self._loss_margin(tx)
        should_spawn = not self.cells or (
            loss_before > self.cfg.spawn_loss_threshold
            and margin < self.cfg.spawn_margin_threshold
        )
        if should_spawn and self.active_cells < self.cfg.max_cells:
            self._spawn(tx, step)
            action = "spawn"
        else:
            action = "reuse"
            if loss_before > self.cfg.reuse_update_loss_threshold:
                self._reuse_update(tx)
        loss_after, _ = self._loss_margin(tx)
        return {
            "step": step,
            "phase": tx.phase,
            "action": action,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "active_cells": self.active_cells,
        }


def _mean_mse(memory: GrowingCellMemory, txs: Iterable[Transaction]) -> float:
    losses = []
    for tx in txs:
        pred, _, _ = memory.predict(tx.x)
        losses.append(float(torch.mean((pred - tx.y).square()).item()))
    return float(sum(losses) / max(len(losses), 1))


def _alignment(memory: GrowingCellMemory, world: World) -> dict[str, Any]:
    keys, values = memory._stack()
    value_cos = F.normalize(values, dim=1) @ F.normalize(world.effect_atoms, dim=1).T
    key_cos = keys @ F.normalize(world.context_atoms, dim=1).T
    best_value, best_factor = value_cos.max(dim=1)
    matched_key = key_cos.gather(1, best_factor[:, None]).squeeze(1)
    covered = len(set(int(i) for i in best_factor[best_value > 0.80].tolist()))
    return {
        "mean_best_value_cosine": float(best_value.mean().item()),
        "mean_matched_key_cosine": float(matched_key.mean().item()),
        "covered_factors": covered,
        "cell_to_factor": [int(i) for i in best_factor.tolist()],
    }


def _route_recall(
    memory: GrowingCellMemory,
    world: World,
    txs: Iterable[Transaction],
) -> float:
    mapping = _alignment(memory, world)["cell_to_factor"]
    scores: list[float] = []
    for tx in txs:
        _, _, indices = memory.predict(tx.x)
        target = set(tx.factors)
        for row in indices.tolist():
            predicted = {mapping[int(i)] for i in row}
            scores.append(len(target & predicted) / max(len(target), 1))
    return float(sum(scores) / max(len(scores), 1))


def _shuffled_address_world(world: World, seed: int) -> World:
    rng = random.Random(seed ^ 0x5A17)
    contexts = [tx.x for tx in world.transactions]
    perm = list(range(len(contexts)))
    rng.shuffle(perm)
    txs = tuple(
        Transaction(tx.factors, contexts[perm[i]].clone(), tx.y.clone(), tx.phase)
        for i, tx in enumerate(world.transactions)
    )
    return World(
        world.context_atoms,
        world.effect_atoms,
        txs,
        world.heldout_singletons,
        world.heldout_pairs,
    )


def evaluate_gates(result: dict[str, Any], cfg: WorldConfig) -> dict[str, bool]:
    return {
        "bounded_growth": int(result["active_cells"]) <= cfg.factor_count + 2,
        "late_growth_low": int(result["late_spawns"]) <= 1,
        "compression": float(result["independent_memory_compression"]) >= 4.0,
        "singleton_quality": float(result["heldout_singleton_mse"]) <= 0.020,
        "composition_quality": float(result["heldout_pair_mse"]) <= 0.030,
        "single_route_recall": float(result["heldout_single_route_recall"]) >= 0.90,
        "pair_route_recall": float(result["heldout_pair_route_recall"]) >= 0.85,
        "value_coordinate_recovery": (
            float(result["alignment"]["mean_best_value_cosine"]) >= 0.90
        ),
        "key_value_alignment": (
            float(result["alignment"]["mean_matched_key_cosine"]) >= 0.85
        ),
        "factor_coverage": int(result["alignment"]["covered_factors"]) >= cfg.factor_count - 1,
        "beats_shuffled_address": (
            float(result["heldout_pair_mse"])
            <= 0.50 * float(result["shuffled_address_pair_mse"])
        ),
    }


def run_seed(
    seed: int,
    world_cfg: WorldConfig = WorldConfig(),
    model_cfg: ModelConfig = ModelConfig(),
) -> dict[str, Any]:
    world = build_world(seed, world_cfg)
    memory = GrowingCellMemory(model_cfg)
    records = [memory.observe(tx, step) for step, tx in enumerate(world.transactions)]
    alignment = _alignment(memory, world)

    shuffled_world = _shuffled_address_world(world, seed)
    shuffled = GrowingCellMemory(model_cfg)
    for step, tx in enumerate(shuffled_world.transactions):
        shuffled.observe(tx, step)

    half = len(world.transactions) // 2
    result = {
        "seed": int(seed),
        "world": asdict(world_cfg),
        "model": asdict(model_cfg),
        "transactions": len(world.transactions),
        "active_cells": memory.active_cells,
        "spawn_events": list(memory.spawn_events),
        "early_spawns": sum(step < half for step in memory.spawn_events),
        "late_spawns": sum(step >= half for step in memory.spawn_events),
        "independent_memory_compression": (
            len(world.transactions) / max(memory.active_cells, 1)
        ),
        "heldout_singleton_mse": _mean_mse(memory, world.heldout_singletons),
        "heldout_pair_mse": _mean_mse(memory, world.heldout_pairs),
        "heldout_single_route_recall": _route_recall(
            memory, world, world.heldout_singletons
        ),
        "heldout_pair_route_recall": _route_recall(memory, world, world.heldout_pairs),
        "alignment": alignment,
        "shuffled_address_pair_mse": _mean_mse(shuffled, world.heldout_pairs),
        "shuffled_address_cells": shuffled.active_cells,
        "records": records,
    }
    result["gates"] = evaluate_gates(result, world_cfg)
    result["pass"] = all(result["gates"].values())
    return result


def run_formal(seeds: Iterable[int] = FORMAL_SEEDS) -> dict[str, Any]:
    results = [run_seed(int(seed)) for seed in seeds]
    passed = sum(bool(result["pass"]) for result in results)
    return {
        "experiment": "constructive-clm-001",
        "status": (
            "LEARNED_COORDINATE_FORMATION_SUPPORTED"
            if passed == len(results)
            else "LEARNED_COORDINATE_FORMATION_NOT_SUPPORTED"
        ),
        "scientific_decision": True,
        "passed_seeds": passed,
        "total_seeds": len(results),
        "results": results,
    }
