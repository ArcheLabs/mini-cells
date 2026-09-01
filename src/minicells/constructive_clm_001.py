from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


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
    factors: tuple[int, ...]
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


def _orthonormal_rows(count: int, dim: int, *, generator: torch.Generator) -> torch.Tensor:
    if count > dim:
        raise ValueError("count must not exceed dim")
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    return q[:, :count].T.contiguous()


def _sample_transaction(
    context_atoms: torch.Tensor,
    effect_atoms: torch.Tensor,
    factors: Iterable[int],
    *,
    samples: int,
    noise: float,
    generator: torch.Generator,
    phase: str,
) -> Transaction:
    idx = tuple(sorted(int(i) for i in factors))
    if not idx or len(idx) > 2:
        raise ValueError("transactions must contain one or two factors")
    base_x = context_atoms[list(idx)].mean(dim=0)
    base_y = effect_atoms[list(idx)].mean(dim=0)
    x = base_x.repeat(samples, 1)
    y = base_y.repeat(samples, 1)
    if noise > 0:
        x = x + noise * torch.randn(x.shape, generator=generator, dtype=x.dtype)
        y = y + 0.20 * noise * torch.randn(y.shape, generator=generator, dtype=y.dtype)
    return Transaction(idx, x, y, phase)


def build_world(seed: int, cfg: WorldConfig = WorldConfig()) -> World:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    context_atoms = _orthonormal_rows(cfg.factor_count, cfg.context_dim, generator=gen)
    effect_atoms = _orthonormal_rows(cfg.factor_count, cfg.effect_dim, generator=gen)

    txs: list[Transaction] = []
    seen: list[int] = []
    for f in range(cfg.factor_count):
        if f < 2:
            repeats = cfg.warmup_repeats
            phase = "warmup"
        else:
            repeats = cfg.intro_repeats
            phase = "introduction"
        # A new factor first appears alone. The learner is never given f; this only
        # defines the hidden evaluation world and guarantees a clean novelty event.
        txs.append(
            _sample_transaction(
                context_atoms,
                effect_atoms,
                [f],
                samples=cfg.samples_per_transaction,
                noise=cfg.train_noise,
                generator=gen,
                phase=phase,
            )
        )
        seen.append(f)
        for r in range(repeats - 1):
            if len(seen) == 1 or r == 0:
                factors = [f]
            else:
                partner = seen[(f + r) % (len(seen) - 1)]
                factors = [f, partner]
            txs.append(
                _sample_transaction(
                    context_atoms,
                    effect_atoms,
                    factors,
                    samples=cfg.samples_per_transaction,
                    noise=cfg.train_noise,
                    generator=gen,
                    phase=phase,
                )
            )

    pair_space = [(i, j) for i in range(cfg.factor_count) for j in range(i + 1, cfg.factor_count)]
    py_rng = random.Random(seed ^ 0xC1A0)
    for t in range(cfg.consolidation_transactions):
        if t % 4 == 0:
            factors = [t % cfg.factor_count]
        else:
            factors = list(pair_space[py_rng.randrange(len(pair_space))])
        txs.append(
            _sample_transaction(
                context_atoms,
                effect_atoms,
                factors,
                samples=cfg.samples_per_transaction,
                noise=cfg.train_noise,
                generator=gen,
                phase="consolidation",
            )
        )

    singles = tuple(
        _sample_transaction(
            context_atoms,
            effect_atoms,
            [i],
            samples=64,
            noise=cfg.eval_noise,
            generator=gen,
            phase="heldout-single",
        )
        for i in range(cfg.factor_count)
    )
    pairs = tuple(
        _sample_transaction(
            context_atoms,
            effect_atoms,
            [i, j],
            samples=64,
            noise=cfg.eval_noise,
            generator=gen,
            phase="heldout-pair",
        )
        for i in range(cfg.factor_count)
        for j in range(i + 1, cfg.factor_count)
    )
    return World(context_atoms, effect_atoms, tuple(txs), singles, pairs)


class GrowingCellMemory:
    """A minimal learned Cell coordinate system.

    Training sees only (x, y). Hidden factor IDs exist solely in World/Transaction for
    post-hoc scientific evaluation. Each Cell learns a read key and an effect value.
    Growth is residual-triggered; reuse updates are local EMAs rather than replay.
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        self.cfg = cfg
        self.cells: list[CellState] = []
        self.spawn_events: list[int] = []
        self.transaction_losses: list[float] = []
        self.transaction_margins: list[float] = []

    @property
    def active_cells(self) -> int:
        return len(self.cells)

    def _stack(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cells:
            raise RuntimeError("memory has no active cells")
        keys = F.normalize(torch.stack([c.key for c in self.cells]), dim=1)
        values = torch.stack([c.value for c in self.cells])
        return keys, values

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.cells:
            raise RuntimeError("memory has no active cells")
        keys, values = self._stack()
        logits = x @ keys.T
        k = min(self.cfg.top_k, self.active_cells)
        scores, indices = torch.topk(logits, k=k, dim=1)
        weights = torch.softmax(scores / self.cfg.temperature, dim=1)
        selected = values[indices]
        pred = (weights.unsqueeze(-1) * selected).sum(dim=1)
        return pred, weights, indices

    def _loss_and_margin(self, tx: Transaction) -> tuple[float, float]:
        if not self.cells:
            loss = float(torch.mean(tx.y.square()).item())
            return loss, 0.0
        pred, weights, _ = self.predict(tx.x)
        loss = float(torch.mean((pred - tx.y).square()).item())
        margin = 0.0
        if weights.shape[1] > 1:
            margin = float((weights[:, 0] - weights[:, 1]).abs().mean().item())
        return loss, margin

    def _spawn(self, tx: Transaction, step: int) -> None:
        if self.active_cells >= self.cfg.max_cells:
            raise RuntimeError("maximum Cell count reached")
        # Data-only birth: no factor/task label is used. A newborn Cell is a
        # probationary local prototype of the transaction it was created to fit.
        key = F.normalize(tx.x.mean(dim=0), dim=0)
        value = tx.y.mean(dim=0).clone()
        self.cells.append(CellState(key=key, value=value, usage=1, born_at=step))
        self.spawn_events.append(step)

    def _reuse_update(self, tx: Transaction) -> None:
        if not self.cells:
            return
        _, _, indices = self.predict(tx.x)
        # Update only the most-used routes of the current transaction. This is a
        # local online write; there is no replay buffer or hidden factor oracle.
        dominant = indices[:, 0]
        for cell_idx in dominant.unique(sorted=True).tolist():
            mask = dominant == int(cell_idx)
            if not bool(mask.any()):
                continue
            cell = self.cells[int(cell_idx)]
            x_mean = F.normalize(tx.x[mask].mean(dim=0), dim=0)
            y_mean = tx.y[mask].mean(dim=0)
            kr = self.cfg.key_update_rate
            vr = self.cfg.value_update_rate
            cell.key = F.normalize((1.0 - kr) * cell.key + kr * x_mean, dim=0)
            cell.value = (1.0 - vr) * cell.value + vr * y_mean
            cell.usage += int(mask.sum().item())

    def observe(self, tx: Transaction, step: int) -> dict[str, Any]:
        loss_before, margin = self._loss_and_margin(tx)
        should_spawn = (
            not self.cells
            or (
                loss_before > self.cfg.spawn_loss_threshold
                and margin < self.cfg.spawn_margin_threshold
            )
        )
        if should_spawn and self.active_cells < self.cfg.max_cells:
            self._spawn(tx, step)
            action = "spawn"
        else:
            action = "reuse"
            if loss_before > self.cfg.reuse_update_loss_threshold:
                self._reuse_update(tx)
        loss_after, _ = self._loss_and_margin(tx)
        self.transaction_losses.append(loss_after)
        self.transaction_margins.append(margin)
        return {
            "step": step,
            "phase": tx.phase,
            "action": action,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "active_cells": self.active_cells,
        }


def _evaluate(memory: GrowingCellMemory, txs: Iterable[Transaction]) -> dict[str, float]:
    losses: list[float] = []
    for tx in txs:
        pred, _, _ = memory.predict(tx.x)
        losses.append(float(torch.mean((pred - tx.y).square()).item()))
    return {
        "mse": float(sum(losses) / max(len(losses), 1)),
        "count": float(len(losses)),
    }


def _cell_alignment(memory: GrowingCellMemory, world: World) -> dict[str, Any]:
    if not memory.cells:
        return {
            "mean_best_value_cosine": 0.0,
            "mean_matched_key_cosine": 0.0,
            "covered_factors": 0,
            "cell_to_factor": [],
        }
    keys, values = memory._stack()
    value_cos = F.normalize(values, dim=1) @ F.normalize(world.effect_atoms, dim=1).T
    key_cos = keys @ F.normalize(world.context_atoms, dim=1).T
    best_v, best_idx = value_cos.max(dim=1)
    best_k = key_cos.gather(1, best_idx[:, None]).squeeze(1)
    covered = len(set(int(i) for i in best_idx[best_v > 0.80].tolist()))
    return {
        "mean_best_value_cosine": float(best_v.mean().item()),
        "mean_matched_key_cosine": float(best_k.mean().item()),
        "covered_factors": int(covered),
        "cell_to_factor": [int(i) for i in best_idx.tolist()],
    }


def _routing_recall(memory: GrowingCellMemory, world: World, txs: Iterable[Transaction]) -> float:
    if not memory.cells:
        return 0.0
    alignment = _cell_alignment(memory, world)
    mapping = alignment["cell_to_factor"]
    scores: list[float] = []
    for tx in txs:
        _, _, indices = memory.predict(tx.x)
        target = set(tx.factors)
        for row in indices.tolist():
            predicted = {mapping[int(i)] for i in row}
            scores.append(len(target & predicted) / max(len(target), 1))
    return float(sum(scores) / max(len(scores), 1))


def _shuffled_address_control(world: World, seed: int) -> World:
    # Break the stable read->effect relation while preserving all target effects and
    # transaction counts. Each transaction receives a deterministic random context
    # from a different hidden factor/composition.
    rng = random.Random(seed ^ 0x5A17)
    sources = [tx.x for tx in world.transactions]
    perm = list(range(len(sources)))
    rng.shuffle(perm)
    txs = tuple(
        Transaction(tx.factors, sources[perm[i]].clone(), tx.y.clone(), tx.phase)
        for i, tx in enumerate(world.transactions)
    )
    return World(
        world.context_atoms,
        world.effect_atoms,
        txs,
        world.heldout_singletons,
        world.heldout_pairs,
    )


def run_seed(
    seed: int,
    world_cfg: WorldConfig = WorldConfig(),
    model_cfg: ModelConfig = ModelConfig(),
) -> dict[str, Any]:
    world = build_world(seed, world_cfg)
    memory = GrowingCellMemory(model_cfg)
    records = [memory.observe(tx, i) for i, tx in enumerate(world.transactions)]

    singles = _evaluate(memory, world.heldout_singletons)
    pairs = _evaluate(memory, world.heldout_pairs)
    alignment = _cell_alignment(memory, world)
    pair_route_recall = _routing_recall(memory, world, world.heldout_pairs)
    single_route_recall = _routing_recall(memory, world, world.heldout_singletons)

    shuffled_world = _shuffled_address_control(world, seed)
    shuffled = GrowingCellMemory(model_cfg)
    for i, tx in enumerate(shuffled_world.transactions):
        shuffled.observe(tx, i)
    shuffled_eval = _evaluate(shuffled, world.heldout_pairs)

    half = len(world.transactions) // 2
    early_spawns = sum(1 for s in memory.spawn_events if s < half)
    late_spawns = sum(1 for s in memory.spawn_events if s >= half)
    independent_memory_ratio = len(world.transactions) / max(memory.active_cells, 1)

    result = {
        "seed": int(seed),
        "world": asdict(world_cfg),
        "model": asdict(model_cfg),
        "transactions": len(world.transactions),
        "active_cells": memory.active_cells,
        "spawn_events": list(memory.spawn_events),
        "early_spawns": early_spawns,
        "late_spawns": late_spawns,
        "independent_memory_compression": float(independent_memory_ratio),
        "heldout_singleton_mse": singles["mse"],
        "heldout_pair_mse": pairs["mse"],
        "heldout_single_route_recall": single_route_recall,
        "heldout_pair_route_recall": pair_route_recall,
        "alignment": alignment,
        "shuffled_address_pair_mse": shuffled_eval["mse"],
        "shuffled_address_cells": shuffled.active_cells,
        "records": records,
    }
    result["gates"] = evaluate_gates(result, world_cfg)
    result["pass"] = all(result["gates"].values())
    return result


def evaluate_gates(result: dict[str, Any], world_cfg: WorldConfig) -> dict[str, bool]:
    return {
        "bounded_growth": int(result["active_cells"]) <= world_cfg.factor_count + 2,
        "late_growth_low": int(result["late_spawns"]) <= 1,
        "compression": float(result["independent_memory_compression"]) >= 4.0,
        "singleton_quality": float(result["heldout_singleton_mse"]) <= 0.020,
        "composition_quality": float(result["heldout_pair_mse"]) <= 0.030,
        "single_route_recall": float(result["heldout_single_route_recall"]) >= 0.90,
        "pair_route_recall": float(result["heldout_pair_route_recall"]) >= 0.85,
        "value_coordinate_recovery": float(result["alignment"]["mean_best_value_cosine"]) >= 0.90,
        "key_value_alignment": float(result["alignment"]["mean_matched_key_cosine"]) >= 0.85,
        "factor_coverage": int(result["alignment"]["covered_factors"]) >= world_cfg.factor_count - 1,
        "beats_shuffled_address": float(result["heldout_pair_mse"])
        <= 0.50 * float(result["shuffled_address_pair_mse"]),
    }


def run_formal(seeds: Iterable[int] = (1001, 1002, 1003)) -> dict[str, Any]:
    results = [run_seed(int(seed)) for seed in seeds]
    passed = sum(bool(r["pass"]) for r in results)
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
