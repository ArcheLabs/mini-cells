from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from minicells.constructive_clm_001b import LearnerConfig as BootstrapLearnerConfig
from minicells.constructive_clm_001b import RelationalCellLearner
from minicells.constructive_clm_002 import GrowthConfig
from minicells.constructive_clm_002 import StreamingGrowthLearner
from minicells.constructive_clm_002 import WorldConfig as GrowthWorldConfig
from minicells.constructive_clm_002 import _sample_mixture as sample_growth_mixture
from minicells.constructive_clm_002 import build_world as build_growth_world
from minicells.subspace_mitosis_005 import constrained_update, extend_basis

_DTYPE = torch.float64
VARIANTS = (
    "unsafe",
    "certificate_no_growth",
    "certificate_growth",
    "replay_growth_oracle",
)


@dataclass(frozen=True)
class StructuralBridgeConfig:
    max_transactions: int = 1024
    initial_factors: int = 6
    final_factors: int = 12
    context_dim: int = 48
    effect_dim: int = 40
    bootstrap_cycles: int = 6
    introduction_repeats: int = 4
    growth_alpha: float = 0.60
    stabilization_tail: int = 64
    samples_per_transaction: int = 16
    train_noise: float = 0.02


@dataclass(frozen=True)
class ProtectionConfig:
    modes_per_root: int = 3
    mode_offset: float = 0.25
    route_context_noise: float = 0.004
    feature_dim: int = 8
    output_dim: int = 4
    examples_per_write: int = 16
    acquisition_blocks: tuple[tuple[int, int], ...] = ((0, 1), (2, 3))
    tail_reuse_transactions: int = 720
    certificate_feasibility_threshold: float = 1e-8
    numerical_rank_tolerance: float = 1e-10
    reuse_residual_threshold: float = 1e-10


@dataclass(frozen=True)
class WriteTransaction:
    index: int
    root_id: int
    mode_id: int
    block_id: int
    context: torch.Tensor
    z: torch.Tensor
    target: torch.Tensor
    phase: str


@dataclass
class MutableCell:
    cell_id: int
    root_id: int
    route_key: torch.Tensor
    weight: torch.Tensor
    basis: torch.Tensor
    birth_step: int
    usage_count: int = 0

    def clone(self) -> "MutableCell":
        return MutableCell(
            cell_id=int(self.cell_id),
            root_id=int(self.root_id),
            route_key=self.route_key.clone(),
            weight=self.weight.clone(),
            basis=self.basis.clone(),
            birth_step=int(self.birth_step),
            usage_count=int(self.usage_count),
        )


@dataclass(frozen=True)
class HistoryItem:
    cell_id: int
    context: torch.Tensor
    z: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class ProtectionWorld:
    root_anchors: torch.Tensor
    mode_keys: torch.Tensor
    decoders: torch.Tensor
    acquisition: tuple[WriteTransaction, ...]
    tail: tuple[WriteTransaction, ...]


class HierarchicalProtectedModel:
    """Learned root routing plus context-keyed exclusive lineage routing."""

    def __init__(
        self,
        root_anchors: torch.Tensor,
        *,
        feature_dim: int,
        output_dim: int,
    ) -> None:
        anchors = F.normalize(root_anchors.to(dtype=_DTYPE), dim=1)
        self.root_anchors = anchors.clone()
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.cells: dict[int, MutableCell] = {}
        self.lineages: dict[int, list[int]] = {}
        for root_id in range(int(anchors.shape[0])):
            cell = MutableCell(
                cell_id=root_id,
                root_id=root_id,
                route_key=anchors[root_id].clone(),
                weight=torch.zeros(output_dim, feature_dim, dtype=_DTYPE),
                basis=torch.zeros(feature_dim, 0, dtype=_DTYPE),
                birth_step=-1,
            )
            self.cells[root_id] = cell
            self.lineages[root_id] = [root_id]
        self.next_cell_id = int(anchors.shape[0])
        self.spawn_steps: list[int] = []

    def clone(self) -> "HierarchicalProtectedModel":
        other = object.__new__(HierarchicalProtectedModel)
        other.root_anchors = self.root_anchors.clone()
        other.feature_dim = self.feature_dim
        other.output_dim = self.output_dim
        other.cells = {key: cell.clone() for key, cell in self.cells.items()}
        other.lineages = {key: list(value) for key, value in self.lineages.items()}
        other.next_cell_id = int(self.next_cell_id)
        other.spawn_steps = list(self.spawn_steps)
        return other

    @property
    def root_count(self) -> int:
        return int(self.root_anchors.shape[0])

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def route_root(self, context: torch.Tensor) -> int:
        query = context.mean(dim=0) if context.ndim == 2 else context
        query = F.normalize(query.to(dtype=_DTYPE), dim=0)
        return int(torch.argmax(self.root_anchors @ query).item())

    def route(self, context: torch.Tensor) -> tuple[int, int]:
        query = context.mean(dim=0) if context.ndim == 2 else context
        query = F.normalize(query.to(dtype=_DTYPE), dim=0)
        root_id = self.route_root(query)
        lineage = self.lineages[root_id]
        scores = torch.tensor(
            [float(torch.dot(self.cells[cell_id].route_key, query).item()) for cell_id in lineage],
            dtype=_DTYPE,
        )
        cell_id = int(lineage[int(torch.argmax(scores).item())])
        return root_id, cell_id

    def forward(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        _, cell_id = self.route(context)
        return z.to(dtype=_DTYPE) @ self.cells[cell_id].weight.T

    def spawn_child(self, root_id: int, route_key: torch.Tensor, step: int) -> int:
        cell_id = int(self.next_cell_id)
        self.next_cell_id += 1
        key = route_key.mean(dim=0) if route_key.ndim == 2 else route_key
        cell = MutableCell(
            cell_id=cell_id,
            root_id=int(root_id),
            route_key=F.normalize(key.to(dtype=_DTYPE), dim=0),
            weight=torch.zeros(self.output_dim, self.feature_dim, dtype=_DTYPE),
            basis=torch.zeros(self.feature_dim, 0, dtype=_DTYPE),
            birth_step=int(step),
        )
        self.cells[cell_id] = cell
        self.lineages[int(root_id)].append(cell_id)
        self.spawn_steps.append(int(step))
        return cell_id


class HistoryEvaluator:
    """Evaluator/replay-oracle history; certificate_growth never receives this object."""

    def __init__(self, feature_dim: int, numerical_rank_tolerance: float) -> None:
        self.feature_dim = int(feature_dim)
        self.tolerance = float(numerical_rank_tolerance)
        self.items: list[HistoryItem] = []

    def clone(self) -> "HistoryEvaluator":
        other = HistoryEvaluator(self.feature_dim, self.tolerance)
        other.items = [
            HistoryItem(
                cell_id=int(item.cell_id),
                context=item.context.clone(),
                z=item.z.clone(),
                output=item.output.clone(),
            )
            for item in self.items
        ]
        return other

    def behavior_mse(self, model: HierarchicalProtectedModel) -> float:
        if not self.items:
            return 0.0
        values = [
            float(torch.mean((model.forward(item.context, item.z) - item.output).square()).item())
            for item in self.items
        ]
        return float(statistics.fmean(values))

    def record(self, model: HierarchicalProtectedModel, transaction: WriteTransaction) -> None:
        _, cell_id = model.route(transaction.context)
        self.items.append(
            HistoryItem(
                cell_id=int(cell_id),
                context=transaction.context.detach().clone(),
                z=transaction.z.detach().clone(),
                output=model.forward(transaction.context, transaction.z).detach().clone(),
            )
        )

    def items_for_cell(self, cell_id: int) -> list[HistoryItem]:
        return [item for item in self.items if int(item.cell_id) == int(cell_id)]

    def full_basis(self, cell_id: int) -> torch.Tensor:
        items = self.items_for_cell(cell_id)
        if not items:
            return torch.zeros(self.feature_dim, 0, dtype=_DTYPE)
        return extend_basis(
            torch.zeros(self.feature_dim, 0, dtype=_DTYPE),
            torch.cat([item.z for item in items], dim=0),
            tolerance=self.tolerance,
        )


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def _empty_basis(feature_dim: int) -> torch.Tensor:
    return torch.zeros(feature_dim, 0, dtype=_DTYPE)


def _bridge_world_config(cfg: StructuralBridgeConfig) -> GrowthWorldConfig:
    return GrowthWorldConfig(
        max_transactions=cfg.max_transactions,
        initial_factors=cfg.initial_factors,
        max_factors=cfg.final_factors,
        context_dim=cfg.context_dim,
        effect_dim=cfg.effect_dim,
        context_common_rho=0.18,
        effect_common_rho=0.12,
        atom_jitter=0.05,
        bootstrap_cycles=cfg.bootstrap_cycles,
        introduction_repeats=cfg.introduction_repeats,
        growth_alpha=cfg.growth_alpha,
        stabilization_tail=cfg.stabilization_tail,
        samples_per_transaction=cfg.samples_per_transaction,
        train_noise=cfg.train_noise,
        eval_noise=0.01,
        checkpoints=(cfg.max_transactions,),
    )


def learn_structural_roots(
    seed: int,
    cfg: StructuralBridgeConfig = StructuralBridgeConfig(),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reuse the CLM-002 mechanism to obtain learned root coordinates for CLM-003."""
    world_cfg = _bridge_world_config(cfg)
    world = build_growth_world(int(seed), world_cfg)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0xC1A003)
    rng = random.Random(int(seed) ^ 0x51A003)

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
            transaction = sample_growth_mixture(
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
        raise RuntimeError(f"CLM-001B bootstrap failed inside CLM-003: {fit}")

    keys, values = bootstrap.keys_values()
    learner = StreamingGrowthLearner(
        keys,
        values,
        born_at=step - 1,
        cfg=GrowthConfig(),
    )
    active_true_factors = cfg.initial_factors
    introduction_queue: list[tuple[int, int]] = []

    while step < cfg.max_transactions:
        if step in world.introduction_starts:
            novel_factor = int(world.introduction_starts[step])
            anchors = rng.sample(range(novel_factor), cfg.introduction_repeats)
            introduction_queue = [(novel_factor, int(anchor)) for anchor in anchors]

        if introduction_queue:
            novel_factor, anchor = introduction_queue.pop(0)
            anchor_weight = 0.30 + 0.40 * rng.random()
            transaction = sample_growth_mixture(
                world,
                (anchor, novel_factor),
                (anchor_weight, 1.0 - anchor_weight),
                samples=cfg.samples_per_transaction,
                context_noise=cfg.train_noise,
                effect_noise=0.30 * cfg.train_noise,
                generator=generator,
                phase="introduction",
            )
            if not introduction_queue:
                active_true_factors = max(active_true_factors, novel_factor + 1)
        else:
            support_size = 2 if rng.random() < 0.65 else 3
            support = tuple(rng.sample(range(active_true_factors), support_size))
            weights = [0.25 + rng.random() for _ in support]
            transaction = sample_growth_mixture(
                world,
                support,
                weights,
                samples=cfg.samples_per_transaction,
                context_noise=cfg.train_noise,
                effect_noise=0.30 * cfg.train_noise,
                generator=generator,
                phase="reuse",
            )
        learner.observe(transaction.x, transaction.y, step)
        step += 1

    normalized_keys = F.normalize(learner.keys, dim=1)
    true_atoms = F.normalize(world.context_atoms[: cfg.final_factors], dim=1)
    cosine = normalized_keys @ true_atoms.T
    best_values, best_indices = cosine.max(dim=1)
    mapping = [int(index) for index in best_indices.tolist()]
    diagnostics = {
        "bootstrap_fit": fit,
        "transactions": int(step),
        "root_cells": int(learner.cell_count),
        "true_factors": int(cfg.final_factors),
        "covered_factors": len(set(mapping)),
        "duplicate_assignments": len(mapping) - len(set(mapping)),
        "mean_matched_root_key_cosine": float(best_values.mean().item()),
        "min_matched_root_key_cosine": float(best_values.min().item()),
        "mapping": mapping,
        "spawn_count": len(learner.spawn_steps),
        "last_spawn_step": int(learner.spawn_steps[-1]) if learner.spawn_steps else -1,
    }
    return normalized_keys.clone(), diagnostics


def _orthogonal_mode_keys(
    root_anchors: torch.Tensor,
    cfg: ProtectionConfig,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    roots = F.normalize(root_anchors.to(dtype=_DTYPE), dim=1)
    rows: list[torch.Tensor] = []
    for root_id in range(int(roots.shape[0])):
        root = roots[root_id]
        modes = [root.clone()]
        directions: list[torch.Tensor] = []
        for _mode in range(1, cfg.modes_per_root):
            accepted: torch.Tensor | None = None
            for _ in range(128):
                direction = torch.randn(root.shape, generator=generator, dtype=_DTYPE)
                direction = direction - torch.dot(direction, root) * root
                for prior in directions:
                    direction = direction - torch.dot(direction, prior) * prior
                norm = float(torch.linalg.norm(direction).item())
                if norm <= 1e-9:
                    continue
                direction = direction / norm
                candidate = F.normalize(root + cfg.mode_offset * direction, dim=0)
                own = float(torch.dot(candidate, root).item())
                other = torch.cat([roots[:root_id], roots[root_id + 1 :]], dim=0)
                competitor = (
                    float(torch.max(other @ candidate).item()) if other.numel() else -1.0
                )
                if own >= competitor + 0.20:
                    accepted = direction
                    break
            if accepted is None:
                raise RuntimeError("failed to construct route-stable local mode key")
            directions.append(accepted)
            modes.append(F.normalize(root + cfg.mode_offset * accepted, dim=0))
        rows.append(torch.stack(modes))
    return torch.stack(rows)


def _decoder_bank(
    roots: int,
    cfg: ProtectionConfig,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    blocks = len(cfg.acquisition_blocks)
    bank = torch.zeros(
        roots,
        cfg.modes_per_root,
        blocks,
        cfg.output_dim,
        2,
        dtype=_DTYPE,
    )
    for root in range(roots):
        for block in range(blocks):
            base = torch.randn(cfg.output_dim, 2, generator=generator, dtype=_DTYPE)
            base = base / max(float(torch.linalg.norm(base).item()), 1e-12)
            bank[root, 0, block] = base
            bank[root, 1, block] = -base
            raw = torch.randn(cfg.output_dim, 2, generator=generator, dtype=_DTYPE)
            flat_base = base.flatten()
            flat_raw = raw.flatten()
            flat_raw = flat_raw - torch.dot(flat_raw, flat_base) * flat_base
            flat_raw = flat_raw / max(float(torch.linalg.norm(flat_raw).item()), 1e-12)
            bank[root, 2, block] = flat_raw.reshape(cfg.output_dim, 2)
    return bank


def _activation_batch(
    cfg: ProtectionConfig,
    dims: tuple[int, int],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    raw = torch.randn(cfg.examples_per_write, len(dims), generator=generator, dtype=_DTYPE)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    coeff = q[:, : len(dims)] * math.sqrt(cfg.examples_per_write)
    z = torch.zeros(cfg.examples_per_write, cfg.feature_dim, dtype=_DTYPE)
    z[:, list(dims)] = coeff
    return z


def _context_batch(
    key: torch.Tensor,
    cfg: ProtectionConfig,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    rows = key.repeat(cfg.examples_per_write, 1)
    if cfg.route_context_noise > 0.0:
        rows = rows + cfg.route_context_noise * torch.randn(
            rows.shape,
            generator=generator,
            dtype=_DTYPE,
        )
    return rows


def build_protection_world(
    root_anchors: torch.Tensor,
    seed: int,
    cfg: ProtectionConfig = ProtectionConfig(),
) -> ProtectionWorld:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0xA003)
    rng = random.Random(int(seed) ^ 0xB003)
    mode_keys = _orthogonal_mode_keys(root_anchors, cfg, generator=generator)
    decoders = _decoder_bank(int(root_anchors.shape[0]), cfg, generator=generator)

    acquisition: list[WriteTransaction] = []
    index = 0
    for mode_id in range(cfg.modes_per_root):
        for block_id, dims in enumerate(cfg.acquisition_blocks):
            roots = list(range(int(root_anchors.shape[0])))
            rng.shuffle(roots)
            for root_id in roots:
                z = _activation_batch(cfg, dims, generator=generator)
                target = z[:, list(dims)] @ decoders[root_id, mode_id, block_id].T
                acquisition.append(
                    WriteTransaction(
                        index=index,
                        root_id=int(root_id),
                        mode_id=int(mode_id),
                        block_id=int(block_id),
                        context=_context_batch(
                            mode_keys[root_id, mode_id],
                            cfg,
                            generator=generator,
                        ),
                        z=z,
                        target=target,
                        phase="acquisition",
                    )
                )
                index += 1

    tail: list[WriteTransaction] = []
    for _ in range(cfg.tail_reuse_transactions):
        root_id = rng.randrange(int(root_anchors.shape[0]))
        mode_id = rng.randrange(cfg.modes_per_root)
        block_id = rng.randrange(len(cfg.acquisition_blocks))
        dims = cfg.acquisition_blocks[block_id]
        z = _activation_batch(cfg, dims, generator=generator)
        target = z[:, list(dims)] @ decoders[root_id, mode_id, block_id].T
        tail.append(
            WriteTransaction(
                index=index,
                root_id=int(root_id),
                mode_id=int(mode_id),
                block_id=int(block_id),
                context=_context_batch(
                    mode_keys[root_id, mode_id],
                    cfg,
                    generator=generator,
                ),
                z=z,
                target=target,
                phase="tail",
            )
        )
        index += 1

    return ProtectionWorld(
        root_anchors=F.normalize(root_anchors.to(dtype=_DTYPE), dim=1),
        mode_keys=mode_keys,
        decoders=decoders,
        acquisition=tuple(acquisition),
        tail=tuple(tail),
    )


def _fresh_behavior_eval(
    model: HierarchicalProtectedModel,
    world: ProtectionWorld,
    seed: int,
    cfg: ProtectionConfig,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0xE003)
    mse: list[float] = []
    for root_id in range(model.root_count):
        for mode_id in range(cfg.modes_per_root):
            for block_id, dims in enumerate(cfg.acquisition_blocks):
                z = _activation_batch(cfg, dims, generator=generator)
                context = _context_batch(
                    world.mode_keys[root_id, mode_id],
                    cfg,
                    generator=generator,
                )
                target = z[:, list(dims)] @ world.decoders[root_id, mode_id, block_id].T
                mse.append(float(torch.mean((model.forward(context, z) - target).square()).item()))
    return {"mse": _mean(mse), "examples": len(mse)}


def _route_metrics(
    model: HierarchicalProtectedModel,
    world: ProtectionWorld,
    expected_mode_cells: dict[tuple[int, int], int],
) -> dict[str, float]:
    root_hits: list[float] = []
    exact_hits: list[float] = []
    for root_id in range(model.root_count):
        for mode_id in range(world.mode_keys.shape[1]):
            key = world.mode_keys[root_id, mode_id]
            routed_root, routed_cell = model.route(key)
            root_hits.append(float(routed_root == root_id))
            expected = expected_mode_cells.get((root_id, mode_id))
            if expected is not None:
                exact_hits.append(float(routed_cell == expected))
    return {
        "root_accuracy": _mean(root_hits),
        "exact_mode_accuracy": _mean(exact_hits),
    }


def _apply_delta(cell: MutableCell, delta_weight: torch.Tensor) -> None:
    cell.weight = cell.weight + delta_weight.to(dtype=_DTYPE)


def run_variant(
    variant: str,
    world: ProtectionWorld,
    cfg: ProtectionConfig = ProtectionConfig(),
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    model = HierarchicalProtectedModel(
        world.root_anchors,
        feature_dim=cfg.feature_dim,
        output_dim=cfg.output_dim,
    )
    history = HistoryEvaluator(cfg.feature_dim, cfg.numerical_rank_tolerance)
    expected_mode_cells: dict[tuple[int, int], int] = {
        (root_id, 0): root_id for root_id in range(model.root_count)
    }
    acquisition_gain = 0.0
    accepted_acquisition = 0
    rejected_acquisition = 0
    cumulative_positive_regression = 0.0
    max_historical_regression = 0.0
    replay_accesses = 0
    old_sample_accesses = 0
    old_label_accesses = 0
    growth_rescues = 0
    safe_extensions = 0
    tail_spawns = 0
    tail_child_route_hits: list[float] = []
    action_log: list[dict[str, Any]] = []

    stream = [*world.acquisition, *world.tail]
    for step, transaction in enumerate(stream):
        root_id, cell_id = model.route(transaction.context)
        if root_id != transaction.root_id:
            action_log.append(
                {
                    "index": transaction.index,
                    "phase": transaction.phase,
                    "root": transaction.root_id,
                    "mode": transaction.mode_id,
                    "block": transaction.block_id,
                    "action": "root_misroute",
                    "routed_root": root_id,
                    "routed_cell": cell_id,
                }
            )
            continue

        if transaction.phase == "tail" and transaction.mode_id > 0:
            expected = expected_mode_cells.get((transaction.root_id, transaction.mode_id))
            if expected is not None:
                tail_child_route_hits.append(float(cell_id == expected))

        prediction = model.forward(transaction.context, transaction.z)
        residual = transaction.target - prediction
        pre_mse = float(torch.mean(residual.square()).item())
        before_history_mse = history.behavior_mse(model)
        action = "reuse"
        solved: dict[str, Any] | None = None
        committed = False
        grew = False
        write_cell_id = cell_id

        if pre_mse <= cfg.reuse_residual_threshold:
            model.cells[cell_id].usage_count += 1
        elif variant == "unsafe":
            solved = constrained_update(
                transaction.z,
                residual,
                _empty_basis(cfg.feature_dim),
                numerical_rank_tolerance=cfg.numerical_rank_tolerance,
            )
            _apply_delta(model.cells[cell_id], solved["delta_weight"])
            model.cells[cell_id].usage_count += 1
            action = "write"
            committed = True
        else:
            if variant == "replay_growth_oracle":
                replay_items = history.items_for_cell(cell_id)
                replay_accesses += len(replay_items)
                old_sample_accesses += len(replay_items)
                old_label_accesses += len(replay_items)
                basis = history.full_basis(cell_id)
            else:
                basis = model.cells[cell_id].basis
            solved = constrained_update(
                transaction.z,
                residual,
                basis,
                numerical_rank_tolerance=cfg.numerical_rank_tolerance,
            )
            feasible = bool(solved["fit_error"] <= cfg.certificate_feasibility_threshold)
            if feasible:
                _apply_delta(model.cells[cell_id], solved["delta_weight"])
                if variant != "replay_growth_oracle":
                    model.cells[cell_id].basis = extend_basis(
                        model.cells[cell_id].basis,
                        transaction.z,
                        tolerance=cfg.numerical_rank_tolerance,
                    )
                model.cells[cell_id].usage_count += 1
                action = "write"
                committed = True
            elif variant == "certificate_no_growth":
                action = "reject"
            else:
                child_id = model.spawn_child(root_id, transaction.context, step)
                child = model.cells[child_id]
                child_solved = constrained_update(
                    transaction.z,
                    transaction.target,
                    _empty_basis(cfg.feature_dim),
                    numerical_rank_tolerance=cfg.numerical_rank_tolerance,
                )
                _apply_delta(child, child_solved["delta_weight"])
                if variant != "replay_growth_oracle":
                    child.basis = extend_basis(
                        child.basis,
                        transaction.z,
                        tolerance=cfg.numerical_rank_tolerance,
                    )
                child.usage_count += 1
                write_cell_id = child_id
                expected_mode_cells[(transaction.root_id, transaction.mode_id)] = child_id
                action = "grow"
                committed = True
                grew = True

        if transaction.phase == "acquisition":
            if committed:
                accepted_acquisition += 1
                post = model.forward(transaction.context, transaction.z)
                post_mse = float(torch.mean((post - transaction.target).square()).item())
                gain = max(0.0, 1.0 - post_mse / max(pre_mse, 1e-30))
                acquisition_gain += gain
            else:
                rejected_acquisition += 1
            if transaction.mode_id > 0 and transaction.block_id == 0 and grew:
                growth_rescues += 1
            if (
                (transaction.mode_id == 0 or transaction.block_id == 1)
                and action == "write"
            ):
                safe_extensions += 1

        after_history_mse = history.behavior_mse(model)
        cumulative_positive_regression += max(0.0, after_history_mse - before_history_mse)
        max_historical_regression = max(max_historical_regression, after_history_mse)

        if committed and transaction.phase == "acquisition":
            history.record(model, transaction)

        if transaction.phase == "tail" and grew:
            tail_spawns += 1

        action_log.append(
            {
                "index": transaction.index,
                "phase": transaction.phase,
                "root": transaction.root_id,
                "mode": transaction.mode_id,
                "block": transaction.block_id,
                "action": action,
                "routed_root": root_id,
                "routed_cell": cell_id,
                "write_cell": write_cell_id,
                "pre_mse": pre_mse,
                "fit_error": float(solved["fit_error"]) if solved is not None else 0.0,
                "cells": model.cell_count,
            }
        )

    final_history_mse = history.behavior_mse(model)
    route = _route_metrics(model, world, expected_mode_cells)
    behavior = _fresh_behavior_eval(model, world, seed=0xF003 + model.root_count, cfg=cfg)
    lineage_sizes = [len(model.lineages[root_id]) for root_id in range(model.root_count)]
    child_count = model.cell_count - model.root_count
    acquisition_actions = [row["action"] for row in action_log if row["phase"] == "acquisition"]
    tail_actions = [row["action"] for row in action_log if row["phase"] == "tail"]
    return {
        "variant": variant,
        "root_count": model.root_count,
        "final_cells": model.cell_count,
        "child_count": int(child_count),
        "lineage_sizes": lineage_sizes,
        "spawn_steps": list(model.spawn_steps),
        "tail_spawns": int(tail_spawns),
        "acquisition_gain": float(acquisition_gain),
        "accepted_acquisition": int(accepted_acquisition),
        "rejected_acquisition": int(rejected_acquisition),
        "growth_rescues": int(growth_rescues),
        "safe_extensions": int(safe_extensions),
        "historical_items": len(history.items),
        "final_historical_regression_mse": float(final_history_mse),
        "max_historical_regression_mse": float(max_historical_regression),
        "cumulative_positive_historical_regression": float(cumulative_positive_regression),
        "replay_accesses": int(replay_accesses),
        "old_sample_accesses": int(old_sample_accesses),
        "old_label_accesses": int(old_label_accesses),
        "route": route,
        "tail_child_route_accuracy": _mean(tail_child_route_hits),
        "final_behavior": behavior,
        "acquisition_actions": acquisition_actions,
        "tail_action_counts": {
            action: sum(row == action for row in tail_actions)
            for action in ("reuse", "write", "reject", "grow", "root_misroute")
        },
        "write_transaction_to_cell_compression": float(
            (len(world.acquisition) + len(world.tail)) / max(model.cell_count, 1)
        ),
        "action_log": action_log,
    }


def evaluate_gates(result: dict[str, Any]) -> dict[str, bool]:
    bridge = result["structural_bridge"]
    variants = result["variants"]
    unsafe = variants["unsafe"]
    no_growth = variants["certificate_no_growth"]
    certificate = variants["certificate_growth"]
    replay = variants["replay_growth_oracle"]

    cert_actions = certificate["acquisition_actions"]
    replay_actions = replay["acquisition_actions"]
    action_agreement = _mean(
        float(first == second) for first, second in zip(cert_actions, replay_actions)
    )
    expected_conflicts = int(result["protection"]["modes_per_root"] - 1) * int(
        bridge["root_cells"]
    )
    expected_safe = len(result["protection"]["acquisition_blocks"]) * int(
        bridge["root_cells"]
    ) + (int(result["protection"]["modes_per_root"]) - 1) * int(bridge["root_cells"])
    true_functional_cells = int(bridge["root_cells"]) * int(result["protection"]["modes_per_root"])

    replay_gain = max(float(replay["acquisition_gain"]), 1e-12)
    gates = {
        "structural_bridge_valid": int(bridge["root_cells"]) == int(bridge["true_factors"]) == 12
        and int(bridge["covered_factors"]) == 12
        and int(bridge["duplicate_assignments"]) == 0
        and float(bridge["mean_matched_root_key_cosine"]) >= 0.985,
        "pre_protection_root_routing": float(result["pre_protection_root_route_accuracy"]) == 1.0,
        "unsafe_control_forgets": float(unsafe["final_historical_regression_mse"]) >= 1e-4,
        "no_growth_exposes_stability_plasticity_limit": float(no_growth["final_historical_regression_mse"]) <= 1e-10
        and float(no_growth["acquisition_gain"]) <= 0.50 * replay_gain,
        "certificate_growth_zero_replay": int(certificate["replay_accesses"]) == 0
        and int(certificate["old_sample_accesses"]) == 0
        and int(certificate["old_label_accesses"]) == 0,
        "certificate_growth_retention": float(certificate["final_historical_regression_mse"]) <= 1e-10
        and float(certificate["cumulative_positive_historical_regression"]) <= 1e-9,
        "certificate_growth_plasticity": float(certificate["acquisition_gain"]) >= 0.98 * replay_gain,
        "certificate_matches_replay_decisions": action_agreement >= 0.99,
        "growth_rescue": int(certificate["growth_rescues"]) == expected_conflicts
        and int(certificate["safe_extensions"]) == expected_safe,
        "route_stability": float(certificate["route"]["exact_mode_accuracy"]) >= 0.99
        and float(certificate["route"]["root_accuracy"]) == 1.0,
        "child_reuse": float(certificate["tail_child_route_accuracy"]) >= 0.99
        and int(certificate["tail_spawns"]) == 0,
        "bounded_functional_growth": int(certificate["final_cells"]) == true_functional_cells == 36
        and all(int(size) == int(result["protection"]["modes_per_root"]) for size in certificate["lineage_sizes"]),
        "final_behavior_quality": float(certificate["final_behavior"]["mse"]) <= 1e-8
        and float(replay["final_behavior"]["mse"]) <= 1e-8,
        "state_compression": float(certificate["write_transaction_to_cell_compression"]) >= 20.0,
        "replay_oracle_actually_uses_history": int(replay["replay_accesses"]) > 0,
    }
    result["action_agreement"] = float(action_agreement)
    return gates


def run_seed(
    seed: int,
    structural_cfg: StructuralBridgeConfig = StructuralBridgeConfig(),
    protection_cfg: ProtectionConfig = ProtectionConfig(),
) -> dict[str, Any]:
    root_anchors, structural = learn_structural_roots(int(seed), structural_cfg)
    world = build_protection_world(root_anchors, int(seed), protection_cfg)
    root_probe = HierarchicalProtectedModel(
        root_anchors,
        feature_dim=protection_cfg.feature_dim,
        output_dim=protection_cfg.output_dim,
    )
    root_hits = []
    for root_id in range(root_probe.root_count):
        for mode_id in range(protection_cfg.modes_per_root):
            root_hits.append(
                float(root_probe.route_root(world.mode_keys[root_id, mode_id]) == root_id)
            )

    variants = {
        variant: run_variant(variant, world, protection_cfg)
        for variant in VARIANTS
    }
    result: dict[str, Any] = {
        "seed": int(seed),
        "structural_config": asdict(structural_cfg),
        "protection": asdict(protection_cfg),
        "hard_cell_cap": None,
        "structural_bridge": structural,
        "pre_protection_root_route_accuracy": _mean(root_hits),
        "acquisition_transactions": len(world.acquisition),
        "tail_transactions": len(world.tail),
        "total_write_transactions": len(world.acquisition) + len(world.tail),
        "true_functional_cells": int(root_probe.root_count * protection_cfg.modes_per_root),
        "variants": variants,
    }
    result["gates"] = evaluate_gates(result)
    result["pass"] = all(result["gates"].values())
    return result


def protection_only_smoke(seed: int = 401) -> dict[str, Any]:
    """Fast invariant smoke that does not invoke the CLM-002 structural bridge."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    roots_raw = torch.randn(6, 24, generator=generator, dtype=_DTYPE)
    roots, _ = torch.linalg.qr(roots_raw.T, mode="reduced")
    root_anchors = roots[:, :6].T.contiguous()
    cfg = ProtectionConfig(tail_reuse_transactions=72)
    world = build_protection_world(root_anchors, int(seed), cfg)
    variants = {
        variant: run_variant(variant, world, cfg)
        for variant in VARIANTS
    }
    return {
        "roots": 6,
        "true_functional_cells": 18,
        "variants": variants,
    }
