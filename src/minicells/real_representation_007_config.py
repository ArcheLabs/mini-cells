"""Frozen configuration for Core Validation 007."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .real_representation_006_config import CoreValidation006Config


@dataclass(frozen=True)
class CoreValidation007Config:
    base: CoreValidation006Config
    boundary_candidates: tuple[str, ...]
    maximum_modes_per_address: int
    maximum_write_rank: int
    mode_creation_cosine_threshold: float
    split_conflict_threshold: float
    maximum_splits_per_transaction: int
    deploy_soft_top2_temperature: float
    discovery_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    smoke_seed: int
    minimum_discovery_routing_agreement: float
    minimum_confirmation_routing_agreement: float
    maximum_confirmation_deploy_nll_gap: float
    minimum_confirmation_split_conflict_reduction: float
    maximum_confirmation_spawned_fraction_of_addresses: float
    maximum_confirmation_regression_ratio_vs_unsafe: float
    minimum_confirmation_gain_ratio_vs_replay: float
    minimum_confirmation_child_reuse_transactions: int
    selection_conflict_weight: float
    selection_routing_weight: float
    selection_balance_weight: float

    @classmethod
    def from_protocol(cls, path: str | Path) -> "CoreValidation007Config":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        # Core 007 intentionally pins the same foundation/data geometry as 006.
        b = p["base_protocol"]
        base = CoreValidation006Config(
            model_id=str(p["foundation"]["model_id"]),
            model_revision=str(p["foundation"]["model_revision"]),
            dataset_id=str(p["data"]["dataset_id"]),
            dataset_revision=str(p["data"]["dataset_revision"]),
            dataset_split=str(p["data"]["split"]),
            sources=tuple(str(x) for x in p["data"]["sources"]),
            sequence_length=int(p["data"]["sequence_length"]),
            router_bootstrap_sequences_per_source=int(p["data"]["router_bootstrap_sequences_per_source"]),
            train_sequences_per_source=int(p["data"]["train_sequences_per_source"]),
            eval_sequences_per_source=int(p["data"]["eval_sequences_per_source"]),
            sequences_per_transaction=int(p["continual_stream"]["sequences_per_transaction"]),
            addresses=int(p["router"]["addresses"]),
            base_cells=int(p["router"]["base_cells"]),
            kmeans_iterations=int(p["router"]["kmeans_iterations"]),
            cell_dim=int(p["cells"]["cell_dim"]),
            certificate_energy=float(p["cells"]["certificate_energy"]),
            ridge=float(p["cells"]["ridge"]),
            functional_step=float(p["cells"]["functional_step"]),
            maximum_delta_norm=float(p["cells"]["maximum_delta_norm"]),
            split_conflict_threshold=float(b["core006_split_conflict_threshold"]),
            maximum_splits_per_transaction=int(b["core006_maximum_splits_per_transaction"]),
            replay_buffer_sequences=int(p["baselines"]["replay_buffer_sequences"]),
            replay_sequences_per_transaction=int(p["baselines"]["replay_sequences_per_transaction"]),
            retention_checkpoint_every_transactions=int(p["evaluation"]["retention_checkpoint_every_transactions"]),
            minimum_midstream_reuse_ratio=float(b["core006_minimum_midstream_reuse_ratio"]),
            maximum_midstream_energy_rank_fraction=float(b["core006_maximum_midstream_energy_rank_fraction"]),
            maximum_registered_regression_ratio_vs_unsafe=float(b["core006_maximum_registered_regression_ratio_vs_unsafe"]),
            minimum_gain_ratio_vs_replay=float(b["core006_minimum_gain_ratio_vs_replay"]),
            minimum_split_conflict_reduction=float(b["core006_minimum_split_conflict_reduction"]),
            maximum_spawned_fraction_of_addresses=float(b["core006_maximum_spawned_fraction_of_addresses"]),
            minimum_child_reuse_transactions=int(b["core006_minimum_child_reuse_transactions"]),
            formal_seeds=tuple(int(x) for x in p["replication"]["confirmation_seeds"]),
            smoke_seed=int(p["replication"]["smoke_seed"]),
        )
        base.validate()
        cfg = cls(
            base=base,
            boundary_candidates=tuple(str(x) for x in p["functional_modes"]["boundary_candidates"]),
            maximum_modes_per_address=int(p["functional_modes"]["maximum_modes_per_address"]),
            maximum_write_rank=int(p["functional_modes"]["maximum_write_rank"]),
            mode_creation_cosine_threshold=float(p["functional_modes"]["creation_cosine_threshold"]),
            split_conflict_threshold=float(p["mitosis"]["split_conflict_threshold"]),
            maximum_splits_per_transaction=int(p["mitosis"]["maximum_splits_per_transaction"]),
            deploy_soft_top2_temperature=float(p["routing"]["soft_top2_temperature"]),
            discovery_seeds=tuple(int(x) for x in p["replication"]["discovery_seeds"]),
            confirmation_seeds=tuple(int(x) for x in p["replication"]["confirmation_seeds"]),
            smoke_seed=int(p["replication"]["smoke_seed"]),
            minimum_discovery_routing_agreement=float(p["discovery"]["minimum_routing_agreement"]),
            minimum_confirmation_routing_agreement=float(p["confirmation_gates"]["minimum_routing_agreement"]),
            maximum_confirmation_deploy_nll_gap=float(p["confirmation_gates"]["maximum_deploy_nll_gap"]),
            minimum_confirmation_split_conflict_reduction=float(p["confirmation_gates"]["minimum_split_conflict_reduction"]),
            maximum_confirmation_spawned_fraction_of_addresses=float(p["confirmation_gates"]["maximum_spawned_fraction_of_addresses"]),
            maximum_confirmation_regression_ratio_vs_unsafe=float(p["confirmation_gates"]["maximum_regression_ratio_vs_unsafe"]),
            minimum_confirmation_gain_ratio_vs_replay=float(p["confirmation_gates"]["minimum_gain_ratio_vs_replay"]),
            minimum_confirmation_child_reuse_transactions=int(p["confirmation_gates"]["minimum_child_reuse_transactions"]),
            selection_conflict_weight=float(p["discovery"]["selection_weights"]["interference_cut"]),
            selection_routing_weight=float(p["discovery"]["selection_weights"]["routing_agreement"]),
            selection_balance_weight=float(p["discovery"]["selection_weights"]["balance"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        expected = {
            "semantic_singleton",
            "activation_community",
            "write_community",
            "interference_cut",
        }
        if set(self.boundary_candidates) != expected:
            raise ValueError(f"Core 007 boundary candidates must be exactly {sorted(expected)}")
        if self.maximum_modes_per_address < 1:
            raise ValueError("maximum_modes_per_address must be positive")
        if self.maximum_write_rank < 1:
            raise ValueError("maximum_write_rank must be positive")
        if not -1.0 <= self.mode_creation_cosine_threshold <= 1.0:
            raise ValueError("mode creation cosine threshold must be in [-1,1]")
        if not self.discovery_seeds or not self.confirmation_seeds:
            raise ValueError("discovery and confirmation seeds are required")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("discovery and confirmation seeds must be disjoint")


def smoke_config(cfg: CoreValidation007Config) -> CoreValidation007Config:
    b = cfg.base
    b = replace(
        b,
        sources=b.sources[:2],
        sequence_length=min(b.sequence_length, 32),
        router_bootstrap_sequences_per_source=4,
        train_sequences_per_source=4,
        eval_sequences_per_source=2,
        sequences_per_transaction=2,
        addresses=4,
        base_cells=2,
        cell_dim=min(b.cell_dim, 8),
        replay_buffer_sequences=4,
        replay_sequences_per_transaction=2,
        retention_checkpoint_every_transactions=1,
    )
    return replace(
        cfg,
        base=b,
        maximum_modes_per_address=min(cfg.maximum_modes_per_address, 2),
        maximum_write_rank=min(cfg.maximum_write_rank, 4),
    )
