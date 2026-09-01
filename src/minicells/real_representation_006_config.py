"""Frozen configuration for Core Validation 006."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class CoreValidation006Config:
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    sources: tuple[str, ...]
    sequence_length: int
    router_bootstrap_sequences_per_source: int
    train_sequences_per_source: int
    eval_sequences_per_source: int
    sequences_per_transaction: int
    addresses: int
    base_cells: int
    kmeans_iterations: int
    cell_dim: int
    certificate_energy: float
    ridge: float
    functional_step: float
    maximum_delta_norm: float
    split_conflict_threshold: float
    maximum_splits_per_transaction: int
    replay_buffer_sequences: int
    replay_sequences_per_transaction: int
    retention_checkpoint_every_transactions: int
    minimum_midstream_reuse_ratio: float
    maximum_midstream_energy_rank_fraction: float
    maximum_registered_regression_ratio_vs_unsafe: float
    minimum_gain_ratio_vs_replay: float
    minimum_split_conflict_reduction: float
    maximum_spawned_fraction_of_addresses: float
    minimum_child_reuse_transactions: int
    formal_seeds: tuple[int, ...]
    smoke_seed: int

    @classmethod
    def from_protocol(cls, path: str | Path) -> "CoreValidation006Config":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(
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
            split_conflict_threshold=float(p["mitosis"]["split_conflict_threshold"]),
            maximum_splits_per_transaction=int(p["mitosis"]["maximum_splits_per_transaction"]),
            replay_buffer_sequences=int(p["baselines"]["replay_buffer_sequences"]),
            replay_sequences_per_transaction=int(p["baselines"]["replay_sequences_per_transaction"]),
            retention_checkpoint_every_transactions=int(p["evaluation"]["retention_checkpoint_every_transactions"]),
            minimum_midstream_reuse_ratio=float(p["gates"]["minimum_midstream_reuse_ratio"]),
            maximum_midstream_energy_rank_fraction=float(p["gates"]["maximum_midstream_energy_rank_fraction"]),
            maximum_registered_regression_ratio_vs_unsafe=float(p["gates"]["maximum_registered_regression_ratio_vs_unsafe"]),
            minimum_gain_ratio_vs_replay=float(p["gates"]["minimum_gain_ratio_vs_replay"]),
            minimum_split_conflict_reduction=float(p["gates"]["minimum_split_conflict_reduction"]),
            maximum_spawned_fraction_of_addresses=float(p["gates"]["maximum_spawned_fraction_of_addresses"]),
            minimum_child_reuse_transactions=int(p["gates"]["minimum_child_reuse_transactions"]),
            formal_seeds=tuple(int(x) for x in p["replication"]["formal_seeds"]),
            smoke_seed=int(p["replication"]["smoke_seed"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.cell_dim <= 0:
            raise ValueError("cell_dim must be positive")
        if self.addresses <= self.base_cells:
            raise ValueError("addresses must exceed base_cells so mitosis can partition routes")
        if self.train_sequences_per_source % self.sequences_per_transaction != 0:
            raise ValueError("train_sequences_per_source must divide into exact transactions")
        if not 0 < self.certificate_energy <= 1:
            raise ValueError("certificate_energy must be in (0, 1]")
        if not self.formal_seeds:
            raise ValueError("formal seeds required")

    @property
    def transactions_per_source(self) -> int:
        return self.train_sequences_per_source // self.sequences_per_transaction

    @property
    def transactions(self) -> int:
        return len(self.sources) * self.transactions_per_source


def smoke_config(cfg: CoreValidation006Config) -> CoreValidation006Config:
    return replace(
        cfg,
        sources=cfg.sources[:2],
        sequence_length=min(cfg.sequence_length, 32),
        router_bootstrap_sequences_per_source=4,
        train_sequences_per_source=4,
        eval_sequences_per_source=2,
        sequences_per_transaction=2,
        addresses=4,
        base_cells=2,
        cell_dim=min(cfg.cell_dim, 8),
        replay_buffer_sequences=4,
        replay_sequences_per_transaction=2,
        retention_checkpoint_every_transactions=1,
    )
