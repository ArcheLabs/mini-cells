"""Native CLM v0 M3L-1: historical address-state capacity curve diagnostic.

Checkpoint-only. The module reuses the exact M3L temporal-lineage ownership and
sequence-group-heldout samples, then sweeps diagonal/low-rank/full-covariance
Gaussian address states against the same offline linear oracle. Native CLM
parameters are never updated and no new continual-learning seed is consumed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .native_clm_m2 import sha256_file
from .native_clm_m3l_gate import (
    M3LQuerySketchConfig,
    LowRankGaussianSketch,
    _balanced_concat,
    _collect_domain_queries,
    _cosine_auc,
    _edge_ownership_metadata,
    _fit_oracle,
    _group_split,
    _normalized_oracle_recovery,
    _parameter_state_sha256,
    _score_gate,
    derive_sketch_gate,
    fit_low_rank_sketch,
)
from .native_clm_m3r import LineageNativeCLM
from .native_clm_m3r_address_diag import _auc

_NORMAL_90 = 1.2815515655446004


@dataclass(frozen=True)
class M3L1CapacityConfig:
    max_batches_per_domain: int = 64
    batch_size: int = 8
    max_samples_per_domain_per_edge: int = 4096
    minimum_train_samples_per_side: int = 512
    minimum_test_samples_per_side: int = 256
    train_group_fraction: float = 0.7
    split_seed_base: int = 73800
    ranks: tuple[int, ...] = (0, 8, 16, 32, 64, 128)
    diagonal_regularization: float = 1e-4
    target_old_fpr: float = 0.1
    oracle_steps: int = 300
    oracle_learning_rate: float = 0.05
    oracle_weight_decay: float = 1e-4
    minimum_valid_edge_fraction: float = 0.75
    oracle_separable_median_auc: float = 0.85
    oracle_edge_auc_floor: float = 0.80
    minimum_fraction_oracle_edges_above_floor: float = 0.75
    candidate_median_auc: float = 0.90
    candidate_edge_auc_floor: float = 0.85
    minimum_fraction_candidate_edges_above_floor: float = 0.75
    median_normalized_oracle_excess_recovery: float = 0.85
    median_old_fpr_max: float = 0.20
    median_current_tpr_min: float = 0.70

    def validate(self) -> None:
        if self.max_batches_per_domain < 1 or self.batch_size < 1:
            raise ValueError("sampling budget must be positive")
        if not self.ranks or tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("capacity ranks must be unique and sorted")
        if self.ranks[0] != 0 or any(rank < 0 for rank in self.ranks):
            raise ValueError("capacity grid must start at diagonal rank 0")
        if self.diagonal_regularization <= 0:
            raise ValueError("diagonal regularization must be positive")
        if abs(self.target_old_fpr - 0.1) > 1e-12:
            raise ValueError("registered M3L-1 supports target_old_fpr=0.1 only")
        if not 0.5 < self.train_group_fraction < 0.95:
            raise ValueError("train_group_fraction must be in (0.5, 0.95)")

    def oracle_config(self) -> M3LQuerySketchConfig:
        """Return the exact M3L sampling/oracle configuration for the shared oracle."""

        return M3LQuerySketchConfig(
            max_batches_per_domain=self.max_batches_per_domain,
            batch_size=self.batch_size,
            max_samples_per_domain_per_edge=self.max_samples_per_domain_per_edge,
            minimum_train_samples_per_side=self.minimum_train_samples_per_side,
            minimum_test_samples_per_side=self.minimum_test_samples_per_side,
            train_group_fraction=self.train_group_fraction,
            split_seed_base=self.split_seed_base,
            sketch_rank=16,
            diagonal_regularization=self.diagonal_regularization,
            target_sketch_old_fpr=self.target_old_fpr,
            oracle_steps=self.oracle_steps,
            oracle_learning_rate=self.oracle_learning_rate,
            oracle_weight_decay=self.oracle_weight_decay,
            minimum_valid_edge_fraction=self.minimum_valid_edge_fraction,
            oracle_separable_median_auc=self.oracle_separable_median_auc,
            oracle_edge_auc_floor=self.oracle_edge_auc_floor,
            minimum_fraction_oracle_edges_above_floor=(
                self.minimum_fraction_oracle_edges_above_floor
            ),
            sketch_gate_median_auc=self.candidate_median_auc,
            sketch_gate_edge_auc_floor=self.candidate_edge_auc_floor,
            minimum_fraction_sketch_edges_above_floor=(
                self.minimum_fraction_candidate_edges_above_floor
            ),
            median_normalized_oracle_excess_recovery=(
                self.median_normalized_oracle_excess_recovery
            ),
            median_old_fpr_max=self.median_old_fpr_max,
            median_current_tpr_min=self.median_current_tpr_min,
        )


@dataclass
class FullGaussianState:
    count: int
    mean: Tensor
    covariance: Tensor

    @property
    def storage_bytes(self) -> int:
        return int(8 + 4 * (self.mean.numel() + self.covariance.numel()))


def _fit_rank0_sketch(
    queries: Tensor,
    *,
    diagonal_regularization: float,
    device: torch.device,
) -> LowRankGaussianSketch:
    if queries.ndim != 2 or queries.size(0) < 2:
        raise ValueError("rank-0 sketch requires a 2-D sample matrix")
    x = F.normalize(queries.float().to(device), dim=-1)
    mean = x.mean(dim=0)
    centered = x - mean
    width = x.size(1)
    return LowRankGaussianSketch(
        count=int(x.size(0)),
        mean=mean.detach(),
        basis=torch.empty((width, 0), device=device, dtype=x.dtype),
        eigenvalues=torch.empty((0,), device=device, dtype=x.dtype),
        residual_variance=(
            centered.square().mean(dim=0).clamp_min(diagonal_regularization).detach()
        ),
    )


def _fit_capacity_sketch(
    queries: Tensor,
    *,
    rank: int,
    diagonal_regularization: float,
    device: torch.device,
) -> LowRankGaussianSketch:
    if rank == 0:
        return _fit_rank0_sketch(
            queries,
            diagonal_regularization=diagonal_regularization,
            device=device,
        )
    return fit_low_rank_sketch(
        queries,
        rank=rank,
        diagonal_regularization=diagonal_regularization,
        device=device,
    )


def _derive_rank0_gate(
    old: LowRankGaussianSketch,
    current: LowRankGaussianSketch,
    *,
    diagonal_regularization: float,
) -> dict[str, Tensor | float]:
    diagonal = (
        0.5 * (old.residual_variance + current.residual_variance)
        + diagonal_regularization
    )
    weight = (current.mean - old.mean) / diagonal
    bias = -0.5 * torch.dot(weight, current.mean + old.mean)
    old_score_mean = torch.dot(weight, old.mean) + bias
    old_score_variance = torch.dot(
        weight,
        old.residual_variance * weight,
    ).clamp_min(1e-12)
    threshold = old_score_mean + _NORMAL_90 * torch.sqrt(old_score_variance)
    return {
        "weight": weight.detach(),
        "bias": float(bias.detach().cpu()),
        "threshold": float(threshold.detach().cpu()),
    }


def _derive_capacity_sketch_gate(
    old: LowRankGaussianSketch,
    current: LowRankGaussianSketch,
    *,
    diagonal_regularization: float,
    target_old_fpr: float,
) -> dict[str, Tensor | float]:
    if old.rank == 0 and current.rank == 0:
        if abs(target_old_fpr - 0.1) > 1e-12:
            raise ValueError("rank-0 registered threshold requires target_old_fpr=0.1")
        return _derive_rank0_gate(
            old,
            current,
            diagonal_regularization=diagonal_regularization,
        )
    return derive_sketch_gate(
        old,
        current,
        diagonal_regularization=diagonal_regularization,
        target_old_fpr=target_old_fpr,
    )


def fit_full_gaussian_state(
    queries: Tensor,
    *,
    diagonal_regularization: float,
    device: torch.device,
) -> FullGaussianState:
    if queries.ndim != 2 or queries.size(0) < 2:
        raise ValueError("full Gaussian state requires a 2-D sample matrix")
    x = F.normalize(queries.float().to(device), dim=-1)
    mean = x.mean(dim=0)
    centered = x - mean
    covariance = centered.transpose(0, 1).matmul(centered) / max(1, x.size(0) - 1)
    covariance = covariance + diagonal_regularization * torch.eye(
        x.size(1), device=device, dtype=x.dtype
    )
    return FullGaussianState(
        count=int(x.size(0)),
        mean=mean.detach(),
        covariance=covariance.detach(),
    )


def derive_full_covariance_gate(
    old: FullGaussianState,
    current: FullGaussianState,
    *,
    diagonal_regularization: float,
    target_old_fpr: float,
) -> dict[str, Tensor | float]:
    if abs(target_old_fpr - 0.1) > 1e-12:
        raise ValueError("registered full-covariance threshold requires target_old_fpr=0.1")
    width = old.mean.numel()
    pooled = 0.5 * (old.covariance + current.covariance)
    pooled = pooled + diagonal_regularization * torch.eye(
        width,
        device=pooled.device,
        dtype=pooled.dtype,
    )
    delta = current.mean - old.mean
    weight = torch.linalg.solve(pooled, delta)
    bias = -0.5 * torch.dot(weight, current.mean + old.mean)
    old_score_mean = torch.dot(weight, old.mean) + bias
    old_score_variance = torch.dot(
        weight,
        old.covariance.matmul(weight),
    ).clamp_min(1e-12)
    threshold = old_score_mean + _NORMAL_90 * torch.sqrt(old_score_variance)
    return {
        "weight": weight.detach(),
        "bias": float(bias.detach().cpu()),
        "threshold": float(threshold.detach().cpu()),
    }


def _evaluate_gate(
    old_test: Tensor,
    current_test: Tensor,
    gate: dict[str, Tensor | float],
    *,
    oracle_auc: float,
    device: torch.device,
) -> dict[str, float]:
    old_scores = _score_gate(old_test, gate, device)
    current_scores = _score_gate(current_test, gate, device)
    labels = torch.cat(
        [
            torch.zeros(old_scores.size(0), device=device),
            torch.ones(current_scores.size(0), device=device),
        ]
    )
    scores = torch.cat([old_scores, current_scores], dim=0)
    threshold = float(gate["threshold"])
    auc = float(_auc(scores, labels))
    return {
        "auc": auc,
        "old_fpr": float((old_scores > threshold).float().mean().cpu()),
        "current_tpr": float((current_scores > threshold).float().mean().cpu()),
        "normalized_oracle_excess_recovery": _normalized_oracle_recovery(
            auc, oracle_auc
        ),
    }


def _transition_label(edge: dict[str, Any]) -> str:
    return f"{'+'.join(str(value) for value in edge['old_domains'])}->{edge['current_domain']}"


def diagnose_m3l1_seed(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    eval_paths: dict[str, str | Path],
    output_path: str | Path,
    seed: int,
    config: M3L1CapacityConfig,
    device: str,
) -> dict[str, Any]:
    config.validate()
    checkpoint = Path(checkpoint_path)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"M3L-1 checkpoint SHA mismatch for seed {seed}: "
            f"expected {expected_checkpoint_sha256}, got {actual_sha}"
        )
    target_device = torch.device(device)
    model, extra = LineageNativeCLM.load_checkpoint(checkpoint, map_location="cpu")
    if int(extra.get("seed", -1)) != int(seed) or extra.get("arm") != "lineage_growth":
        raise RuntimeError("checkpoint metadata does not match requested M3R lineage seed")
    growth_events = list(extra.get("growth_events", []))
    if not growth_events:
        raise RuntimeError("lineage checkpoint has no growth events")
    model.to(target_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_hash_before = _parameter_state_sha256(model)
    edges = _edge_ownership_metadata(model, growth_events)
    oracle_config = config.oracle_config()

    by_domain: dict[str, dict[int, dict[str, Tensor]]] = {}
    for domain_index, domain in enumerate(("A", "B", "C", "D")):
        if domain not in eval_paths:
            raise ValueError(f"missing evaluation path for domain {domain}")
        by_domain[domain] = _collect_domain_queries(
            model,
            edges,
            domain,
            eval_paths[domain],
            device=target_device,
            config=oracle_config,
            seq_len=model.config.max_seq_len,
            seed=config.split_seed_base + seed + 1000 * domain_index,
        )

    edge_results: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(edges):
        child_id = int(edge["child_id"])
        old_train_parts: list[Tensor] = []
        old_test_parts: list[Tensor] = []
        per_old_domain: dict[str, dict[str, int]] = {}
        for old_index, domain in enumerate(edge["old_domains"]):
            packed = by_domain[domain].get(child_id)
            if packed is None:
                train = torch.empty((0, model.config.d_model))
                test = torch.empty((0, model.config.d_model))
            else:
                train, test = _group_split(
                    packed["query"],
                    packed["group"],
                    train_fraction=config.train_group_fraction,
                    seed=config.split_seed_base + seed + edge_index * 101 + old_index,
                )
            old_train_parts.append(train)
            old_test_parts.append(test)
            per_old_domain[str(domain)] = {
                "train": int(train.size(0)),
                "test": int(test.size(0)),
            }

        old_train = _balanced_concat(old_train_parts)
        old_test = _balanced_concat(old_test_parts)
        current_domain = str(edge["current_domain"])
        current_packed = by_domain[current_domain].get(child_id)
        if current_packed is None:
            current_train = torch.empty((0, model.config.d_model))
            current_test = torch.empty((0, model.config.d_model))
        else:
            current_train, current_test = _group_split(
                current_packed["query"],
                current_packed["group"],
                train_fraction=config.train_group_fraction,
                seed=config.split_seed_base + seed + edge_index * 101 + 97,
            )

        valid = bool(
            old_train.size(0) >= config.minimum_train_samples_per_side
            and current_train.size(0) >= config.minimum_train_samples_per_side
            and old_test.size(0) >= config.minimum_test_samples_per_side
            and current_test.size(0) >= config.minimum_test_samples_per_side
        )
        result: dict[str, Any] = {
            **edge,
            "seed": int(seed),
            "transition": _transition_label(edge),
            "old_domain_counts": per_old_domain,
            "old_train_samples": int(old_train.size(0)),
            "old_test_samples": int(old_test.size(0)),
            "current_train_samples": int(current_train.size(0)),
            "current_test_samples": int(current_test.size(0)),
            "valid": valid,
            "current_cosine_auc": None,
            "offline_oracle_auc": None,
            "offline_oracle_accuracy": None,
            "candidates": [],
        }
        if valid:
            oracle = _fit_oracle(
                old_train,
                current_train,
                old_test,
                current_test,
                config=oracle_config,
                seed=config.split_seed_base + seed + edge_index * 211,
                device=target_device,
            )
            oracle_auc = float(oracle["auc"])
            result["offline_oracle_auc"] = oracle_auc
            result["offline_oracle_accuracy"] = float(oracle["accuracy"])
            result["current_cosine_auc"] = _cosine_auc(
                model,
                int(edge["parent_id"]),
                child_id,
                old_test,
                current_test,
            )

            for rank in config.ranks:
                old_sketch = _fit_capacity_sketch(
                    old_train,
                    rank=rank,
                    diagonal_regularization=config.diagonal_regularization,
                    device=target_device,
                )
                current_sketch = _fit_capacity_sketch(
                    current_train,
                    rank=rank,
                    diagonal_regularization=config.diagonal_regularization,
                    device=target_device,
                )
                gate = _derive_capacity_sketch_gate(
                    old_sketch,
                    current_sketch,
                    diagonal_regularization=config.diagonal_regularization,
                    target_old_fpr=config.target_old_fpr,
                )
                metrics = _evaluate_gate(
                    old_test,
                    current_test,
                    gate,
                    oracle_auc=oracle_auc,
                    device=target_device,
                )
                result["candidates"].append(
                    {
                        "candidate": f"rank-{rank}",
                        "family": "low_rank_gaussian",
                        "rank": int(rank),
                        "historical_address_state_bytes": old_sketch.storage_bytes,
                        **metrics,
                    }
                )

            old_full = fit_full_gaussian_state(
                old_train,
                diagonal_regularization=config.diagonal_regularization,
                device=target_device,
            )
            current_full = fit_full_gaussian_state(
                current_train,
                diagonal_regularization=config.diagonal_regularization,
                device=target_device,
            )
            full_gate = derive_full_covariance_gate(
                old_full,
                current_full,
                diagonal_regularization=config.diagonal_regularization,
                target_old_fpr=config.target_old_fpr,
            )
            full_metrics = _evaluate_gate(
                old_test,
                current_test,
                full_gate,
                oracle_auc=oracle_auc,
                device=target_device,
            )
            result["candidates"].append(
                {
                    "candidate": "full-covariance",
                    "family": "full_covariance_gaussian",
                    "rank": None,
                    "historical_address_state_bytes": old_full.storage_bytes,
                    **full_metrics,
                }
            )
        edge_results.append(result)

    parameter_hash_after = _parameter_state_sha256(model)
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("M3L-1 diagnostic mutated Native CLM trainable parameters")
    valid_edges = [edge for edge in edge_results if edge["valid"]]
    summary = {
        "format": "minicells.native-clm-v0.m3l1-address-state-capacity.seed.v1",
        "seed": int(seed),
        "checkpoint_sha256": actual_sha,
        "checkpoint_arm": "lineage_growth",
        "native_clm_training": False,
        "new_formal_seeds_consumed": False,
        "parameter_state_sha256_before": parameter_hash_before,
        "parameter_state_sha256_after": parameter_hash_after,
        "edge_count": len(edge_results),
        "valid_edge_count": len(valid_edges),
        "valid_edge_fraction": float(len(valid_edges) / max(1, len(edge_results))),
        "edges": edge_results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": float("nan"), "mean": float("nan")}
    return {
        "median": float(median(values)),
        "mean": float(sum(values) / len(values)),
    }


def _candidate_summary(
    rows: list[dict[str, Any]],
    *,
    config: M3L1CapacityConfig,
) -> dict[str, Any]:
    auc = [float(row["auc"]) for row in rows]
    old_fpr = [float(row["old_fpr"]) for row in rows]
    current_tpr = [float(row["current_tpr"]) for row in rows]
    recovery = [float(row["normalized_oracle_excess_recovery"]) for row in rows]
    storage = [float(row["historical_address_state_bytes"]) for row in rows]
    auc_summary = _summarize(auc)
    fraction_floor = (
        float(sum(value >= config.candidate_edge_auc_floor for value in auc) / len(auc))
        if auc
        else 0.0
    )
    passes = bool(
        auc
        and auc_summary["median"] >= config.candidate_median_auc
        and fraction_floor >= config.minimum_fraction_candidate_edges_above_floor
        and median(recovery) >= config.median_normalized_oracle_excess_recovery
        and median(old_fpr) <= config.median_old_fpr_max
        and median(current_tpr) >= config.median_current_tpr_min
    )
    return {
        **auc_summary,
        "fraction_auc_ge_floor": fraction_floor,
        "median_old_fpr": float(median(old_fpr)) if old_fpr else float("nan"),
        "median_current_tpr": (
            float(median(current_tpr)) if current_tpr else float("nan")
        ),
        "median_normalized_oracle_excess_recovery": (
            float(median(recovery)) if recovery else float("nan")
        ),
        "median_historical_address_state_bytes": (
            float(median(storage)) if storage else float("nan")
        ),
        "passes_m3l_feasibility_gates": passes,
    }


def aggregate_m3l1_capacity(
    seed_summaries: list[dict[str, Any]],
    *,
    config: M3L1CapacityConfig,
    parent_m3l_commit: str,
    parent_m3r_hf_revision: str,
) -> dict[str, Any]:
    config.validate()
    all_edges = [edge for summary in seed_summaries for edge in summary["edges"]]
    valid_edges = [edge for edge in all_edges if edge["valid"]]
    valid_fraction = float(len(valid_edges) / max(1, len(all_edges)))
    oracle = [float(edge["offline_oracle_auc"]) for edge in valid_edges]
    oracle_summary = _summarize(oracle)
    oracle_fraction = (
        float(sum(value >= config.oracle_edge_auc_floor for value in oracle) / len(oracle))
        if oracle
        else 0.0
    )
    oracle_separable = bool(
        oracle
        and oracle_summary["median"] >= config.oracle_separable_median_auc
        and oracle_fraction >= config.minimum_fraction_oracle_edges_above_floor
    )

    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    for edge in valid_edges:
        for candidate in edge["candidates"]:
            candidate_rows.setdefault(str(candidate["candidate"]), []).append(candidate)
    candidate_summaries = {
        label: _candidate_summary(rows, config=config)
        for label, rows in sorted(candidate_rows.items())
    }

    transition_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for edge in valid_edges:
        transition = str(edge["transition"])
        transition_summaries.setdefault(transition, {})
        for candidate in edge["candidates"]:
            label = str(candidate["candidate"])
            transition_summaries[transition].setdefault(label, {"rows": []})["rows"].append(candidate)
    transition_output: dict[str, dict[str, Any]] = {}
    for transition, candidates in sorted(transition_summaries.items()):
        transition_output[transition] = {
            label: _candidate_summary(value["rows"], config=config)
            for label, value in sorted(candidates.items())
        }

    passing_low_ranks = [
        rank
        for rank in config.ranks
        if candidate_summaries.get(f"rank-{rank}", {}).get(
            "passes_m3l_feasibility_gates", False
        )
    ]
    full_passes = bool(
        candidate_summaries.get("full-covariance", {}).get(
            "passes_m3l_feasibility_gates", False
        )
    )
    if valid_fraction < config.minimum_valid_edge_fraction:
        classification = "INCONCLUSIVE_COVERAGE"
    elif not oracle_separable:
        classification = "ORACLE_NOT_SEPARABLE"
    elif passing_low_ranks:
        classification = "LOW_RANK_CAPACITY_SUFFICIENT"
    elif full_passes:
        classification = "FULL_COVARIANCE_REQUIRED"
    else:
        classification = "GAUSSIAN_FAMILY_LIMITED"

    interpretation = {
        "INCONCLUSIVE_COVERAGE": "Too many lineage edges lack enough matched sequence-group-heldout samples for a reliable address-state capacity decision.",
        "ORACLE_NOT_SEPARABLE": "The stricter matched edge-local oracle is not consistently separable; capacity-family conclusions are not licensed.",
        "LOW_RANK_CAPACITY_SUFFICIENT": "A finite low-rank Gaussian historical address state satisfies the original M3L feasibility gates; the M3L shortfall is primarily rank/capacity limited rather than a Gaussian-family failure.",
        "FULL_COVARIANCE_REQUIRED": "No registered low-rank state through rank 128 passes, but dense full covariance does; the boundary remains second-order Gaussian but requires substantially higher historical address-state capacity.",
        "GAUSSIAN_FAMILY_LIMITED": "Even dense full-covariance Gaussian LDA fails the original M3L feasibility gates while the linear oracle remains separable; improving rank alone is insufficient and the address-state family must become richer or learned.",
    }[classification]

    return {
        "format": "minicells.native-clm-v0.m3l1-address-state-capacity.aggregate.v1",
        "classification": classification,
        "scientific_decision": False,
        "checkpoint_seeds": [int(summary["seed"]) for summary in seed_summaries],
        "parent_m3l_publish_commit": parent_m3l_commit,
        "parent_m3r_hf_revision": parent_m3r_hf_revision,
        "native_clm_training": False,
        "new_formal_seeds_consumed": False,
        "edge_count": len(all_edges),
        "valid_edge_count": len(valid_edges),
        "valid_edge_fraction": valid_fraction,
        "offline_oracle": {
            **oracle_summary,
            "fraction_auc_ge_floor": oracle_fraction,
        },
        "capacity_curve": candidate_summaries,
        "transition_capacity_curves": transition_output,
        "minimum_passing_low_rank": min(passing_low_ranks) if passing_low_ranks else None,
        "full_covariance_passes": full_passes,
        "thresholds": {
            "minimum_valid_edge_fraction": config.minimum_valid_edge_fraction,
            "oracle_separable_median_auc": config.oracle_separable_median_auc,
            "oracle_edge_auc_floor": config.oracle_edge_auc_floor,
            "minimum_fraction_oracle_edges_above_floor": (
                config.minimum_fraction_oracle_edges_above_floor
            ),
            "candidate_median_auc": config.candidate_median_auc,
            "candidate_edge_auc_floor": config.candidate_edge_auc_floor,
            "minimum_fraction_candidate_edges_above_floor": (
                config.minimum_fraction_candidate_edges_above_floor
            ),
            "median_normalized_oracle_excess_recovery": (
                config.median_normalized_oracle_excess_recovery
            ),
            "median_old_fpr_max": config.median_old_fpr_max,
            "median_current_tpr_min": config.median_current_tpr_min,
        },
        "interpretation": interpretation,
    }


def write_m3l1_capacity_csv(seed_summaries: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "parent_id",
        "child_id",
        "root_id",
        "transition",
        "candidate",
        "family",
        "rank",
        "offline_oracle_auc",
        "current_cosine_auc",
        "auc",
        "normalized_oracle_excess_recovery",
        "old_fpr",
        "current_tpr",
        "historical_address_state_bytes",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in seed_summaries:
            for edge in summary["edges"]:
                if not edge["valid"]:
                    continue
                for candidate in edge["candidates"]:
                    writer.writerow(
                        {
                            "seed": edge["seed"],
                            "parent_id": edge["parent_id"],
                            "child_id": edge["child_id"],
                            "root_id": edge["root_id"],
                            "transition": edge["transition"],
                            "candidate": candidate["candidate"],
                            "family": candidate["family"],
                            "rank": "" if candidate["rank"] is None else candidate["rank"],
                            "offline_oracle_auc": edge["offline_oracle_auc"],
                            "current_cosine_auc": edge["current_cosine_auc"],
                            "auc": candidate["auc"],
                            "normalized_oracle_excess_recovery": candidate[
                                "normalized_oracle_excess_recovery"
                            ],
                            "old_fpr": candidate["old_fpr"],
                            "current_tpr": candidate["current_tpr"],
                            "historical_address_state_bytes": candidate[
                                "historical_address_state_bytes"
                            ],
                        }
                    )
