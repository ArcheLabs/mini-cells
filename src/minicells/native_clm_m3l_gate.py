"""Native CLM v0 M3L: replay-free learned lineage-local gate diagnostic.

This module is checkpoint-only. It reconstructs compact historical query sketches,
derives an affine parent-vs-child gate from sketch/current moments, and evaluates the
gate on sequence-group-heldout queries. Native CLM parameters are never updated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .native_clm_m2 import sha256_file
from .native_clm_m3 import _loader
from .native_clm_m3r import LineageNativeCLM
from .native_clm_m3r_address_diag import _auc, _edge_eligible_mask, _edge_metadata

PHASE_ORDER = ("A", "B", "C", "D")
_NORMAL_90 = 1.2815515655446004


@dataclass(frozen=True)
class M3LQuerySketchConfig:
    max_batches_per_domain: int = 64
    batch_size: int = 8
    max_samples_per_domain_per_edge: int = 4096
    minimum_train_samples_per_side: int = 512
    minimum_test_samples_per_side: int = 256
    train_group_fraction: float = 0.7
    split_seed_base: int = 73800
    sketch_rank: int = 16
    diagonal_regularization: float = 1e-4
    target_sketch_old_fpr: float = 0.1
    oracle_steps: int = 300
    oracle_learning_rate: float = 0.05
    oracle_weight_decay: float = 1e-4
    minimum_valid_edge_fraction: float = 0.75
    oracle_separable_median_auc: float = 0.85
    oracle_edge_auc_floor: float = 0.80
    minimum_fraction_oracle_edges_above_floor: float = 0.75
    sketch_gate_median_auc: float = 0.90
    sketch_gate_edge_auc_floor: float = 0.85
    minimum_fraction_sketch_edges_above_floor: float = 0.75
    median_normalized_oracle_excess_recovery: float = 0.85
    median_old_fpr_max: float = 0.20
    median_current_tpr_min: float = 0.70

    def validate(self) -> None:
        if self.max_batches_per_domain < 1 or self.batch_size < 1:
            raise ValueError("sampling budget must be positive")
        if self.max_samples_per_domain_per_edge < self.minimum_train_samples_per_side:
            raise ValueError("sample cap cannot be below minimum train count")
        if self.minimum_test_samples_per_side < 1:
            raise ValueError("minimum test count must be positive")
        if not 0.5 < self.train_group_fraction < 0.95:
            raise ValueError("train_group_fraction must be in (0.5, 0.95)")
        if self.sketch_rank < 1:
            raise ValueError("sketch rank must be positive")
        if self.diagonal_regularization <= 0:
            raise ValueError("diagonal regularization must be positive")
        if not 0.0 < self.target_sketch_old_fpr < 0.5:
            raise ValueError("target old FPR must lie in (0, 0.5)")


@dataclass
class LowRankGaussianSketch:
    count: int
    mean: Tensor
    basis: Tensor
    eigenvalues: Tensor
    residual_variance: Tensor

    @property
    def rank(self) -> int:
        return int(self.basis.size(1))

    @property
    def width(self) -> int:
        return int(self.mean.numel())

    @property
    def storage_bytes(self) -> int:
        float_count = (
            self.mean.numel()
            + self.basis.numel()
            + self.eigenvalues.numel()
            + self.residual_variance.numel()
        )
        return int(8 + 4 * float_count)


def _domain_index(domain: str) -> int:
    try:
        return PHASE_ORDER.index(domain)
    except ValueError as exc:
        raise ValueError(f"unknown M3L domain {domain}") from exc


def _edge_ownership_metadata(
    model: LineageNativeCLM,
    growth_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_edges = _edge_metadata(model, growth_events)
    child_birth = {int(edge["child_id"]): str(edge["birth_domain"]) for edge in base_edges}
    result: list[dict[str, Any]] = []
    for edge in base_edges:
        parent_id = int(edge["parent_id"])
        child_birth_domain = str(edge["birth_domain"])
        parent_birth_domain = (
            "A" if parent_id < model.lineage_root_count else child_birth[parent_id]
        )
        start = _domain_index(parent_birth_domain)
        stop = _domain_index(child_birth_domain)
        if stop <= start:
            raise RuntimeError(
                f"non-forward lineage lifetime for edge {parent_id}->{edge['child_id']}: "
                f"{parent_birth_domain}->{child_birth_domain}"
            )
        old_domains = list(PHASE_ORDER[start:stop])
        result.append(
            {
                **edge,
                "parent_birth_domain": parent_birth_domain,
                "old_domains": old_domains,
                "current_domain": child_birth_domain,
            }
        )
    return result


def _query_route_features(model: LineageNativeCLM, tokens: Tensor) -> dict[str, Tensor]:
    """Extract frozen pre-Cell query geometry without executing/mutating Cells."""

    positions = torch.arange(tokens.size(1), device=tokens.device)
    with torch.no_grad():
        hidden = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
        hidden = model.dropout(hidden)
        for index, block in enumerate(model.blocks):
            hidden = block(hidden)
            if index == model.config.cellular_layer_index:
                details = model._lineage_route_details(hidden)
                return {
                    "query": details["query"].detach(),
                    "root_idx": details["root_idx"].detach(),
                }
    raise RuntimeError("Cellular Layer boundary was not reached")


def _parameter_state_sha256(model: LineageNativeCLM) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0]):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _collect_domain_queries(
    model: LineageNativeCLM,
    edges: list[dict[str, Any]],
    domain: str,
    path: str | Path,
    *,
    device: torch.device,
    config: M3LQuerySketchConfig,
    seq_len: int,
    seed: int,
) -> dict[int, dict[str, Tensor]]:
    relevant = [
        edge
        for edge in edges
        if domain in edge["old_domains"] or domain == edge["current_domain"]
    ]
    stores: dict[int, dict[str, list[Tensor]]] = {
        int(edge["child_id"]): {"query": [], "group": []} for edge in relevant
    }
    counts = {child_id: 0 for child_id in stores}
    if not stores:
        return {}

    loader = _loader(
        path,
        seq_len=seq_len,
        batch_size=config.batch_size,
        seed=seed,
        num_workers=0,
    )
    model.eval()
    sequence_offset = 0
    for batch_idx, (tokens, _) in enumerate(loader):
        if batch_idx >= config.max_batches_per_domain:
            break
        tokens = tokens.to(device)
        features = _query_route_features(model, tokens)
        query = features["query"]
        root_idx = features["root_idx"]
        batch, seq_length, _ = query.shape
        groups = (
            torch.arange(sequence_offset, sequence_offset + batch, device=device)
            .unsqueeze(1)
            .expand(batch, seq_length)
        )
        sequence_offset += batch

        for edge in relevant:
            child_id = int(edge["child_id"])
            remaining = config.max_samples_per_domain_per_edge - counts[child_id]
            if remaining <= 0:
                continue
            mask = _edge_eligible_mask(
                model,
                query,
                root_idx,
                path_to_parent=[int(value) for value in edge["path_to_parent"]],
            )
            if not bool(mask.any()):
                continue
            q_values = query[mask]
            g_values = groups[mask]
            take = min(remaining, q_values.size(0))
            stores[child_id]["query"].append(q_values[:take].detach().float().cpu())
            stores[child_id]["group"].append(g_values[:take].detach().long().cpu())
            counts[child_id] += take

        if counts and all(
            count >= config.max_samples_per_domain_per_edge for count in counts.values()
        ):
            break

    packed: dict[int, dict[str, Tensor]] = {}
    for child_id, pieces in stores.items():
        packed[child_id] = {
            "query": (
                torch.cat(pieces["query"], dim=0)
                if pieces["query"]
                else torch.empty((0, model.config.d_model), dtype=torch.float32)
            ),
            "group": (
                torch.cat(pieces["group"], dim=0)
                if pieces["group"]
                else torch.empty((0,), dtype=torch.long)
            ),
        }
    return packed


def _group_split(
    queries: Tensor,
    groups: Tensor,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if queries.size(0) != groups.numel():
        raise ValueError("query/group size mismatch")
    unique = torch.unique(groups, sorted=True)
    if unique.numel() < 2:
        return queries[:0], queries[:0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = unique[torch.randperm(unique.numel(), generator=generator)]
    train_groups = max(1, min(unique.numel() - 1, math.floor(unique.numel() * train_fraction)))
    train_ids = order[:train_groups]
    test_ids = order[train_groups:]
    train_mask = torch.isin(groups, train_ids)
    test_mask = torch.isin(groups, test_ids)
    return queries[train_mask], queries[test_mask]


def _balanced_concat(parts: list[Tensor]) -> Tensor:
    if not parts or any(part.size(0) == 0 for part in parts):
        width = parts[0].size(1) if parts else 0
        return torch.empty((0, width), dtype=torch.float32)
    take = min(part.size(0) for part in parts)
    return torch.cat([part[:take] for part in parts], dim=0)


def fit_low_rank_sketch(
    queries: Tensor,
    *,
    rank: int,
    diagonal_regularization: float,
    device: torch.device,
) -> LowRankGaussianSketch:
    if queries.ndim != 2 or queries.size(0) < 2:
        raise ValueError("sketch requires a 2-D sample matrix with at least two rows")
    x = F.normalize(queries.float().to(device), dim=-1)
    mean = x.mean(dim=0)
    centered = x - mean
    max_rank = min(rank, centered.size(0) - 1, centered.size(1))
    if max_rank < 1:
        raise ValueError("insufficient samples for low-rank sketch")
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:max_rank].transpose(0, 1).contiguous()
    eigenvalues = singular[:max_rank].square() / max(1, centered.size(0) - 1)
    projected = centered.matmul(basis).matmul(basis.transpose(0, 1))
    residual = centered - projected
    residual_variance = residual.square().mean(dim=0).clamp_min(diagonal_regularization)
    return LowRankGaussianSketch(
        count=int(x.size(0)),
        mean=mean.detach(),
        basis=basis.detach(),
        eigenvalues=eigenvalues.clamp_min(0).detach(),
        residual_variance=residual_variance.detach(),
    )


def sketch_covariance_matvec(sketch: LowRankGaussianSketch, vector: Tensor) -> Tensor:
    vector = vector.to(device=sketch.mean.device, dtype=sketch.mean.dtype)
    low = sketch.basis.transpose(0, 1).matmul(vector)
    low = sketch.basis.matmul(sketch.eigenvalues * low)
    return sketch.residual_variance * vector + low


def _woodbury_solve_pooled(
    old: LowRankGaussianSketch,
    current: LowRankGaussianSketch,
    delta_mean: Tensor,
    *,
    diagonal_regularization: float,
) -> Tensor:
    if old.width != current.width:
        raise ValueError("old/current sketch width mismatch")
    diagonal = (
        0.5 * (old.residual_variance + current.residual_variance)
        + diagonal_regularization
    )
    old_factor = old.basis * torch.sqrt(0.5 * old.eigenvalues).unsqueeze(0)
    current_factor = current.basis * torch.sqrt(0.5 * current.eigenvalues).unsqueeze(0)
    factor = torch.cat([old_factor, current_factor], dim=1)
    inv_diag = diagonal.reciprocal()
    inv_delta = inv_diag * delta_mean
    inv_factor = inv_diag.unsqueeze(1) * factor
    middle = torch.eye(factor.size(1), device=factor.device, dtype=factor.dtype)
    middle = middle + factor.transpose(0, 1).matmul(inv_factor)
    correction = torch.linalg.solve(
        middle,
        factor.transpose(0, 1).matmul(inv_delta),
    )
    return inv_delta - inv_factor.matmul(correction)


def derive_sketch_gate(
    old: LowRankGaussianSketch,
    current: LowRankGaussianSketch,
    *,
    diagonal_regularization: float,
    target_old_fpr: float,
) -> dict[str, Tensor | float]:
    if abs(target_old_fpr - 0.1) > 1e-12:
        raise ValueError("registered M3L v1 currently supports target_old_fpr=0.1 only")
    delta_mean = current.mean - old.mean
    weight = _woodbury_solve_pooled(
        old,
        current,
        delta_mean,
        diagonal_regularization=diagonal_regularization,
    )
    bias = -0.5 * torch.dot(weight, current.mean + old.mean)
    old_score_mean = torch.dot(weight, old.mean) + bias
    old_score_variance = torch.dot(weight, sketch_covariance_matvec(old, weight)).clamp_min(1e-12)
    threshold = old_score_mean + _NORMAL_90 * torch.sqrt(old_score_variance)
    return {
        "weight": weight.detach(),
        "bias": float(bias.detach().cpu()),
        "threshold": float(threshold.detach().cpu()),
        "old_score_mean": float(old_score_mean.detach().cpu()),
        "old_score_std": float(torch.sqrt(old_score_variance).detach().cpu()),
    }


def _score_gate(queries: Tensor, gate: dict[str, Tensor | float], device: torch.device) -> Tensor:
    weight = gate["weight"]
    if not isinstance(weight, Tensor):
        raise TypeError("gate weight is not a Tensor")
    x = F.normalize(queries.float().to(device), dim=-1)
    return x.matmul(weight.to(device)) + float(gate["bias"])


def _fit_oracle(
    old_train: Tensor,
    current_train: Tensor,
    old_test: Tensor,
    current_test: Tensor,
    *,
    config: M3LQuerySketchConfig,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    n_train = min(old_train.size(0), current_train.size(0))
    if n_train < 2:
        return {"auc": float("nan"), "accuracy": float("nan")}
    old = F.normalize(old_train[:n_train].float(), dim=-1)
    current = F.normalize(current_train[:n_train].float(), dim=-1)
    x_train = torch.cat([old, current], dim=0).to(device)
    y_train = torch.cat([torch.zeros(n_train), torch.ones(n_train)], dim=0).to(device)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    x_train = (x_train - mean) / std

    torch.manual_seed(seed)
    probe = nn.Linear(x_train.size(-1), 1).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.oracle_learning_rate,
        weight_decay=config.oracle_weight_decay,
    )
    for _ in range(config.oracle_steps):
        optimizer.zero_grad(set_to_none=True)
        scores = probe(x_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(scores, y_train)
        loss.backward()
        optimizer.step()

    old_eval = (F.normalize(old_test.float(), dim=-1).to(device) - mean) / std
    current_eval = (F.normalize(current_test.float(), dim=-1).to(device) - mean) / std
    with torch.no_grad():
        scores = torch.cat(
            [probe(old_eval).squeeze(-1), probe(current_eval).squeeze(-1)], dim=0
        )
    labels = torch.cat(
        [torch.zeros(old_eval.size(0), device=device), torch.ones(current_eval.size(0), device=device)]
    )
    auc = _auc(scores, labels)
    accuracy = float(((scores >= 0) == (labels >= 0.5)).float().mean().cpu())
    return {"auc": float(auc), "accuracy": accuracy}


def _cosine_auc(
    model: LineageNativeCLM,
    parent_id: int,
    child_id: int,
    old_test: Tensor,
    current_test: Tensor,
) -> float:
    parent_key = F.normalize(model.cellular.cells[parent_id].route_key.detach().cpu(), dim=0)
    child_key = F.normalize(model.cellular.cells[child_id].route_key.detach().cpu(), dim=0)
    old_scores = F.normalize(old_test.float(), dim=-1).matmul(child_key - parent_key)
    current_scores = F.normalize(current_test.float(), dim=-1).matmul(child_key - parent_key)
    scores = torch.cat([old_scores, current_scores], dim=0)
    labels = torch.cat([torch.zeros(old_scores.size(0)), torch.ones(current_scores.size(0))])
    return float(_auc(scores, labels))


def _normalized_oracle_recovery(sketch_auc: float, oracle_auc: float) -> float:
    denominator = oracle_auc - 0.5
    if denominator <= 1e-9:
        return float("nan")
    return float((sketch_auc - 0.5) / denominator)


def diagnose_m3l_seed(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    eval_paths: dict[str, str | Path],
    output_path: str | Path,
    seed: int,
    config: M3LQuerySketchConfig,
    device: str,
) -> dict[str, Any]:
    config.validate()
    checkpoint = Path(checkpoint_path)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"M3L checkpoint SHA mismatch for seed {seed}: expected {expected_checkpoint_sha256}, got {actual_sha}"
        )
    target_device = torch.device(device)
    model, extra = LineageNativeCLM.load_checkpoint(checkpoint, map_location="cpu")
    if int(extra.get("seed", -1)) != int(seed) or extra.get("arm") != "lineage_growth":
        raise RuntimeError("checkpoint metadata does not match the requested M3R lineage seed")
    growth_events = list(extra.get("growth_events", []))
    if not growth_events:
        raise RuntimeError("lineage checkpoint has no growth events")
    model.to(target_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_hash_before = _parameter_state_sha256(model)
    edges = _edge_ownership_metadata(model, growth_events)

    by_domain: dict[str, dict[int, dict[str, Tensor]]] = {}
    for domain_index, domain in enumerate(PHASE_ORDER):
        if domain not in eval_paths:
            raise ValueError(f"missing evaluation path for domain {domain}")
        by_domain[domain] = _collect_domain_queries(
            model,
            edges,
            domain,
            eval_paths[domain],
            device=target_device,
            config=config,
            seq_len=model.config.max_seq_len,
            seed=config.split_seed_base + seed + 1000 * domain_index,
        )

    edge_results: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(edges):
        child_id = int(edge["child_id"])
        current_domain = str(edge["current_domain"])
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
            per_old_domain[domain] = {"train": int(train.size(0)), "test": int(test.size(0))}

        old_train = _balanced_concat(old_train_parts)
        old_test = _balanced_concat(old_test_parts)
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
            "old_domain_counts": per_old_domain,
            "old_train_samples": int(old_train.size(0)),
            "old_test_samples": int(old_test.size(0)),
            "current_train_samples": int(current_train.size(0)),
            "current_test_samples": int(current_test.size(0)),
            "valid": valid,
            "current_cosine_auc": None,
            "offline_oracle_auc": None,
            "offline_oracle_accuracy": None,
            "sketch_gate_auc": None,
            "sketch_gate_old_fpr": None,
            "sketch_gate_current_tpr": None,
            "normalized_oracle_excess_recovery": None,
            "historical_sketch_bytes": None,
            "historical_sketch_rank": None,
        }
        if valid:
            old_sketch = fit_low_rank_sketch(
                old_train,
                rank=config.sketch_rank,
                diagonal_regularization=config.diagonal_regularization,
                device=target_device,
            )
            current_sketch = fit_low_rank_sketch(
                current_train,
                rank=config.sketch_rank,
                diagonal_regularization=config.diagonal_regularization,
                device=target_device,
            )
            gate = derive_sketch_gate(
                old_sketch,
                current_sketch,
                diagonal_regularization=config.diagonal_regularization,
                target_old_fpr=config.target_sketch_old_fpr,
            )
            old_scores = _score_gate(old_test, gate, target_device)
            current_scores = _score_gate(current_test, gate, target_device)
            labels = torch.cat(
                [
                    torch.zeros(old_scores.size(0), device=target_device),
                    torch.ones(current_scores.size(0), device=target_device),
                ]
            )
            scores = torch.cat([old_scores, current_scores], dim=0)
            sketch_auc = float(_auc(scores, labels))
            threshold = float(gate["threshold"])
            old_fpr = float((old_scores > threshold).float().mean().cpu())
            current_tpr = float((current_scores > threshold).float().mean().cpu())
            oracle = _fit_oracle(
                old_train,
                current_train,
                old_test,
                current_test,
                config=config,
                seed=config.split_seed_base + seed + edge_index * 211,
                device=target_device,
            )
            cosine_auc = _cosine_auc(
                model,
                int(edge["parent_id"]),
                child_id,
                old_test,
                current_test,
            )
            result.update(
                {
                    "current_cosine_auc": cosine_auc,
                    "offline_oracle_auc": float(oracle["auc"]),
                    "offline_oracle_accuracy": float(oracle["accuracy"]),
                    "sketch_gate_auc": sketch_auc,
                    "sketch_gate_old_fpr": old_fpr,
                    "sketch_gate_current_tpr": current_tpr,
                    "normalized_oracle_excess_recovery": _normalized_oracle_recovery(
                        sketch_auc, float(oracle["auc"])
                    ),
                    "historical_sketch_bytes": old_sketch.storage_bytes,
                    "historical_sketch_rank": old_sketch.rank,
                    "gate_weight_norm": float(
                        torch.linalg.vector_norm(gate["weight"]).detach().cpu()
                    ),
                    "gate_threshold": threshold,
                }
            )
        edge_results.append(result)

    parameter_hash_after = _parameter_state_sha256(model)
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("M3L diagnostic mutated Native CLM trainable parameters")
    valid_edges = [edge for edge in edge_results if edge["valid"]]
    summary = {
        "format": "minicells.native-clm-v0.m3l-query-sketch-gate.seed.v1",
        "seed": int(seed),
        "checkpoint_sha256": actual_sha,
        "checkpoint_arm": "lineage_growth",
        "native_clm_training": False,
        "old_raw_query_replay_in_gate_fit": False,
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


def _metric_values(valid_edges: list[dict[str, Any]], name: str) -> list[float]:
    return [float(edge[name]) for edge in valid_edges if edge.get(name) is not None]


def aggregate_m3l_diagnostic(
    seed_summaries: list[dict[str, Any]],
    *,
    config: M3LQuerySketchConfig,
    parent_m3r_hf_revision: str,
    parent_address_commit: str,
) -> dict[str, Any]:
    all_edges = [edge for summary in seed_summaries for edge in summary["edges"]]
    valid_edges = [edge for edge in all_edges if edge["valid"]]
    valid_fraction = float(len(valid_edges) / max(1, len(all_edges)))

    cosine = _metric_values(valid_edges, "current_cosine_auc")
    oracle = _metric_values(valid_edges, "offline_oracle_auc")
    sketch = _metric_values(valid_edges, "sketch_gate_auc")
    recovery = _metric_values(valid_edges, "normalized_oracle_excess_recovery")
    old_fpr = _metric_values(valid_edges, "sketch_gate_old_fpr")
    current_tpr = _metric_values(valid_edges, "sketch_gate_current_tpr")

    def summarize(values: list[float]) -> dict[str, float]:
        if not values:
            return {"median": float("nan"), "mean": float("nan")}
        return {"median": float(median(values)), "mean": float(sum(values) / len(values))}

    oracle_summary = summarize(oracle)
    sketch_summary = summarize(sketch)
    oracle_fraction = (
        float(sum(value >= config.oracle_edge_auc_floor for value in oracle) / len(oracle))
        if oracle
        else 0.0
    )
    sketch_fraction = (
        float(sum(value >= config.sketch_gate_edge_auc_floor for value in sketch) / len(sketch))
        if sketch
        else 0.0
    )
    oracle_separable = bool(
        oracle
        and oracle_summary["median"] >= config.oracle_separable_median_auc
        and oracle_fraction >= config.minimum_fraction_oracle_edges_above_floor
    )
    sketch_feasible = bool(
        sketch
        and sketch_summary["median"] >= config.sketch_gate_median_auc
        and sketch_fraction >= config.minimum_fraction_sketch_edges_above_floor
        and median(recovery) >= config.median_normalized_oracle_excess_recovery
        and median(old_fpr) <= config.median_old_fpr_max
        and median(current_tpr) >= config.median_current_tpr_min
    )

    if valid_fraction < config.minimum_valid_edge_fraction:
        classification = "INCONCLUSIVE_COVERAGE"
    elif not oracle_separable:
        classification = "EDGE_LOCAL_QUERY_GEOMETRY_NOT_SEPARABLE"
    elif sketch_feasible:
        classification = "QUERY_SKETCH_GATE_FEASIBLE"
    else:
        classification = "QUERY_SKETCH_GATE_NOT_FEASIBLE"

    interpretation = {
        "INCONCLUSIVE_COVERAGE": "Too many lineage edges lack enough sequence-group-heldout ownership samples for a reliable M3L mechanism decision.",
        "EDGE_LOCAL_QUERY_GEOMETRY_NOT_SEPARABLE": "Under temporal parent ownership semantics, the edge-local query boundary is not consistently linearly separable; do not integrate a learned local gate yet.",
        "QUERY_SKETCH_GATE_FEASIBLE": "A compact historical query sketch plus current conflict-query moments recovers the local affine boundary without old-sample replay; proceed to an online sketch-maintaining M3L continual-language experiment.",
        "QUERY_SKETCH_GATE_NOT_FEASIBLE": "The offline edge-local oracle is separable, but the registered compact historical sketch loses too much boundary information; improve the address-state representation before a new continual-language formal run.",
    }[classification]

    return {
        "format": "minicells.native-clm-v0.m3l-query-sketch-gate.aggregate.v1",
        "classification": classification,
        "scientific_decision": False,
        "checkpoint_seeds": [int(summary["seed"]) for summary in seed_summaries],
        "parent_m3r_hf_revision": parent_m3r_hf_revision,
        "parent_address_diagnostic_commit": parent_address_commit,
        "native_clm_training": False,
        "old_raw_query_replay_in_gate_fit": False,
        "new_formal_seeds_consumed": False,
        "edge_count": len(all_edges),
        "valid_edge_count": len(valid_edges),
        "valid_edge_fraction": valid_fraction,
        "current_cosine": summarize(cosine),
        "offline_oracle": {
            **oracle_summary,
            "fraction_auc_ge_floor": oracle_fraction,
        },
        "sketch_gate": {
            **sketch_summary,
            "fraction_auc_ge_floor": sketch_fraction,
            "median_old_fpr": float(median(old_fpr)) if old_fpr else float("nan"),
            "median_current_tpr": float(median(current_tpr)) if current_tpr else float("nan"),
            "median_normalized_oracle_excess_recovery": (
                float(median(recovery)) if recovery else float("nan")
            ),
            "median_sketch_bytes": float(median(_metric_values(valid_edges, "historical_sketch_bytes")))
            if valid_edges
            else float("nan"),
        },
        "thresholds": {
            "minimum_valid_edge_fraction": config.minimum_valid_edge_fraction,
            "oracle_separable_median_auc": config.oracle_separable_median_auc,
            "oracle_edge_auc_floor": config.oracle_edge_auc_floor,
            "minimum_fraction_oracle_edges_above_floor": config.minimum_fraction_oracle_edges_above_floor,
            "sketch_gate_median_auc": config.sketch_gate_median_auc,
            "sketch_gate_edge_auc_floor": config.sketch_gate_edge_auc_floor,
            "minimum_fraction_sketch_edges_above_floor": config.minimum_fraction_sketch_edges_above_floor,
            "median_normalized_oracle_excess_recovery": config.median_normalized_oracle_excess_recovery,
            "median_old_fpr_max": config.median_old_fpr_max,
            "median_current_tpr_min": config.median_current_tpr_min,
        },
        "interpretation": interpretation,
    }


def write_m3l_edge_csv(seed_summaries: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "parent_id",
        "child_id",
        "root_id",
        "parent_birth_domain",
        "birth_domain",
        "old_domains",
        "current_domain",
        "old_train_samples",
        "old_test_samples",
        "current_train_samples",
        "current_test_samples",
        "valid",
        "current_cosine_auc",
        "offline_oracle_auc",
        "sketch_gate_auc",
        "normalized_oracle_excess_recovery",
        "sketch_gate_old_fpr",
        "sketch_gate_current_tpr",
        "historical_sketch_bytes",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in seed_summaries:
            for edge in summary["edges"]:
                row = {field: edge.get(field, "") for field in fields}
                row["old_domains"] = "+".join(edge["old_domains"])
                writer.writerow(row)
