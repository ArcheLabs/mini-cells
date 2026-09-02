"""Checkpoint-only M3R parent/child address diagnostic.

This module never updates Native CLM parameters. It probes whether lineage-local
A-vs-birth-domain boundaries are recoverable from frozen query geometry or from
local write/effect factors at the Cellular Layer boundary.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .native_clm_m3 import _loader
from .native_clm_m3r import LineageNativeCLM


@dataclass(frozen=True)
class AddressDiagnosticConfig:
    max_batches_per_domain: int = 48
    batch_size: int = 8
    max_samples_per_class_per_edge: int = 2048
    minimum_samples_per_class_per_edge: int = 256
    train_fraction: float = 0.7
    probe_steps: int = 300
    probe_learning_rate: float = 0.05
    probe_weight_decay: float = 1e-4
    probe_split_seed_base: int = 73700
    minimum_valid_edge_fraction: float = 0.75
    separable_median_auc: float = 0.85
    separable_edge_auc_floor: float = 0.80
    minimum_fraction_edges_above_floor: float = 0.75

    def validate(self) -> None:
        if self.max_batches_per_domain < 1:
            raise ValueError("max_batches_per_domain must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.minimum_samples_per_class_per_edge < 16:
            raise ValueError("minimum sample count is too small")
        if self.max_samples_per_class_per_edge < self.minimum_samples_per_class_per_edge:
            raise ValueError("max samples must cover the minimum")
        if not 0.5 < self.train_fraction < 0.95:
            raise ValueError("train_fraction must be in (0.5, 0.95)")
        if self.probe_steps < 1 or self.probe_learning_rate <= 0:
            raise ValueError("invalid probe optimization settings")


FEATURE_NAMES = (
    "query",
    "write_input",
    "write_left",
    "write_pair",
    "certificate_residual",
)


def _birth_domain(global_step: int) -> str:
    if global_step <= 400:
        return "B"
    if global_step <= 800:
        return "C"
    return "D"


def _root_and_path_to_parent(model: LineageNativeCLM, parent_id: int) -> tuple[int, list[int]]:
    path = [int(parent_id)]
    current = int(parent_id)
    while current >= model.lineage_root_count:
        current = int(model.cellular.cells[current].parent_id.item())
        if current < 0:
            raise RuntimeError(f"invalid lineage while tracing parent {parent_id}")
        path.append(current)
    path.reverse()
    return current, path


def _edge_metadata(model: LineageNativeCLM, growth_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for event in growth_events:
        parent_id = int(event["parent_id"])
        child_id = int(event["child_id"])
        root_id, path_to_parent = _root_and_path_to_parent(model, parent_id)
        edges.append(
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "root_id": root_id,
                "path_to_parent": path_to_parent,
                "birth_step": int(event["global_step"]),
                "birth_domain": _birth_domain(int(event["global_step"])),
            }
        )
    return edges


def _forward_boundary_features(
    model: LineageNativeCLM,
    tokens: Tensor,
    targets: Tensor,
) -> dict[str, Tensor]:
    """Return frozen read features and dL/dh at the post-Cell boundary."""

    positions = torch.arange(tokens.size(1), device=tokens.device)
    with torch.no_grad():
        hidden = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
        hidden = model.dropout(hidden)
        cell_info: dict[str, Any] | None = None
        start_after = -1
        for index, block in enumerate(model.blocks):
            hidden = block(hidden)
            if index == model.config.cellular_layer_index:
                hidden, cell_info = model._lineage_cellular_forward(hidden, return_info=True)
                start_after = index + 1
                break
    if cell_info is None or start_after < 0:
        raise RuntimeError("Cellular Layer boundary was not reached")

    boundary = hidden.detach().requires_grad_(True)
    downstream = boundary
    with torch.enable_grad():
        for index in range(start_after, len(model.blocks)):
            downstream = model.blocks[index](downstream)
        downstream = model.final_norm(downstream)
        logits = model.lm_head(downstream)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        write_left = torch.autograd.grad(loss, boundary, retain_graph=False, create_graph=False)[0]

    route_input = cell_info["route_input"]
    with torch.no_grad():
        query = F.normalize(model.cellular.query_proj(route_input), dim=-1)
    return {
        "query": query.detach(),
        "write_input": cell_info["cell_input"].detach(),
        "write_left": write_left.detach(),
        "root_idx": cell_info["root_idx"].detach(),
    }


def _edge_eligible_mask(
    model: LineageNativeCLM,
    query: Tensor,
    root_idx: Tensor,
    *,
    path_to_parent: list[int],
) -> Tensor:
    root_id = int(path_to_parent[0])
    eligible = (root_idx == root_id).any(dim=-1)
    if len(path_to_parent) == 1:
        return eligible
    keys = [F.normalize(cell.route_key.detach(), dim=0) for cell in model.cellular.cells]
    for parent_id, child_id in itertools.pairwise(path_to_parent):
        parent_score = query.matmul(keys[parent_id])
        child_score = query.matmul(keys[child_id])
        eligible = eligible & (child_score > parent_score)
    return eligible


def _certificate_residual(model: LineageNativeCLM, parent_id: int, x: Tensor) -> Tensor:
    cell = model.cellular.cells[parent_id]
    if cell.rank == 0:
        return x
    q = cell.certificate_basis[: cell.rank].to(device=x.device, dtype=x.dtype)
    return x - x.matmul(q.transpose(0, 1)).matmul(q)


def _append_limited(target: list[Tensor], values: Tensor, remaining: int) -> int:
    if remaining <= 0 or values.numel() == 0:
        return 0
    take = min(remaining, values.size(0))
    target.append(values[:take].detach().float().cpu())
    return take


def _collect_edge_domain_samples(
    model: LineageNativeCLM,
    edges: list[dict[str, Any]],
    domain: str,
    path: str | Path,
    *,
    device: torch.device,
    config: AddressDiagnosticConfig,
    seq_len: int,
    seed: int,
) -> dict[int, dict[str, Tensor]]:
    relevant = [edge for edge in edges if domain == "A" or edge["birth_domain"] == domain]
    stores: dict[int, dict[str, list[Tensor]]] = {
        int(edge["child_id"]): {
            "query": [],
            "write_input": [],
            "write_left": [],
            "certificate_residual": [],
            "cosine_margin": [],
        }
        for edge in relevant
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
    iterator = iter(loader)
    model.eval()
    for batch_idx in range(config.max_batches_per_domain):
        try:
            tokens, targets = next(iterator)
        except StopIteration:
            break
        tokens = tokens.to(device)
        targets = targets.to(device)
        features = _forward_boundary_features(model, tokens, targets)
        query = features["query"]
        write_input = features["write_input"]
        write_left = features["write_left"]
        root_idx = features["root_idx"]

        for edge in relevant:
            child_id = int(edge["child_id"])
            remaining = config.max_samples_per_class_per_edge - counts[child_id]
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
            x_values = write_input[mask]
            left_values = write_left[mask]
            parent_id = int(edge["parent_id"])
            parent_key = F.normalize(model.cellular.cells[parent_id].route_key.detach(), dim=0)
            child_key = F.normalize(model.cellular.cells[child_id].route_key.detach(), dim=0)
            margins = q_values.matmul(child_key) - q_values.matmul(parent_key)
            residual = _certificate_residual(model, parent_id, x_values)

            take = min(remaining, q_values.size(0))
            stores[child_id]["query"].append(q_values[:take].detach().float().cpu())
            stores[child_id]["write_input"].append(x_values[:take].detach().float().cpu())
            stores[child_id]["write_left"].append(left_values[:take].detach().float().cpu())
            stores[child_id]["certificate_residual"].append(residual[:take].detach().float().cpu())
            stores[child_id]["cosine_margin"].append(margins[:take].detach().float().cpu().unsqueeze(-1))
            counts[child_id] += take

        if all(count >= config.max_samples_per_class_per_edge for count in counts.values()):
            break

    packed: dict[int, dict[str, Tensor]] = {}
    for child_id, feature_lists in stores.items():
        packed[child_id] = {}
        for name, pieces in feature_lists.items():
            width = 1 if name == "cosine_margin" else model.config.d_model
            packed[child_id][name] = (
                torch.cat(pieces, dim=0) if pieces else torch.empty((0, width), dtype=torch.float32)
            )
    return packed


def _auc(scores: Tensor, labels: Tensor) -> float:
    values = scores.detach().float().cpu().reshape(-1).numpy()
    target = labels.detach().long().cpu().reshape(-1).numpy()
    n_pos = int((target == 1).sum())
    n_neg = int((target == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[target == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _split_indices(n: int, train_fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    train_count = max(1, min(n - 1, math.floor(n * train_fraction)))
    return order[:train_count], order[train_count:]


def _fit_probe(
    negative: Tensor,
    positive: Tensor,
    *,
    config: AddressDiagnosticConfig,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    n = min(negative.size(0), positive.size(0), config.max_samples_per_class_per_edge)
    neg = negative[:n].float()
    pos = positive[:n].float()
    train_neg, test_neg = _split_indices(n, config.train_fraction, seed)
    train_pos, test_pos = _split_indices(n, config.train_fraction, seed + 1)
    x_train = torch.cat([neg[train_neg], pos[train_pos]], dim=0).to(device)
    y_train = torch.cat(
        [torch.zeros(len(train_neg)), torch.ones(len(train_pos))], dim=0
    ).to(device)
    x_test = torch.cat([neg[test_neg], pos[test_pos]], dim=0).to(device)
    y_test = torch.cat([torch.zeros(len(test_neg)), torch.ones(len(test_pos))], dim=0).to(device)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    torch.manual_seed(seed)
    probe = nn.Linear(x_train.size(-1), 1, bias=True).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.probe_learning_rate,
        weight_decay=config.probe_weight_decay,
    )
    for _ in range(config.probe_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = probe(x_test).squeeze(-1)
        auc = _auc(scores, y_test)
        accuracy = float(((scores >= 0) == (y_test >= 0.5)).float().mean().cpu())
    return {
        "auc": float(auc),
        "accuracy": accuracy,
        "samples_per_class": int(n),
        "train_samples": int(x_train.size(0)),
        "test_samples": int(x_test.size(0)),
    }


def diagnose_seed(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    eval_paths: dict[str, str | Path],
    output_path: str | Path,
    seed: int,
    config: AddressDiagnosticConfig,
    device: str,
) -> dict[str, Any]:
    config.validate()
    from .native_clm_m2 import sha256_file

    checkpoint = Path(checkpoint_path)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"M3R checkpoint SHA mismatch for seed {seed}: expected {expected_checkpoint_sha256}, got {actual_sha}"
        )
    target_device = torch.device(device)
    model, extra = LineageNativeCLM.load_checkpoint(checkpoint, map_location="cpu")
    if int(extra.get("seed", -1)) != int(seed) or extra.get("arm") != "lineage_growth":
        raise RuntimeError("checkpoint extra metadata does not match the requested M3R lineage seed")
    growth_events = list(extra.get("growth_events", []))
    if not growth_events:
        raise RuntimeError("M3R lineage checkpoint has no recorded growth events")
    model.to(target_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    edges = _edge_metadata(model, growth_events)

    by_domain: dict[str, dict[int, dict[str, Tensor]]] = {}
    for index, domain in enumerate(("A", "B", "C", "D")):
        if domain not in eval_paths:
            raise ValueError(f"missing evaluation path for domain {domain}")
        by_domain[domain] = _collect_edge_domain_samples(
            model,
            edges,
            domain,
            eval_paths[domain],
            device=target_device,
            config=config,
            seq_len=model.config.max_seq_len,
            seed=config.probe_split_seed_base + seed + 100 * index,
        )

    edge_results: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(edges):
        child_id = int(edge["child_id"])
        birth_domain = str(edge["birth_domain"])
        old = by_domain["A"].get(child_id)
        new = by_domain[birth_domain].get(child_id)
        old_count = 0 if old is None else int(old["query"].size(0))
        new_count = 0 if new is None else int(new["query"].size(0))
        valid = min(old_count, new_count) >= config.minimum_samples_per_class_per_edge
        result: dict[str, Any] = {
            **edge,
            "seed": int(seed),
            "old_domain": "A",
            "old_samples": old_count,
            "birth_domain_samples": new_count,
            "valid": bool(valid),
            "current_cosine_auc": None,
            "probes": {},
        }
        if valid and old is not None and new is not None:
            n = min(old_count, new_count, config.max_samples_per_class_per_edge)
            neg_train, neg_test = _split_indices(n, config.train_fraction, config.probe_split_seed_base + seed + edge_index * 13)
            pos_train, pos_test = _split_indices(n, config.train_fraction, config.probe_split_seed_base + seed + edge_index * 13 + 1)
            del neg_train, pos_train
            cosine_scores = torch.cat(
                [old["cosine_margin"][:n][neg_test], new["cosine_margin"][:n][pos_test]], dim=0
            ).squeeze(-1)
            cosine_labels = torch.cat(
                [torch.zeros(len(neg_test)), torch.ones(len(pos_test))], dim=0
            )
            result["current_cosine_auc"] = float(_auc(cosine_scores, cosine_labels))

            write_input_old = old["write_input"]
            write_input_new = new["write_input"]
            write_left_old = old["write_left"]
            write_left_new = new["write_left"]
            feature_pairs = {
                "query": (old["query"], new["query"]),
                "write_input": (write_input_old, write_input_new),
                "write_left": (write_left_old, write_left_new),
                "write_pair": (
                    torch.cat(
                        [F.normalize(write_input_old, dim=-1), F.normalize(write_left_old, dim=-1)], dim=-1
                    ),
                    torch.cat(
                        [F.normalize(write_input_new, dim=-1), F.normalize(write_left_new, dim=-1)], dim=-1
                    ),
                ),
                "certificate_residual": (
                    old["certificate_residual"],
                    new["certificate_residual"],
                ),
            }
            for feature_offset, (feature_name, (negative, positive)) in enumerate(feature_pairs.items()):
                result["probes"][feature_name] = _fit_probe(
                    negative,
                    positive,
                    config=config,
                    seed=config.probe_split_seed_base + seed + edge_index * 101 + feature_offset,
                    device=target_device,
                )
        edge_results.append(result)

    valid_edges = [result for result in edge_results if result["valid"]]
    seed_summary = {
        "format": "minicells.native-clm-v0.m3r-address-diagnostic.seed.v1",
        "seed": int(seed),
        "checkpoint_sha256": actual_sha,
        "checkpoint_arm": "lineage_growth",
        "model_training": False,
        "growth": False,
        "certificate_updates": False,
        "edge_count": len(edge_results),
        "valid_edge_count": len(valid_edges),
        "valid_edge_fraction": float(len(valid_edges) / max(1, len(edge_results))),
        "edges": edge_results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(seed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return seed_summary


def _feature_summary(valid_edges: list[dict[str, Any]], feature: str) -> dict[str, float]:
    values = [float(edge["probes"][feature]["auc"]) for edge in valid_edges]
    if not values:
        return {"median_auc": float("nan"), "mean_auc": float("nan"), "fraction_auc_ge_floor": 0.0}
    return {
        "median_auc": float(median(values)),
        "mean_auc": float(sum(values) / len(values)),
    }


def aggregate_diagnostic(
    seed_summaries: list[dict[str, Any]],
    *,
    config: AddressDiagnosticConfig,
    parent_protocol_sha256: str,
    parent_data_manifest_sha256: str,
    hf_revision: str,
) -> dict[str, Any]:
    all_edges = [edge for summary in seed_summaries for edge in summary["edges"]]
    valid_edges = [edge for edge in all_edges if edge["valid"]]
    valid_fraction = float(len(valid_edges) / max(1, len(all_edges)))
    feature_summaries: dict[str, dict[str, float]] = {}
    for feature in FEATURE_NAMES:
        values = [float(edge["probes"][feature]["auc"]) for edge in valid_edges]
        if values:
            feature_summaries[feature] = {
                "median_auc": float(median(values)),
                "mean_auc": float(sum(values) / len(values)),
                "fraction_auc_ge_floor": float(
                    sum(value >= config.separable_edge_auc_floor for value in values) / len(values)
                ),
            }
        else:
            feature_summaries[feature] = {
                "median_auc": float("nan"),
                "mean_auc": float("nan"),
                "fraction_auc_ge_floor": 0.0,
            }
    cosine_values = [float(edge["current_cosine_auc"]) for edge in valid_edges]
    cosine_summary = {
        "median_auc": float(median(cosine_values)) if cosine_values else float("nan"),
        "mean_auc": float(sum(cosine_values) / len(cosine_values)) if cosine_values else float("nan"),
    }

    def separable(feature: str) -> bool:
        record = feature_summaries[feature]
        return bool(
            record["median_auc"] >= config.separable_median_auc
            and record["fraction_auc_ge_floor"] >= config.minimum_fraction_edges_above_floor
        )

    if valid_fraction < config.minimum_valid_edge_fraction:
        classification = "INCONCLUSIVE_COVERAGE"
    elif separable("query"):
        classification = "QUERY_GEOMETRY_SEPARABLE"
    elif any(separable(name) for name in ("write_left", "write_pair", "certificate_residual")):
        classification = "WRITE_EFFECT_GEOMETRY_SEPARABLE"
    else:
        classification = "NO_CLEAR_LOCAL_BOUNDARY"

    result = {
        "format": "minicells.native-clm-v0.m3r-address-diagnostic.aggregate.v1",
        "classification": classification,
        "scientific_decision": False,
        "parent_m3r_protocol_sha256": parent_protocol_sha256,
        "parent_m3r_data_manifest_sha256": parent_data_manifest_sha256,
        "parent_m3r_hf_revision": hf_revision,
        "checkpoint_seeds": [int(summary["seed"]) for summary in seed_summaries],
        "model_training": False,
        "new_formal_seeds_consumed": False,
        "edge_count": len(all_edges),
        "valid_edge_count": len(valid_edges),
        "valid_edge_fraction": valid_fraction,
        "current_cosine": cosine_summary,
        "features": feature_summaries,
        "thresholds": {
            "minimum_valid_edge_fraction": config.minimum_valid_edge_fraction,
            "separable_median_auc": config.separable_median_auc,
            "separable_edge_auc_floor": config.separable_edge_auc_floor,
            "minimum_fraction_edges_above_floor": config.minimum_fraction_edges_above_floor,
        },
        "interpretation": {
            "QUERY_GEOMETRY_SEPARABLE": "Frozen query geometry contains a stable local boundary; prioritize a learned lineage-local read gate.",
            "WRITE_EFFECT_GEOMETRY_SEPARABLE": "Query geometry is insufficient but write/effect factors are separable; prioritize separate read/write addressing and a write-side conflict controller.",
            "NO_CLEAR_LOCAL_BOUNDARY": "No registered linear local boundary is stable enough; investigate richer learned functional coordinates before another routing heuristic.",
            "INCONCLUSIVE_COVERAGE": "Too many lineage edges lack sufficient old/birth-domain eligible samples for this diagnostic.",
        }[classification],
    }
    return result


def write_edge_csv(seed_summaries: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "parent_id",
        "child_id",
        "root_id",
        "birth_step",
        "birth_domain",
        "old_samples",
        "birth_domain_samples",
        "valid",
        "current_cosine_auc",
        *[f"{name}_auc" for name in FEATURE_NAMES],
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in seed_summaries:
            for edge in summary["edges"]:
                row = {field: "" for field in fields}
                for key in (
                    "seed",
                    "parent_id",
                    "child_id",
                    "root_id",
                    "birth_step",
                    "birth_domain",
                    "old_samples",
                    "birth_domain_samples",
                    "valid",
                    "current_cosine_auc",
                ):
                    row[key] = edge.get(key, "")
                if edge["valid"]:
                    for name in FEATURE_NAMES:
                        row[f"{name}_auc"] = edge["probes"][name]["auc"]
                writer.writerow(row)
