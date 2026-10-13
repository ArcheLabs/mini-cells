"""Deterministic, fixed-degree candidate graphs and auditable edge refreshes."""
from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
from torch import Tensor


START = -1


def reachable_from_start(adjacency: Tensor) -> set[int]:
    """Return reachable Cell ids; the START row is stored at index ``cell_count``."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] + 1:
        raise ValueError("adjacency must have shape (cell_count + 1, cell_count)")
    count = adjacency.shape[1]
    seen: set[int] = set()
    queue = deque(torch.where(adjacency[count].bool())[0].tolist())
    while queue:
        node = int(queue.popleft())
        if node in seen:
            continue
        seen.add(node)
        queue.extend(torch.where(adjacency[node].bool())[0].tolist())
    return seen


def validate_adjacency(adjacency: Tensor, out_degree: int, *, require_reachable: bool = True) -> None:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] + 1:
        raise ValueError("adjacency must include one START row and one row per Cell")
    if adjacency.dtype != torch.bool:
        raise ValueError("adjacency must be boolean")
    if not torch.all(adjacency.sum(dim=1) == out_degree):
        raise ValueError("every source, including START, must have the registered out-degree")
    if require_reachable and reachable_from_start(adjacency) != set(range(adjacency.shape[1])):
        raise ValueError("every Cell must be reachable from START")


def make_initial_graph(cell_count: int, out_degree: int, seed: int) -> Tensor:
    """Create a seeded graph with exact degree and a guaranteed START-to-all cycle."""
    if not 2 <= out_degree <= cell_count:
        raise ValueError("out_degree must be in [2, cell_count]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    graph = torch.zeros((cell_count + 1, cell_count), dtype=torch.bool)
    # The ring supplies strong connectivity. The remaining edges are seeded and unique.
    for source in range(cell_count):
        targets = {(source + 1) % cell_count}
        candidates = torch.randperm(cell_count, generator=generator).tolist()
        for target in candidates:
            if len(targets) >= out_degree:
                break
            targets.add(int(target))
        graph[source, list(targets)] = True
    start_targets = list(range(out_degree))
    graph[cell_count, start_targets] = True
    validate_adjacency(graph, out_degree)
    return graph


def refresh_edges(
    adjacency: Tensor,
    candidate_losses: Tensor,
    *,
    step: int,
    max_replacements: int,
    min_improvement: float = 0.0,
) -> tuple[Tensor, list[dict[str, float | int]]]:
    """Replace edges only when a training-only NLL probe demonstrates improvement.

    ``candidate_losses[source, target]`` must be computed on the frozen training
    probe set. NaN means that edge was not probed. Lower NLL is better. A proposed
    swap is skipped if it breaks exact degree or START reachability.
    """
    validate_adjacency(adjacency, int(adjacency.sum(1)[0]))
    if candidate_losses.shape != adjacency.shape:
        raise ValueError("candidate_losses must match adjacency shape")
    if max_replacements < 0 or min_improvement < 0:
        raise ValueError("refresh limits must be non-negative")
    result = adjacency.clone()
    degree = int(adjacency.sum(1)[0])
    proposals: list[tuple[float, int, int, int, float, float]] = []
    for source in range(result.shape[0]):
        present = torch.where(result[source])[0].tolist()
        measured = [(float(candidate_losses[source, target]), source, target)
                    for target in present if torch.isfinite(candidate_losses[source, target])]
        outsiders = [(float(candidate_losses[source, target]), source, target)
                     for target in range(result.shape[1])
                     if not result[source, target] and torch.isfinite(candidate_losses[source, target])]
        if not measured or not outsiders:
            continue
        worst_loss, _, old_target = max(measured)
        best_loss, _, new_target = min(outsiders)
        improvement = worst_loss - best_loss
        if improvement > min_improvement:
            proposals.append((-improvement, source, old_target, new_target, worst_loss, best_loss))
    changes: list[dict[str, float | int]] = []
    for _, source, old_target, new_target, old_loss, new_loss in sorted(proposals):
        if len(changes) >= max_replacements:
            break
        result[source, old_target] = False
        result[source, new_target] = True
        try:
            validate_adjacency(result, degree)
        except ValueError:
            result[source, new_target] = False
            result[source, old_target] = True
            continue
        changes.append({"step": int(step), "source": source, "removed": old_target,
                        "added": new_target, "removed_probe_nll": old_loss,
                        "added_probe_nll": new_loss, "probe_improvement": old_loss - new_loss})
    validate_adjacency(result, degree)
    return result, changes


def permute_same_degree(adjacency: Tensor, seed: int, swaps: int = 8) -> Tensor:
    """Deterministically perturb a graph while preserving row degrees and reachability."""
    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    result = adjacency.clone()
    degree = int(result.sum(1)[0])
    successful, attempts = 0, 0
    max_attempts = max(100, swaps * 100)
    while successful < swaps and attempts < max_attempts:
        attempts += 1
        sources = torch.randperm(result.shape[0], generator=generator)[:2].tolist()
        source_a, source_b = int(sources[0]), int(sources[1])
        edges_a, edges_b = torch.where(result[source_a])[0], torch.where(result[source_b])[0]
        target_a = int(edges_a[torch.randint(len(edges_a), (1,), generator=generator)])
        target_b = int(edges_b[torch.randint(len(edges_b), (1,), generator=generator)])
        if target_a == target_b or result[source_a, target_b] or result[source_b, target_a]:
            continue
        result[source_a, target_a] = False
        result[source_b, target_b] = False
        result[source_a, target_b] = True
        result[source_b, target_a] = True
        try:
            validate_adjacency(result, degree)
        except ValueError:
            result[source_a, target_b] = False
            result[source_b, target_a] = False
            result[source_a, target_a] = True
            result[source_b, target_b] = True
        else:
            successful += 1
    if successful != swaps:
        raise RuntimeError("could not generate the registered number of reachable graph edge swaps")
    validate_adjacency(result, degree)
    return result


def perturb_important_edges(adjacency: Tensor, transition_counts: Tensor, *,
                            seed: int, replacements: int) -> tuple[Tensor, list[dict[str, int]]]:
    """Replace high-use training-probe edges while preserving degree/reachability."""
    validate_adjacency(adjacency, int(adjacency.sum(1)[0]))
    if transition_counts.shape != adjacency.shape or replacements <= 0:
        raise ValueError("transition counts and positive replacement budget are required")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    result = adjacency.clone()
    used_edges = [(int(transition_counts[source, target]), source, target)
                  for source, target in torch.nonzero(adjacency, as_tuple=False).tolist()
                  if int(transition_counts[source, target]) > 0]
    used_edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    changes: list[dict[str, int]] = []
    for uses, source, removed in used_edges:
        if len(changes) >= replacements:
            break
        candidates = torch.randperm(result.shape[1], generator=generator).tolist()
        for added in candidates:
            if result[source, added]:
                continue
            result[source, removed] = False
            result[source, added] = True
            try:
                validate_adjacency(result, int(adjacency.sum(1)[0]))
            except ValueError:
                result[source, added] = False
                result[source, removed] = True
                continue
            changes.append({"source": source, "removed": removed, "added": int(added),
                            "training_probe_uses": uses})
            break
    if len(changes) != replacements:
        raise RuntimeError("could not replace the requested number of used graph edges")
    validate_adjacency(result, int(adjacency.sum(1)[0]))
    return result, changes
