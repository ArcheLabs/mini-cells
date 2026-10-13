"""Gate-1 order-sensitive mechanism fixture; never language-model evidence."""
from __future__ import annotations

import torch

from .topology import perturb_important_edges, refresh_edges, validate_adjacency


def run_synthetic_composition_gate() -> dict[str, object]:
    """Check noncommuting Cell order and that NLL-guided refresh improves a bad graph."""
    # Two residual-like affine cells over a two-dimensional state.
    def cell_a(x: torch.Tensor) -> torch.Tensor:
        return torch.stack((x[0] + x[1], x[1]))

    def cell_b(x: torch.Tensor) -> torch.Tensor:
        return torch.stack((x[0], x[0] + x[1]))

    start = torch.tensor([2.0, 1.0])
    ab = cell_b(cell_a(start))
    ba = cell_a(cell_b(start))
    if torch.equal(ab, ba):
        raise AssertionError("synthetic Cell transforms must be order-sensitive")

    # Sources 0=A, 1=B, 2=auxiliary, 3=START. The initial fixed graph
    # deliberately excludes the useful A->B and B->A transitions.
    graph = torch.tensor([
        [True, False, True],
        [False, True, True],
        [True, True, False],
        [True, True, False],
    ], dtype=torch.bool)
    validate_adjacency(graph, 2)
    nll = torch.full(graph.shape, float("nan"), dtype=torch.float64)
    nll[0, 0], nll[0, 2], nll[0, 1] = 1.5, 1.2, 0.2
    nll[1, 1], nll[1, 2], nll[1, 0] = 1.5, 1.2, 0.1
    current_loss = float(torch.stack((nll[0, 0], nll[0, 2], nll[1, 1], nll[1, 2])).mean())
    learned, changes = refresh_edges(graph, nll, step=500, max_replacements=2)
    refreshed_loss = float(torch.stack((nll[0, 1], nll[0, 2], nll[1, 0], nll[1, 2])).mean())
    if len(changes) != 2 or not refreshed_loss < current_loss:
        raise AssertionError("training-probe edge refresh failed to improve the synthetic graph")
    edge_use = torch.zeros_like(graph, dtype=torch.int64)
    edge_use[0, 0], edge_use[1, 1] = 9, 7
    perturbed, removed_important = perturb_important_edges(
        graph, edge_use, seed=2718, replacements=1
    )
    validate_adjacency(perturbed, 2)
    if removed_important[0]["training_probe_uses"] != 9 or torch.equal(perturbed, graph):
        raise AssertionError("important-edge intervention did not remove the most-used training edge")
    return {"status": "GATE_1_MECHANISM_FIXTURE_PASSED",
            "scientific_decision": False,
            "ab_output": ab.tolist(), "ba_output": ba.tolist(),
            "fixed_graph_probe_nll": current_loss,
            "refreshed_graph_probe_nll": refreshed_loss,
            "edge_changes": changes,
            "important_edge_intervention": removed_important,
            "scope": "deterministic synthetic teacher-forced utility fixture; not a trained router or LM result"}
