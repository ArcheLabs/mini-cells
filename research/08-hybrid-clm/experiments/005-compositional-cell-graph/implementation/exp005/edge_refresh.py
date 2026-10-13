"""Balanced, train-only counterfactual edge probes and matched-cost controls."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from experiments.common.train import make_batch
from .accounting import ObservedGemmFlops
from .topology import make_initial_graph, refresh_edges


def scheduled_sources(event: int, width: int, count: int) -> list[int]:
    """Visit START and every Cell in consecutive cycles, including unused Cells."""
    if event < 0 or not 1 <= count <= width + 1:
        raise ValueError("invalid source-row schedule")
    return [(event * count + offset) % (width + 1) for offset in range(count)]


def make_edge_refresh_callback(config: dict[str, Any], assets: dict[str, Any], seed: int):
    cfg, model_cfg, data = config["training"], config["model"], config["data"]
    names = ("edge_refresh_candidate_count", "edge_refresh_probe_count", "edge_refresh_source_count",
             "max_edge_replacements_per_refresh", "edge_refresh_min_improvement")
    if any(cfg.get(name) is None for name in names):
        raise RuntimeError("edge-refresh protocol is not frozen")
    width, degree = int(model_cfg["cell_count"]), int(model_cfg["graph_out_degree"])
    visits, layers = int(model_cfg["cell_visits"]), int(model_cfg["layers"])
    count, candidates_count = int(cfg["edge_refresh_probe_count"]), int(cfg["edge_refresh_candidate_count"])
    source_count = int(cfg["edge_refresh_source_count"])
    starts = np.asarray(assets["probe_starts"], dtype=np.int64)
    if count < 1 or count > len(starts) or not 1 <= candidates_count <= width - degree:
        raise ValueError("invalid fixed probe or outsider candidate count")
    scheduled_sources(0, width, source_count)

    def callback(model, step: int, _rng) -> dict[str, Any]:
        if model.arm not in {"S0", "R0", "G0"}:
            raise ValueError("edge probes require a sequential arm")
        was_training = model.training
        model.eval()
        model.set_collect_trace(True)
        event = step // int(cfg["edge_refresh_every_steps"]) - 1
        sources = scheduled_sources(event, width, source_count)
        selected = np.take(starts, (np.arange(count) + event * count) % len(starts))
        x, y = make_batch(assets["train"], selected, int(data["sequence_length"]), next(model.parameters()).device)
        probe_flops, changes, summaries = 0, [], []
        try:
            with torch.no_grad():
                with ObservedGemmFlops(model) as counter:
                    model(x)
                probe_flops += counter.total
                baseline_paths = [trace["paths"].clone() for trace in model.last_execution]
                for layer in range(layers):
                    ffn = model.blocks[layer].ffn
                    graph = (ffn.adjacency.detach().cpu().clone() if model.arm != "S0"
                             else make_initial_graph(width, degree, seed + layer))
                    losses = torch.full(graph.shape, float("nan"), dtype=torch.float64)
                    rows = []
                    for source in sources:
                        hop = 0 if source == width else 1 + event % (visits - 1)
                        natural_support = y.numel() if source == width else int((baseline_paths[layer][:, :, hop - 1] == source).sum())
                        incumbents = torch.where(graph[source])[0].tolist()
                        outsiders = [target for target in range(width) if target not in incumbents]
                        rng = np.random.default_rng(seed + 1009 * event + layer * 37 + source)
                        sampled = sorted(rng.choice(outsiders, candidates_count, replace=False).tolist())
                        # Force the predecessor itself for unused source Cells. This is
                        # a counterfactual intervention, not claimed as natural visitation.
                        prefix = [] if hop == 0 else [(layer, hop - 1, -1, source)]
                        for target in sorted(incumbents + sampled):
                            with ObservedGemmFlops(model) as counter:
                                logits, _, _ = model(x, forced_transition=prefix + [(layer, hop, source, target)])
                            probe_flops += counter.total
                            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                            if not torch.isfinite(loss):
                                raise RuntimeError("non-finite training-only edge probe")
                            losses[source, target] = float(loss)
                        rows.append({"source": source, "hop": hop, "natural_support_tokens": natural_support,
                                     "effective_probe_tokens": y.numel(), "probed_edges": degree + candidates_count,
                                     "forced_predecessor": hop > 0})
                    if model.arm == "G0":
                        updated, updates = refresh_edges(graph, losses, step=step,
                            max_replacements=int(cfg["max_edge_replacements_per_refresh"]),
                            min_improvement=float(cfg["edge_refresh_min_improvement"]))
                        ffn.adjacency.copy_(updated.to(ffn.adjacency.device))
                        changes.extend({"layer": layer, **change} for change in updates)
                    summaries.append({"layer": layer, "sources": rows})
            return {"step": step, "event": event, "arm": model.arm, "layers": summaries,
                    "probe_source": "fixed training-derived probe windows",
                    "probe_window_count": count, "observed_probe_gemm_flops": probe_flops,
                    "edge_changes": changes, "topology_updated": model.arm == "G0",
                    "matched_compute_only": model.arm in {"S0", "R0"},
                    "loss_definition": "counterfactual whole-window next-token NLL; forced predecessor and target; never holdout",
                    "source_coverage_schedule": "consecutive cyclic rows over all Cells plus START"}
        finally:
            model.set_collect_trace(False)
            model.train(was_training)
    return callback
