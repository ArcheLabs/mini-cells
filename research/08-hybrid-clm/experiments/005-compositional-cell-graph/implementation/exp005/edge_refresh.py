"""Training-only teacher-forced edge probes shared by sequential control arms."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from experiments.common.train import make_batch

from .accounting import ObservedGemmFlops
from .topology import make_initial_graph, refresh_edges


def make_edge_refresh_callback(config: dict[str, Any], assets: dict[str, Any], seed: int):
    """Build a deterministic probe callback; G0 updates, R0/S0 pay the same cost.

    The callback consumes only the fixed training-derived probe starts. For each
    layer it compares the incumbent graph candidates plus a fixed number of
    sampled outsiders using next-token NLL on tokens whose pre-probe path had the
    corresponding source Cell. R0 and S0 run the same interventions and discard
    the update, providing the proposal's matched-compute control.
    """
    refresh_cfg = config["training"]
    required = ("edge_refresh_candidate_count", "edge_refresh_probe_count",
                "max_edge_replacements_per_refresh", "edge_refresh_min_improvement")
    missing = [name for name in required if refresh_cfg.get(name) is None]
    if missing:
        raise RuntimeError("edge-refresh protocol is not frozen: " + ", ".join(missing))
    model_cfg, data_cfg = config["model"], config["data"]
    candidate_count = int(refresh_cfg["edge_refresh_candidate_count"])
    probe_count = int(refresh_cfg["edge_refresh_probe_count"])
    max_replacements = int(refresh_cfg["max_edge_replacements_per_refresh"])
    min_improvement = float(refresh_cfg["edge_refresh_min_improvement"])
    train = assets["train"]
    probe_starts = np.asarray(assets["probe_starts"], dtype=np.int64)
    if candidate_count < 1 or probe_count < 1 or len(probe_starts) < probe_count:
        raise ValueError("invalid edge-refresh probe/candidate counts")
    width = int(model_cfg["cell_count"])
    degree = int(model_cfg["graph_out_degree"])
    visits = int(model_cfg["cell_visits"])
    seq = int(data_cfg["sequence_length"])
    layers = int(model_cfg["layers"])
    interval = int(refresh_cfg["edge_refresh_every_steps"])

    def callback(model, step: int, _rng: np.random.Generator) -> dict[str, Any]:
        previous_training = model.training
        model.eval()
        model.set_collect_trace(True)
        event = max(0, step // interval - 1)
        offset = (event * probe_count) % len(probe_starts)
        positions = np.take(probe_starts, np.arange(offset, offset + probe_count) % len(probe_starts))
        x, targets = make_batch(train, positions, seq, next(model.parameters()).device)
        measurements = [[[] for _ in range(width)] for _ in range(layers)]
        probe_flops = 0
        try:
            with torch.no_grad():
                baseline, _, _ = model(x)
                del baseline
                traces = model.last_execution
                if any(trace is None for trace in traces):
                    raise RuntimeError("edge refresh requires a complete baseline path trace")
                for layer in range(layers):
                    # One source row per layer per scheduled refresh. The source
                    # and hop rotate deterministically; all arms share the schedule.
                    hop = event % visits
                    source = width if hop == 0 else (event + layer * 7) % width
                    paths = traces[layer]["paths"]
                    if hop == 0:
                        source_mask = torch.ones_like(targets, dtype=torch.bool)
                    else:
                        source_mask = paths[:, :, hop - 1] == source

                    if model.arm in {"S0", "R0", "G0"}:
                        ffn = model.blocks[layer].ffn
                        if ffn.arm in {"R0", "G0"}:
                            current_graph = ffn.adjacency.detach().cpu().clone()
                        else:
                            current_graph = make_initial_graph(width, degree, seed + layer)
                    else:
                        raise RuntimeError("edge refresh is only valid for S0/R0/G0")
                    incumbents = torch.where(current_graph[source])[0].tolist()
                    outsiders = [cell for cell in range(width) if cell not in incumbents]
                    candidate_rng = np.random.default_rng(seed + 1009 * event + 37 * layer + source)
                    if candidate_count > len(outsiders):
                        raise ValueError("edge-refresh candidate count exceeds non-neighbor count")
                    sampled = sorted(candidate_rng.choice(outsiders, size=candidate_count,
                                                         replace=False).tolist())
                    candidates = sorted(set(incumbents + sampled))
                    for target in candidates:
                        counter = ObservedGemmFlops(model)
                        with counter, torch.no_grad():
                            logits, _, _ = model(x, forced_transition=(layer, hop, source, target))
                        probe_flops += counter.total
                        selected = logits[source_mask]
                        labels = targets[source_mask]
                        if labels.numel():
                            loss = F.cross_entropy(selected.float().reshape(-1, selected.shape[-1]),
                                                   labels.reshape(-1), reduction="mean")
                            measurements[layer][source].append((target, float(loss)))

            edge_changes = []
            layer_summaries = []
            if model.arm in {"R0", "G0"}:
                for layer in range(layers):
                    graph = model.blocks[layer].ffn.adjacency.detach().cpu().clone()
                    losses = torch.full(graph.shape, float("nan"), dtype=torch.float64)
                    source = width if event % visits == 0 else (event + layer * 7) % width
                    candidates = measurements[layer][source]
                    if candidates:
                        by_target: dict[int, list[float]] = {}
                        for target, value in candidates:
                            by_target.setdefault(target, []).append(value)
                        for target, values in by_target.items():
                            losses[source, target] = float(np.mean(values))
                    if model.arm == "G0":
                        updated, changes = refresh_edges(
                            graph, losses, step=step, max_replacements=max_replacements,
                            min_improvement=min_improvement
                        )
                        model.blocks[layer].ffn.adjacency.copy_(updated.to(
                            model.blocks[layer].ffn.adjacency.device
                        ))
                        edge_changes.extend({"layer": layer, **change} for change in changes)
                    layer_summaries.append({"layer": layer, "source": source,
                                            "probed_edges": len(candidates)})
            for layer in range(layers):
                model.blocks[layer].ffn.last_trace = None
            return {"step": step, "event": event, "arm": model.arm,
                    "probe_source": "fixed training-derived probe windows",
                    "candidate_probe_count_per_layer": candidate_count,
                    "probe_window_count": probe_count,
                    "observed_probe_gemm_flops": probe_flops,
                    "layers": layer_summaries, "edge_changes": edge_changes,
                    "topology_updated": model.arm == "G0",
                    "matched_compute_only": model.arm in {"S0", "R0"},
                    "loss_definition": "mean teacher-forced next-token NLL on pre-probe rows routed from the source Cell"}
        finally:
            model.set_collect_trace(False)
            model.train(previous_training)

    return callback
