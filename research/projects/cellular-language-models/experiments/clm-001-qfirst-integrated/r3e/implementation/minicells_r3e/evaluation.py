"""Calibration is selection data; Development is final-only. Matrix hooks are terminal-only."""

import time
from itertools import islice
import torch
import numpy as np
from minicells_r3c_moe_bridge.data import fixed_windows
from minicells_r3c_moe_bridge.evaluation import evaluate as historical_evaluate
from minicells_r3c_moe_bridge.instrumentation import MatmulAudit
from minicells_r3d.evaluation import mean_nll, sync
from .common import fingerprint


def windows(data, c, role, device):
    if role not in ("calibration", "development"):
        raise ValueError("evaluation role forbidden")
    return fixed_windows(data, c["engine"], role, device)


@torch.no_grad()
def calibration(m, c, data):
    return mean_nll(
        historical_evaluate(m, windows(data, c, "calibration", next(m.parameters()).device), m.seed)
    )


@torch.no_grad()
def reference(m, c, data):
    device = next(m.parameters()).device
    return [
        x
        for _, x, _ in islice(
            windows(data, c, "calibration", device), c["proposal"]["reference_calibration_windows"]
        )
    ]


@torch.no_grad()
def paths(m, xs):
    return [m(x).actions for x in xs]


@torch.no_grad()
def trust(old, new, refs):
    kl = []
    change = []
    path = []
    for x in refs:
        a = old(x, keep_trace=True)
        b = new(x)
        for rec in a.trace:
            t = rec["stage"]
            h = rec["state"]
            prev = rec["previous"]
            qo, _, _ = old.scores(h, t, prev)
            qn, _, _ = new.scores(h, t, prev)
            logp = qo.log_softmax(-1)
            logq = qn.log_softmax(-1)
            kl.append((logp.exp() * (logp - logq)).sum(-1))
            new_action, _ = new.select(h, t, prev)
            change.append((rec["action"] != new_action).float())
        path.append((a.actions != b.actions).any(-1).float())
    return {
        "kl": float(torch.cat(kl).mean()),
        "action_change": float(torch.cat(change).mean()),
        "path_change": float(torch.cat(path).mean()),
    }


def within(metrics, c):
    q = c["proposal"]
    return (
        metrics["kl"] <= q["kl_max"]
        and metrics["action_change"] <= q["action_change_max"]
        and metrics["path_change"] <= q["path_change_max"]
    )


@torch.no_grad()
def graph_summary(m, c, data):
    if not m.graph:
        return None
    device = next(m.parameters()).device
    # Input-only grouping threshold fitted once on Calibration; never fit on Development labels.
    scalars = [x.float().mean().item() for _, x, _ in windows(data, c, "calibration", device)]
    threshold = float(np.median(scalars))
    edges = torch.zeros(6, 6, device=device, dtype=torch.long)
    grouped = torch.zeros(2, 6, 6, device=device, dtype=torch.long)
    steps = torch.zeros(8, 6, device=device, dtype=torch.long)
    path_counts = {}
    for _, x, _ in windows(data, c, "development", device):
        out = m(x)
        a = out.actions
        group = int(x.float().mean().item() > threshold)
        for k in range(8):
            steps[k] += torch.bincount(a[:, k], minlength=6)
        for k in range(1, 8):
            counts = torch.bincount(6 * a[:, k - 1] + a[:, k], minlength=36).reshape(6, 6)
            edges += counts
            grouped[group] += counts
        unique, n = torch.unique(a, dim=0, return_counts=True)
        for row, count in zip(unique.cpu().tolist(), n.cpu().tolist()):
            key = ",".join(map(str, row))
            path_counts[key] = path_counts.get(key, 0) + count
    total = edges.sum().clamp_min(1)
    conditional = edges / edges.sum(-1, keepdim=True).clamp_min(1)
    joint = edges.float() / total
    marginal = joint.sum(0)
    predecessor_tv = float(((conditional - marginal).abs().sum(-1) * joint.sum(-1)).sum() / 2)
    g = grouped.float() / grouped.sum((-1, -2), keepdim=True).clamp_min(1)
    return {
        "edge_counts": edges.cpu().tolist(),
        "step_cell_counts": steps.cpu().tolist(),
        "cell_share": (steps.sum(0) / steps.sum()).cpu().tolist(),
        "self_edge_fraction": float(edges.diag().sum() / total),
        "self_fraction_by_predecessor": (edges.diag() / edges.sum(-1).clamp_min(1)).cpu().tolist(),
        "conditional_successors": conditional.cpu().tolist(),
        "input_group_edges": grouped.cpu().tolist(),
        "input_group_threshold_calibration": threshold,
        "input_group_joint_edge_tv": float((g[0] - g[1]).abs().sum() / 2),
        "predecessor_dependence_tv": predecessor_tv,
        "unique_paths": len(path_counts),
        "path_counts": path_counts,
        "causal_specialization": "NOT_ESTABLISHED",
        "small_world": "NOT_TESTED_NOT_CLAIMED",
    }


@torch.no_grad()
def terminal(m, c, data):
    device = next(m.parameters()).device
    before = fingerprint(m.state_dict())
    m.eval()
    for p in m.parameters():
        p.grad = None
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    quality = historical_evaluate(m, windows(data, c, "development", device), m.seed)
    sync(device)
    elapsed = time.perf_counter() - begin
    tokens = sum(r["tokens"] for r in quality.values())
    _, x, _ = next(windows(data, c, "development", device))
    with MatmulAudit(m) as audit:
        out = m(x)
    cost = audit.finish(out)
    if cost["cell_rows"] != 8 * x.numel() or cost["context_rows"] != x.numel():
        raise RuntimeError("EXECUTION_BUDGET_CHANGED")
    counts = out.operator_rows
    cell_params = [sum(p.numel() for p in cell.parameters()) for cell in m.cells]
    shared = sum(
        p.numel() for n, p in m.named_parameters() if not n.startswith(("cells.", "connection."))
    )
    correction = sum(p.numel() for p in m.connection.parameters())
    cost.update(
        total_parameters=sum(p.numel() for p in m.parameters()),
        cell_parameters=sum(cell_params),
        correction_parameters=correction,
        shared_parameters=shared,
        cell_calls_per_token=8,
        context_reads_per_token=1,
        unique_touched_parameters_upper_bound=shared + sum(cell_params) + correction,
        executed_cell_parameter_work_per_token=8 * cell_params[0],
        parameter_scope="Total resident includes unused zero correction in fixed arms; touched bound not executed work",
        evaluation_tokens=tokens,
        evaluation_seconds=elapsed,
        evaluation_tokens_per_second=tokens / elapsed,
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        memory_scope="model plus terminal forward; optimizer released, Torch allocated peak",
        main_evaluation_stage_rows=counts.cpu().tolist(),
    )
    graph = graph_summary(m, c, data)
    replacements = {}
    if m.graph and m.arm_name == "B-C":
        original = m.arm_name
        try:
            for mode in ("B-F", "B-R"):
                m.arm_name = mode
                replacements[mode] = mean_nll(
                    historical_evaluate(m, windows(data, c, "development", device), m.seed)
                )
        finally:
            m.arm_name = original
    if fingerprint(m.state_dict()) != before:
        raise RuntimeError("EVALUATION_MUTATED_MODEL")
    return {
        "development": quality,
        "nll": mean_nll(quality),
        "inference_cost": cost,
        "graph": graph,
        "same_cells_route_replacement_nll": replacements,
    }
