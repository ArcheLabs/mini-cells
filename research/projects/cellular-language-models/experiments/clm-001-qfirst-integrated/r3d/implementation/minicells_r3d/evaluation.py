"""Terminal-only closed-loop quality and costs, without CF or exploration."""

import time
import numpy as np
import torch
from minicells_r3c_moe_bridge.evaluation import evaluate
from minicells_r3c_moe_bridge.data import fixed_windows
from minicells_r3c_moe_bridge.instrumentation import MatmulAudit
from .common import fingerprint


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def final_evaluation(model, c, data, device):
    before = fingerprint(model.state_dict())
    sync(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    quality = {
        r: evaluate(model, fixed_windows(data, c["engine"], r, device), model.seed + 99001)
        for r in ("calibration", "development")
    }
    sync(device)
    elapsed = time.perf_counter() - begin
    _, x, _ = next(iter(fixed_windows(data, c["engine"], "development", device)))
    with MatmulAudit(model) as audit:
        out = model(x)
    cost = audit.finish(out)
    if cost["cell_rows"] != 8 * x.numel() or cost["context_rows"] != x.numel():
        raise RuntimeError("inference budget changed")
    tokens = sum(v["tokens"] for role in quality.values() for v in role.values())
    cost.update(
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "CPU",
        torch_version=str(torch.__version__),
        evaluation_seconds=elapsed,
        evaluation_tokens=tokens,
        evaluation_tokens_per_second=tokens / elapsed,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
        timing_scope="all terminal Calibration/Development forward+CE+Story aggregation; not decode latency",
        memory_scope="Torch allocated peak for model plus terminal evaluation; training optimizer and gradients released; not reserved memory or entire-device VRAM",
        correction_parameters=sum(p.numel() for p in model.connection.parameters()),
        active_correction_parameters=0
        if getattr(model, "force_original", False)
        else sum(p.numel() for p in model.connection.parameters()),
        deployable_parameters=sum(p.numel() for p in model.parameters())
        - (
            sum(p.numel() for p in model.connection.parameters())
            if getattr(model, "force_original", False)
            else 0
        ),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )
    if fingerprint(model.state_dict()) != before:
        raise RuntimeError("evaluation changed parameters")
    return quality, cost


def mean_nll(rows):
    return float(
        np.average(
            [v["mean_nll"] for v in rows.values()], weights=[v["tokens"] for v in rows.values()]
        )
    )


def exposure(curve, c):
    if not curve:
        return {
            "cell_updates": 0,
            "zero_gradient_step_fraction": None,
            "persistent_zero_exposure": None,
        }
    a = np.asarray([r["stage_cell_counts"] for r in curve])
    g = np.asarray([r["cell_grad_norms"] for r in curve])
    width = c["statistics"]["zero_exposure_window"]
    rolling = [a[i : i + width].sum((0, 1)) for i in range(max(0, len(a) - width + 1))]
    persistent = any(bool((v == 0).any()) for v in rolling)
    return {
        "cell_updates": sum(r["cell_optimizer_updated"] for r in curve),
        "observed_steps": len(curve),
        "stage_cell_counts": a.sum(0).tolist(),
        "stage_pair_counts": np.asarray([r["stage_pair_counts"] for r in curve]).sum(0).tolist(),
        "cell_counts": a.sum((0, 1)).tolist(),
        "zero_gradient_step_fraction": float((g == 0).any(1).mean()),
        "mean_cell_gradient_norms": g.mean(0).tolist(),
        "persistent_zero_exposure": persistent,
        "rolling_window": width,
        "rolling_windows": len(rolling),
        "mean_action_change_fraction": float(np.mean([r["action_change_fraction"] for r in curve])),
    }


def train_cost(curve, c, per_token_forward_matmul):
    m = c["engine"]["model"]
    d = m["width"]
    hidden = m["cell_hidden"]
    gate = m["router_hidden"]
    step = m["step_width"]
    v = m["vocab_size"]
    work = {
        key: sum(r["counterfactual_work"].get(key, 0) for r in curve)
        for key in (
            "counterfactual_cell_rows",
            "counterfactual_gate_rows",
            "counterfactual_head_rows",
        )
    }
    cf = (
        work["counterfactual_cell_rows"] * 4 * d * hidden
        + work["counterfactual_gate_rows"] * 2 * ((d + step) * gate + gate * 6 + (d + 4) * 15)
        + work["counterfactual_head_rows"] * 2 * d * v
    )
    tokens = curve[-1]["tokens"] if curve else 0
    # Common R3c analytic per-token forward: all fixed/shared linear and QK/AV terms.
    main = tokens * per_token_forward_matmul
    return {
        **work,
        "main_training_tokens": tokens,
        "cell_optimizer_updates": sum(r["cell_optimizer_updated"] for r in curve),
        "main_forward_matmul_estimate": main,
        "counterfactual_forward_matmul_estimate_upper_bound": cf,
        "training_forward_plus_backward_proxy_upper_bound": 3 * (main + cf),
        "scope": "analytic matmul proxy; includes repeated executed Cell work; excludes nonmatmul and optimizer; not hardware FLOPs",
        "counterfactual_scope": "upper bound: charges correction per scored-stage gate even when forced action skips it",
        "summed_update_seconds": sum(r["update_wall_seconds"] for r in curve),
        "policy_optimizer_updates": curve[-1]["policy_updates"] if curve else 0,
    }
