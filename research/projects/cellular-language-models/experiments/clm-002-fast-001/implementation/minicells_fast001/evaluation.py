"""Fixed Confirm inputs, hard routing, fixed-path Cell counterfactuals."""

import time
from collections import Counter

import torch
from minicells_clm002.measurement import evaluation_mode, nll, replay, specialization
from minicells_r3e.common import fingerprint

from .data import evaluation_rows


@torch.no_grad()
def evaluate(model, c, folder, *, mechanism):
    device = next(model.parameters()).device
    before = fingerprint(model.state_dict())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    forward_calls, target_evaluations = 0, 0
    tasks = {}
    with evaluation_mode(model):
        for row in evaluation_rows(c, folder, device):
            x, y = row["x"], row["y"]
            out = model(x)
            forward_calls += 1
            target_evaluations += y.numel()
            loss = float(nll(out.logits, y, row["valid"]))
            task = tasks.setdefault(
                row["task"],
                {
                    "loss_sum": 0.0,
                    "targets": 0,
                    "fixed_sum": [0.0] * len(model.cells),
                    "counts": [0] * len(model.cells),
                    "paths": Counter(),
                },
            )
            size = y.numel()
            task["loss_sum"] += loss * size
            task["targets"] += size
            counts = (
                torch.bincount(out.actions.flatten(), minlength=len(model.cells)).cpu().tolist()
            )
            task["counts"] = [a + b for a, b in zip(task["counts"], counts)]
            task["paths"].update(tuple(path) for path in out.actions.cpu().tolist())
            if mechanism:
                for cell in range(len(model.cells)):
                    if counts[cell]:
                        ablated = replay(model, x, out.actions, cell)
                        forward_calls += 1
                        target_evaluations += y.numel()
                        task["fixed_sum"][cell] += (
                            float(nll(ablated.logits, y, row["valid"])) * size
                        )
                    else:
                        task["fixed_sum"][cell] += loss * size
    result = {}
    for name, task in tasks.items():
        mean = task["loss_sum"] / task["targets"]
        value = {
            "nll": mean,
            "tokens": task["targets"],
            "call_share": [v / sum(task["counts"]) for v in task["counts"]],
            "call_counts": task["counts"],
            "complete_route_max_mass": max(task["paths"].values()) / sum(task["paths"].values()),
            "unique_complete_routes": len(task["paths"]),
        }
        if mechanism:
            value["D"] = [
                (v / task["targets"] - mean) / max(abs(mean), 1e-12) for v in task["fixed_sum"]
            ]
        result[name] = value
    if fingerprint(model.state_dict()) != before:
        raise RuntimeError("FAST_CONFIRM_EVALUATION_MUTATED_MODEL")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "tasks": result,
        "L": sum(result[k]["nll"] for k in ("Story", "Arithmetic")) / 2,
        "specialization": specialization(result, c["stage_a"]["cell_call_share_min"])
        if mechanism
        else None,
        "single_route_collapse": any(
            v["complete_route_max_mass"] >= c["stage_a"]["route_mass_collapse"]
            for v in result.values()
        ),
        "scope": "FROZEN_CONTROLLED_CONFIRM_ONLY",
        "diagnostic_seconds": time.perf_counter() - started,
        "diagnostic_forward_calls": forward_calls,
        "diagnostic_target_evaluations": target_evaluations,
    }
