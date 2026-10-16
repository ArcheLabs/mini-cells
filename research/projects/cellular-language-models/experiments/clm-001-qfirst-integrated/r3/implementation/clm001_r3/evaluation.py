import math
import numpy as np
import torch
from pd001.losses import terminal_ce
from .common import canonical_sha256
from .audit import capture, measure

__all__ = ["evaluate", "q_diagnostics"]


@torch.no_grad()
def evaluate(
    model, data, c, role, policy="greedy", *, seed=None, edge_zero=False, intervention=None
):
    if role not in ("calibration", "admission", "development"):
        raise ValueError("evaluation role is not allowed")
    model.eval()
    device = next(model.parameters()).device
    meta = data["windows"][role]["evaluation"]
    length = c["training"]["sequence_length"]
    fixed = c["evaluation"]["seed"] if seed is None else seed
    rng = torch.Generator(device=device).manual_seed(fixed)
    nll = []
    cost = []
    stops = []
    forced = []
    traces = []
    for start in meta["starts"]:
        x = torch.tensor(
            np.array(data[role][start : start + length], dtype="int64")[None], device=device
        )
        y = torch.tensor(
            np.array(data[role][start + 1 : start + length + 1], dtype="int64")[None], device=device
        )
        out = model(x, policy=policy, generator=rng, edge_zero=edge_zero, intervention=intervention)
        nll.append(float(terminal_ce(out, y).mean()))
        cost.append(float(out.cost_units.mean()))
        stops.append(float(out.actual_stop.float().mean()))
        forced.append(float(out.forced_end.float().mean()))
        traces.extend(out.actions[0].cpu().tolist())
    selfloops = 0
    transitions = 0
    ctx = 0
    total = 0
    for path in traces:
        active = [a for a in path if a >= 0 and a != model.registry.stop]
        total += len(active)
        ctx += active.count(model.registry.ctx)
        for a, b in zip(active, active[1:]):
            transitions += 1
            selfloops += int(a == b and a < model.registry.slots)
    identity = canonical_sha256(
        {
            "role": role,
            "windows": meta,
            "seed": fixed,
            "budget": model.config["budget"],
            "policy": policy,
            "cost_lambda": c["policy"]["lambda_cost"],
        }
    )
    mean = float(np.mean(nll))
    cell_counts = {
        str(a): sum(path.count(a) for path in traces) for a in range(model.registry.slots)
    }
    cell_total = sum(cell_counts.values())
    dwell = []
    transition_counts = {}
    for path in traces:
        active = [a for a in path if a >= 0 and a != model.registry.stop]
        for a, b in zip(active, active[1:]):
            transition_counts[str(a) + "->" + str(b)] = (
                transition_counts.get(str(a) + "->" + str(b), 0) + 1
            )
        previous = None
        run = 0
        for a in active + [-1]:
            if a == previous and a < model.registry.slots:
                run += 1
            else:
                if run:
                    dwell.append(run)
                previous = a
                run = 1 if 0 <= a < model.registry.slots else 0
    return {
        "nll": nll,
        "cost": cost,
        "groups": meta["groups"],
        "mean_nll": mean,
        "ppl": math.exp(min(mean, 700)),
        "actual_stop_rate": float(np.mean(stops)),
        "forced_end_rate": float(np.mean(forced)),
        "self_loop_rate": selfloops / max(1, transitions),
        "context_call_fraction": ctx / max(1, total),
        "paired_identity": identity,
        "role": role,
        "policy": policy,
        "path_count": len(traces),
        "cell_exposure": cell_counts,
        "transition_matrix": transition_counts,
        "mean_cell_dwell_time": sum(dwell) / len(dwell) if dwell else None,
        "max_cell_share": max(cell_counts.values()) / cell_total if cell_total else None,
        "mean_path_length": total / len(traces) if traces else None,
        "context_read_before_stop_fraction": sum(model.registry.ctx in path for path in traces)
        / len(traces)
        if traces
        else None,
    }


def q_diagnostics(model, data, c, role="calibration"):
    if role != "calibration":
        raise ValueError("independent Calibration only")
    bank, missing = capture(model, data, c, role)
    result = measure(model, bank, c, resampled=True)
    records = result["state_records"]
    return {
        "role": role,
        "states": len(records),
        "coverage_missing": len(missing),
        "token_positions": sorted(set(row["position"] for row in records)),
        "ctx_seen_states": sum(item["ctx_seen"] for item in bank),
        "mean_cell_router_entropy": sum(row["cell_router_entropy"] for row in records)
        / len(records)
        if records
        else None,
        "router_choice_agreement": sum(r["router_choice_agreement"] for r in records) / len(records)
        if records
        else None,
        "paired_regret": sum(r["paired_regret"] for r in records) / len(records)
        if records
        else None,
        "local_vs_long_horizon": result["summary"],
        "records": records,
        "scientific_decision": False,
    }
