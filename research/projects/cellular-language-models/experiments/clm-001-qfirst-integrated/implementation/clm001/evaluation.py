"""Frozen diagnostic policies, independent role windows and paired story-level Admission."""

import math
import numpy as np
import torch
from pd001.losses import terminal_ce
from .oracle import Snapshot
from .common import canonical_sha256


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
    }


def rank_correlation(x, y):
    def ranks(v):
        out = np.empty(len(v))
        v = np.asarray(v)
        for value in np.unique(v):
            idx = np.where(v == value)[0]
            out[idx] = (v < value).sum() + (len(idx) - 1) / 2
        return out

    x, y = ranks(x), ranks(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def q_diagnostics(model, data, c, role="calibration"):
    if role not in ("train", "calibration"):
        raise ValueError("Q diagnostics only Train or Q-calibration")
    model.eval()
    snapshot = Snapshot(model)
    device = next(model.parameters()).device
    length = c["training"]["sequence_length"]
    window = data["windows"][role]
    window = window.get("evaluation", window)
    records = []
    seed = c["evaluation"]["seed"] + 90
    gen = torch.Generator(device=device).manual_seed(seed)
    for i, start in enumerate(window["starts"][: c["evaluation"]["oracle_states"]]):
        x = torch.tensor(
            np.array(data[role][start : start + length], dtype="int64")[None], device=device
        )
        y = torch.tensor(
            np.array(data[role][start + 1 : start + length + 1], dtype="int64")[None], device=device
        )
        depth = min(1, model.config["budget"] - 1)
        # Fixed uniform policy capture obtains states without selecting labels as features.
        out = model(x, policy="uniform", generator=gen, probe_coordinates=[(0, depth)])
        if not out.probes:
            continue
        state = out.probes[0]
        legal = model.registry.legal(
            depth, model.config["budget"], model.config["minimum_actions"], device
        )
        q, _ = model.controller(
            state.hidden[None],
            torch.tensor([state.previous], device=device),
            depth,
            legal,
            model.registry.version,
        )
        suffix_gen = torch.Generator(device=device).manual_seed(seed + i)
        uniforms = torch.rand(
            c["evaluation"]["oracle_samples"],
            model.config["budget"],
            device=device,
            generator=suffix_gen,
        )
        actions = torch.where(legal)[0].tolist()
        labels = [
            snapshot.label(state, a, y[0, 0], uniforms, suffix_seed=seed + i) for a in actions
        ]
        truth = np.array([lab.value for lab in labels])
        pred = q[0, actions].cpu().numpy()
        best = int(truth.argmin())
        chosen = int(pred.argmin())
        gap = float(np.sort(truth)[1] - truth.min()) if len(truth) > 1 else 0.0
        stop_index = actions.index(model.registry.stop) if model.registry.stop in actions else None
        stop_wrong = stop_index is not None and ((chosen == stop_index) != (best == stop_index))
        records.append(
            {
                "mae": float(np.abs(pred - truth).mean()),
                "spearman": rank_correlation(pred, truth),
                "regret": float(truth[chosen] - truth[best]),
                "uniform_regret": float(truth.mean() - truth[best]),
                "constant_action_regret": float(truth[0] - truth[best]),
                "stop_error": bool(stop_wrong),
                "best_gap": gap,
                "stop_too_early": bool(
                    stop_index is not None and chosen == stop_index and best != stop_index
                ),
                "stop_too_late": bool(
                    stop_index is not None and chosen != stop_index and best == stop_index
                ),
                "q_values": pred.tolist(),
                "oracle_values": truth.tolist(),
                "actions": actions,
                "action_bias": {str(a): float(p - v) for a, p, v in zip(actions, pred, truth)},
                "snapshot_sha": snapshot.sha,
                "policy_sha": snapshot.policy_sha,
                "state_sha": state.fingerprint(),
                "role": role,
            }
        )
    valid = [r["spearman"] for r in records if r["spearman"] is not None]
    return {
        "role": role,
        "states": len(records),
        "mae": float(np.mean([r["mae"] for r in records])) if records else None,
        "spearman": float(np.mean(valid)) if valid else None,
        "regret": float(np.mean([r["regret"] for r in records])) if records else None,
        "uniform_regret": float(np.mean([r["uniform_regret"] for r in records]))
        if records
        else None,
        "stop_error_rate": float(np.mean([r["stop_error"] for r in records])) if records else None,
        "identifiable_states": sum(r["best_gap"] > 1e-8 for r in records),
        "records": records,
        "scientific_decision": False,
    }
