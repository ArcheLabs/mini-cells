"""Inference-only cached-input rollouts and all-15 rerouted counterfactuals."""

from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .common import fingerprint
from .sampling import batches


@dataclass
class Cost:
    input_tokens: int = 0
    context_rows: int = 0
    main_cell_rows: int = 0
    candidate_cell_rows: int = 0
    head_rows: int = 0
    candidate_actions: int = 0
    sampled_states: int = 0
    consistency_full_forwards: int = 0


@torch.inference_mode()
def initial(model, x, cost):
    b, n = x.shape
    d = model.c["model"]["width"]
    positions = torch.arange(n, device=x.device)
    z = model.embedding(x) + model.position(positions)[None]
    memory = model.memory(z)
    h = model.init(z).reshape(-1, d)
    batch = torch.arange(b, device=x.device).repeat_interleave(n)
    h = h + model.c["model"]["alpha"] * model.context(
        model.norm(h), memory, batch, positions.repeat(b)
    )
    cost.input_tokens += x.numel()
    cost.context_rows += x.numel()
    return h, b, n


@torch.inference_mode()
def decision(model, h, t, scale):
    z = model.gate(h, t)
    base = model.connection.base_action(z)
    q, delta = model.connection(model.norm(h), t, z)
    b = z[:, model.connection.pairs].sum(-1)
    scores = b + scale * delta
    if scale == 0:
        action = base
    else:
        action = torch.where((scale * delta == 0).all(-1), base, scores.argmax(-1))
    margin = torch.sort(z, descending=True, dim=-1, stable=True).values
    return action, base, b, q, delta, margin[:, 1] - margin[:, 2]


@torch.inference_mode()
def rollout(model, h0, b, n, scale, cost, *, keep_trace=False, emit_head=True):
    h = h0
    traces = []
    actions = []
    metrics = []
    for t in range(model.c["routing"]["stages"]):
        action, base, bscore, q, delta, margin = decision(model, h, t, scale)
        if keep_trace:
            traces.append(
                {
                    "state": h,
                    "action": action,
                    "base": base,
                    "base_score": bscore,
                    "score": q,
                    "delta": delta,
                }
            )
        actions.append(action)
        metrics.append(
            {
                "local_change": (action != base).float(),
                "gate_margin": margin,
                "delta_abs_mean": delta.abs().mean(-1),
                "delta_abs_max": delta.abs().max(-1).values,
                "scaled_delta_l2": (scale * delta).norm(dim=-1),
            }
        )
        h = model.advance(h, t, action)
        cost.main_cell_rows += 2 * len(h)
    logits = model.head(h.reshape(b, n, -1)) if emit_head else None
    if emit_head:
        cost.head_rows += b * n
    return logits, h, traces, actions, metrics


@torch.inference_mode()
def consistency(model, x, c):
    before = fingerprint(model.state_dict())
    cost = Cost()
    h, b, n = initial(model, x, cost)
    scale = 0.0 if getattr(model, "force_original", False) else 1.0
    logits, last, traces, _, _ = rollout(model, h, b, n, scale, cost, keep_trace=True)
    full = model(x)
    cost.consistency_full_forwards += 1
    cost.input_tokens += x.numel()
    cost.context_rows += x.numel()
    cost.main_cell_rows += 8 * x.numel()
    cost.head_rows += x.numel()
    rtol = c["consistency"]["rtol"]
    atol = c["consistency"]["atol"]
    torch.testing.assert_close(logits, full.logits, rtol=rtol, atol=atol)
    torch.testing.assert_close(last, full.states.reshape_as(last), rtol=rtol, atol=atol)
    for t, trace in enumerate(traces):
        torch.testing.assert_close(trace["state"], full.stages[t].inputs, rtol=rtol, atol=atol)
        indices, weights = model.indices_weights(model.gate(trace["state"], t), trace["action"])
        if not torch.equal(indices, full.stages[t].indices):
            raise RuntimeError("CURRENT_PATH_ACTION_PARITY_FAILED")
        torch.testing.assert_close(weights, full.stages[t].weights, rtol=rtol, atol=atol)
    if fingerprint(model.state_dict()) != before:
        raise RuntimeError("PARAMETERS_CHANGED_DURING_CONSISTENCY")
    return {
        "status": "PASS",
        "rtol": rtol,
        "atol": atol,
        "source_forward_and_advance_parity": True,
        "cost": cost.__dict__,
    }


@torch.inference_mode()
def counterfactual(model, c, original, data, role, selected, device, cost):
    states = [[] for _ in range(4)]
    targets = [[] for _ in range(4)]
    records = [[] for _ in range(4)]
    default = 0.0 if getattr(model, "force_original", False) else 1.0
    for rows, x, y in batches(
        data, original, role, device, c["sampling"]["evaluation_batch_windows"], selected
    ):
        h, b, n = initial(model, x, cost)
        _, _, traces, _, _ = rollout(
            model, h, b, n, default, cost, keep_trace=True, emit_head=False
        )
        for t, trace in enumerate(traces):
            indices = torch.tensor(
                [j * n + r["token_positions"][t] for j, r in enumerate(rows)], device=device
            )
            states[t].append(trace["state"][indices])
            targets[t].append(y.reshape(-1)[indices])
            records[t].append(
                {
                    key: trace[key][indices].cpu().numpy()
                    for key in ("action", "base", "base_score", "score", "delta")
                }
            )
    result = {
        key: [] for key in ("short", "long", "action", "base", "base_score", "score", "delta")
    }
    for t in range(4):
        h = torch.cat(states[t])
        y = torch.cat(targets[t])
        count = len(h)
        actions = len(model.connection.pairs)
        expanded = h.repeat_interleave(actions, dim=0)
        forced = torch.arange(actions, device=device).repeat(count)
        state = model.advance(expanded, t, forced)
        cost.candidate_cell_rows += 2 * len(state)
        target = y.repeat_interleave(actions)
        short = F.cross_entropy(model.head(state), target, reduction="none")
        cost.head_rows += len(state)
        for suffix in range(t + 1, 4):
            chosen = decision(model, state, suffix, default)[0]
            state = model.advance(state, suffix, chosen)
            cost.candidate_cell_rows += 2 * len(state)
        if t == 3:
            long = short.clone()
        else:
            long = F.cross_entropy(model.head(state), target, reduction="none")
            cost.head_rows += len(state)
        if not torch.isfinite(short).all() or not torch.isfinite(long).all():
            raise RuntimeError("NONFINITE_COUNTERFACTUAL_VALUES")
        result["short"].append(short.reshape(count, actions).cpu().numpy())
        result["long"].append(long.reshape(count, actions).cpu().numpy())
        for key in ("action", "base", "base_score", "score", "delta"):
            result[key].append(np.concatenate([r[key] for r in records[t]], axis=0))
        cost.sampled_states += count
        cost.candidate_actions += count * actions
    result = {key: np.stack(values, axis=1) for key, values in result.items()}
    if not np.array_equal(result["short"][:, 3], result["long"][:, 3]):
        raise RuntimeError("LAST_STAGE_HORIZON_IDENTITY_FAILED")
    return result


def distribution(values):
    x = np.asarray(values, float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0, "mean": None, "quantiles": None}
    return {
        "count": len(x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "quantiles": dict(
            zip(
                ("p05", "p25", "p50", "p75", "p95"),
                np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
            )
        ),
    }


@torch.inference_mode()
def interpolation(model, c, original, data, device, cost):
    scales = c["scales"] if not getattr(model, "force_original", False) else [0.0]
    accum = {
        str(scale): {"stories": {}, "stages": [{} for _ in range(4)], "tokens": 0}
        for scale in scales
    }
    for rows, x, y in batches(
        data, original, "development", device, c["sampling"]["evaluation_batch_windows"]
    ):
        h, b, n = initial(model, x, cost)
        baseline = None
        for scale in scales:
            logits, _, _, actions, metrics = rollout(model, h, b, n, scale, cost)
            if scale == 0.0:
                baseline = actions
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
            ).reshape(b, n)
            if not torch.isfinite(loss).all():
                raise RuntimeError("NONFINITE_CLOSED_LOOP_NLL")
            entry = accum[str(scale)]
            entry["tokens"] += y.numel()
            for j, row in enumerate(rows):
                old = entry["stories"].setdefault(row["story"], {"nll_sum": 0.0, "tokens": 0})
                old["nll_sum"] += float(loss[j].sum())
                old["tokens"] += n
            for t, values in enumerate(metrics):
                values = {**values, "trajectory_change": (actions[t] != baseline[t]).float()}
                for name, value in values.items():
                    entry["stages"][t].setdefault(name, []).append(value.cpu().numpy())
    output = {}
    for scale, entry in accum.items():
        stories = {
            k: {**v, "mean_nll": v["nll_sum"] / v["tokens"]} for k, v in entry["stories"].items()
        }
        output[scale] = {
            "mean_nll": sum(r["nll_sum"] for r in stories.values()) / entry["tokens"],
            "stories": stories,
            "tokens": entry["tokens"],
            "stages": [
                {key: distribution(np.concatenate(values)) for key, values in stage.items()}
                for stage in entry["stages"]
            ],
        }
    return output
