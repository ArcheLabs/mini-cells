"""Per-story paired quality and real reexecution interventions, not local probes."""

import math
import numpy as np
import torch
from exp006.reuse import make_batch
from .accounting import Counter
from .losses import terminal_ce
from .diagnostics import routing


@torch.no_grad()
def evaluate_windows(model, tokens, starts, groups, length, *, policy="greedy", seed=1907, intervention=None):
    if len(starts) != len(groups) or not starts:
        raise ValueError("paired evaluation windows required")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    nlls, ids, routes = [], [], []
    with Counter(model) as counter:
        for start in starts:
            x, y = make_batch(tokens, np.asarray([start]), length, device)
            out = model(x, policy=policy, generator=generator, **(intervention or {}))
            nlls.append(float(terminal_ce(out, y).mean()))
            ids.append(int(start))
            if model.arm != "M0":
                routes.append(routing(out, model.config["cell_count"]))
    mean = sum(nlls) / len(nlls)
    return {
        "nll": mean,
        "ppl": math.exp(min(mean, 700)),
        "window_nll": nlls,
        "window_ids": ids,
        "groups": list(groups),
        "routing": routes,
        "cost": counter.report(len(starts) * length),
    }


@torch.no_grad()
def intervention_pair(model, x, y, kind, *, step=1, seed=1911):
    if model.arm == "M0":
        raise ValueError("path interventions are not applied to MoE")
    model.eval()
    c = model.config
    with Counter(model) as before:
        original = model(x)
    replay = original.actions.clone()
    kwargs = {}
    semantics = "policy_replan"
    generator = torch.Generator(device=x.device).manual_seed(seed)
    if kind == "stop_early":
        kwargs["stop_after"] = step
    elif kind == "extra_steps":
        if not 0 <= step < c["budget"]:
            raise ValueError("extension must stay inside registered budget")
        # Force STOP ineligible for the prefix, then replan every remaining action.
        kwargs["minimum_actions"] = step
    elif kind == "edge_zero":
        kwargs["edge_mode"] = "zero"
    elif kind == "edge_shuffle":
        kwargs.update(
            edge_mode="shuffle",
            edge_permutation=torch.randperm(c["cell_count"] + 3, generator=generator, device=x.device),
        )
    elif kind == "context_zero_replay":
        kwargs.update(replay=replay, context_zero=True)
        semantics = "fixed_action_replay"
    elif kind == "order_permutation":
        for row in replay.reshape(-1, c["budget"]):
            valid = torch.where((row >= 0) & (row < c["cell_count"] + 1))[0]
            row[valid] = row[valid][torch.randperm(valid.numel(), generator=generator, device=x.device)]
        kwargs["replay"] = replay
        semantics = "fixed_action_replay"
    elif kind == "self_loop_replacement":
        # Token-wise override changes one valid repeated Cell, then replans the suffix.
        edits = {}
        for t in range(1, c["budget"]):
            repeated = (
                (original.actions[:, :, t] == original.actions[:, :, t - 1])
                & (original.actions[:, :, t] >= 0)
                & (original.actions[:, :, t] < c["cell_count"])
            )
            if repeated.any():
                replacement = (
                    torch.full_like(original.actions[:, :, t], c["cell_count"])
                    if c.get("self_only", False)
                    else (original.actions[:, :, t] + 1) % c["cell_count"]
                )
                edits[t] = (repeated, replacement)
                break
        kwargs["token_overrides"] = edits
        kwargs["replay_prefix"] = original.actions
        kwargs["prefix_length"] = next(iter(edits), 0)
    else:
        raise ValueError("unknown intervention")
    with Counter(model) as after:
        changed = model(x, **kwargs)
    delta = (terminal_ce(changed, y) - terminal_ce(original, y)).reshape(-1)
    return {
        "kind": kind,
        "semantics": semantics,
        "nll_delta_per_token": delta.cpu().tolist(),
        "nll_after_minus_before": float(delta.mean()),
        "delta_forward_gemm_flops": after.total - before.total,
        "before_cost": before.report(x.numel()),
        "after_cost": after.report(x.numel()),
        "paired_tokens": x.numel(),
        "diagnostic_only": True,
        "scientific_decision": False,
    }


@torch.no_grad()
def intervention_suite(model, tokens, starts, groups, length, kind, *, step=1, seed=1911, replicates=5000):
    from .analysis import paired_interval

    device = next(model.parameters()).device
    rows = []
    for i, start in enumerate(starts):
        x, y = make_batch(tokens, np.asarray([start]), length, device)
        row = intervention_pair(model, x, y, kind, step=step, seed=seed + i)
        row.update(window_id=int(start), story_id=groups[i])
        rows.append(row)
    changes = [row["nll_after_minus_before"] for row in rows]
    interval = paired_interval([0.0] * len(rows), changes, groups, seed, replicates)
    return {
        "kind": kind,
        "semantics": rows[0]["semantics"],
        "paired_windows": len(rows),
        "story_paired_nll_interval": interval,
        "mean_delta_forward_gemm_flops": sum(row["delta_forward_gemm_flops"] for row in rows) / len(rows),
        "rows": rows,
        "scientific_decision": False,
        "diagnostic_only": True,
    }
