"""Bounded real reexecution, symmetric S8/S16, never new trained arms."""

import torch
import numpy as np
from pathlib import Path
from exp006.reuse import make_batch
from .io import write_json
from pd001.accounting import Counter
from pd001.losses import terminal_ce
from pd001.evaluation import intervention_pair
from .evaluate import load_run
from .train import identity


@torch.no_grad()
def pair(model, x, y, kind, step=4, seed=151000105):
    model.eval()
    if kind not in ("edge_shuffle", "self_loop_replacement", "truncate"):
        row = intervention_pair(model, x, y, kind, step=step, seed=seed)
        row.update(applicable_tokens=x.numel(), intervention_seed=seed)
        return row
    c = model.config
    with Counter(model) as before:
        original = model(x)
    kwargs = {}
    selected = torch.ones_like(x, dtype=torch.bool)
    if kind == "edge_shuffle":
        generator = torch.Generator(device=x.device).manual_seed(seed)
        kwargs.update(
            edge_mode="shuffle",
            edge_permutation=torch.randperm(
                model.controller.actions + 1, generator=generator, device=x.device
            ),
        )
        meaning = "full random predecessor row permutation; suffix replanned"
    elif kind == "truncate":
        if step not in (4, 8, 12, 16) or step > c["budget"]:
            raise ValueError("illegal truncated readout depth")
        kwargs.update(replay=original.actions[:, :, :step] if step == c["budget"] else None, stop_after=step)
        # Replay has full budget shape; pad inactive suffix to match parent contract.
        replay = original.actions.clone()
        replay[:, :, step:] = -1
        kwargs["replay"] = replay
        meaning = "fixed prefix early terminal readout; not a trained STOP or depth control"
    else:
        selected = torch.zeros_like(x, dtype=torch.bool)
        edits = {}
        for t in range(1, c["budget"]):
            repeated = (
                (original.actions[:, :, t] == original.actions[:, :, t - 1])
                & (original.actions[:, :, t] < c["cell_count"])
                & ~selected
            )
            if repeated.any():
                # Each token gets its first repeated Cell replaced exactly once.
                edits[t] = (repeated, (original.actions[:, :, t] + 1) % c["cell_count"])
                selected |= repeated
        # Per-token prefix preservation, then independent suffix replan, in one call/token.
        deltas = []
        works = []
        details = []
        eligible = int(selected.sum())
        for b, n in selected.nonzero().cpu().tolist()[:8]:
            time_index = next(t for t, (mask, _) in edits.items() if bool(mask[b, n]))
            mask = torch.zeros_like(selected)
            mask[b, n] = True
            replacement = (original.actions[:, :, time_index] + 1) % c["cell_count"]
            with Counter(model) as after:
                changed = model(
                    x,
                    replay_prefix=original.actions,
                    prefix_length=time_index,
                    token_overrides={time_index: (mask, replacement)},
                )
            deltas.append(float((terminal_ce(changed, y) - terminal_ce(original, y))[b, n]))
            works.append(after.total - before.total)
            details.append({"batch": b, "token": n, "edited_step": time_index})
        return {
            "kind": kind,
            "semantics": "per-token first repeated Cell replaced, prefix fixed, suffix replanned",
            "applicable_tokens": len(deltas),
            "observed_tokens": x.numel(),
            "eligible_tokens": eligible,
            "max_evaluated_tokens": 8,
            "nll_delta_per_token": deltas,
            "nll_after_minus_before": float(np.mean(deltas)) if deltas else None,
            "delta_forward_gemm_flops": float(np.mean(works)) if works else None,
            "compute_scope": "full-window real reexecution per edited token; cost not causal token-only work",
            "edits": details,
            "intervention_seed": seed,
            "scientific_decision": False,
            "diagnostic_only": True,
        }
    with Counter(model) as after:
        changed = model(x, **kwargs)
    delta = (terminal_ce(changed, y) - terminal_ce(original, y)).cpu().flatten().tolist()
    return {
        "kind": kind,
        "semantics": meaning,
        "nll_delta_per_token": delta,
        "nll_after_minus_before": float(np.mean(delta)),
        "applicable_tokens": x.numel(),
        "state_norm_before": float(original.states.norm(dim=-1).mean()),
        "state_norm_after": float(changed.states.norm(dim=-1).mean()),
        "delta_forward_gemm_flops": after.total - before.total,
        "before_cost": before.report(x.numel()),
        "after_cost": after.report(x.numel()),
        "intervention_seed": seed,
        "scientific_decision": False,
        "diagnostic_only": True,
    }


def suite(c, arm, assets_dir, folder, device):
    if arm == "M0-TINY":
        raise ValueError("no path intervention for MoE")
    model, assets = load_run(c, arm, assets_dir, folder, device)
    rows = []
    starts = assets["windows"]["evaluation"]["starts"][: c["evaluation"]["intervention_windows"]]
    groups = assets["windows"]["evaluation"]["groups"][: len(starts)]
    for i, start in enumerate(starts):
        x, y = make_batch(
            assets["development"], np.array([start]), c["training"]["sequence_length"], torch.device(device)
        )
        for kind in (
            "edge_zero",
            "edge_shuffle",
            "order_permutation",
            "self_loop_replacement",
            "context_zero_replay",
            "truncate",
        ):
            for depth in (
                [d for d in (4, 8, 12, 16) if d <= model.config["budget"]] if kind == "truncate" else [4]
            ):
                row = pair(model, x, y, kind, depth, c["evaluation"]["intervention_seed"] + i)
                row.update(
                    window_id=int(start),
                    story_id=groups[i],
                    readout_depth=depth if kind == "truncate" else None,
                )
                rows.append(row)
    result = {
        "identity": identity(c, arm, assets),
        "arm": arm,
        "rows": rows,
        "scientific_decision": False,
        "limitations": [
            "single development seed; destructive interventions do not establish superiority",
            "nonlinear operator order is generally noncommutative",
            "self-loop analysis applies only to tokens with a genuine repeated Cell",
        ],
    }
    write_json(Path(folder) / "interventions.json", result)
    return result
