"""Token-aggregated frozen windows; normal quality evaluation installs no counters."""

import math
import time
import torch
import numpy as np
from exp006.reuse import make_batch
from pd001.losses import terminal_ce
from .models import make_model
from .train import identity
from .data import load_assets
from .config import authorize
from exp006.reuse import file_sha256
from .io import write_json
import json
from pathlib import Path


@torch.no_grad()
def quality(model, c, assets, *, final=True):
    model.eval()
    device = next(model.parameters()).device
    selected = assets["windows"]["evaluation"]
    count = c["evaluation"]["final_windows"] if final else c["evaluation"]["intermediate_windows"]
    starts, groups = selected["starts"][:count], selected["groups"][:count]
    result = {}
    for policy in ("greedy", "sampled"):
        generator = torch.Generator(device=device).manual_seed(c["evaluation"]["sampled_seed"])
        values = []
        token_sum = 0.0
        num = 0
        for start in starts:
            x, y = make_batch(
                assets["development"], np.array([start]), c["training"]["sequence_length"], device
            )
            out = model(x, policy=policy, generator=generator)
            losses = terminal_ce(out, y)
            total = float(losses.sum())
            token_sum += total
            num += losses.numel()
            values.append(total / losses.numel())
        mean = token_sum / num
        result[policy] = {
            "nll": mean,
            "ppl": math.exp(min(700, mean)),
            "window_nll": values,
            "tokens": num,
            "window_ids": list(starts),
            "groups": list(groups),
        }
    result["sampled_minus_greedy"] = result["sampled"]["nll"] - result["greedy"]["nll"]
    return result


def load_run(c, arm, assets_dir, folder, device):
    authorize(c, device, "evaluate")
    assets = load_assets(c, assets_dir)
    folder = Path(folder)
    receipt = json.loads((folder / "checkpoint.json").read_text())
    if (
        receipt["identity"] != identity(c, arm, assets)
        or file_sha256(folder / "checkpoint.pt") != receipt["sha256"]
    ):
        raise ValueError("checkpoint evaluation identity mismatch")
    model = make_model(c, arm).to(device)
    model.load_state_dict(
        torch.load(folder / "checkpoint.pt", map_location=device, weights_only=False)["model"]
    )
    model.eval()
    return model, assets


def evaluate(c, arm, assets_dir, folder, device):
    model, assets = load_run(c, arm, assets_dir, folder, device)
    result = quality(model, c, assets, final=True)
    write_json(Path(folder) / "evaluation.json", result)
    return result


@torch.no_grad()
def profile(c, arm, assets_dir, folder, device):
    from pd001.accounting import Counter

    model, assets = load_run(c, arm, assets_dir, folder, device)
    x, y = make_batch(
        assets["development"],
        np.array(assets["windows"]["evaluation"]["starts"][:1]),
        c["training"]["sequence_length"],
        torch.device(device),
    )
    with Counter(model) as counter:
        out = model(x)
    cost = counter.report(x.numel())
    prefix = x[:, :-1]
    cache = model.prefill(prefix).memory
    values = []
    for _ in range(c["evaluation"]["decode_repetitions"] + 3):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        model.decode_step(x[:, -1], cache)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        values.append((time.perf_counter() - tick) * 1000)
    result = {
        "identity": identity(c, arm, assets),
        "cost": cost,
        "decode_ms_p50": float(np.median(values[3:])),
        "decode_scope": "single next token with reusable prefilled prefix; not growing generation",
        "scientific_decision": False,
    }
    if arm != "M0-TINY":
        from pd001.diagnostics import routing

        route = routing(out, c["model"]["cell_count"])
        route.pop("forced_stop_fraction", None)
        route.pop("stop_probability", None)
        route["router_entropy_per_decision"] = route["entropy_per_token"] / c["arms"][arm]["max_actions"]
        route["ctx_read_frequency"] = route["ctx_read_calls"] / out.actions.numel()
        result["routing"] = route
    write_json(Path(folder) / "hardware-profile.json", result)
    return result
