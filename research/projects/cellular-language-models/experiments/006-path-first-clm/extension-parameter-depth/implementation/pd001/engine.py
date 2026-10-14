"""One equivalent optimization step; instruments never enter the objective."""

from contextlib import nullcontext
import torch
from exp006.reuse import make_batch
from pd001.losses import sample_paths, joint_loss, terminal_ce
from pd001.configs import cost_lambda
from pd001.accounting import TrainingGemmCounter
from pd001.diagnostics import gradients, combined_routing


def step(
    model,
    optimizer,
    c,
    index,
    starts,
    tokens,
    generators,
    device,
    *,
    capture=False,
    diagnostics=False,
    dispatch=False,
    routing=False,
    legacy_checks=False,
):
    t = c["training"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()} if diagnostics else None
    lr = t["learning_rate"] * min(1.0, index / max(1, t["warmup_steps"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
    mechanism = c["arms"][model.arm]["mechanism"]
    logs, outputs, totals = [], [], []
    gemm = 0 if dispatch else None
    for start in range(0, t["effective_batch"], t["microbatch"]):
        x, y = make_batch(
            tokens, starts[start : start + t["microbatch"]], t["sequence_length"], torch.device(device)
        )
        with TrainingGemmCounter() if dispatch else nullcontext() as observed:
            out = sample_paths(model, x, generators, mechanism, capture=capture)
            loss, log = joint_loss(out, y, t, mechanism if model.arm != "M0" else "J0", cost_lambda(c, index))
            if legacy_checks and not torch.isfinite(loss):
                raise FloatingPointError("nonfinite joint objective")
            (loss / (t["effective_batch"] // t["microbatch"])).backward()
        if dispatch:
            gemm += observed.flops
        totals.append(loss.detach())
        logs.append(log)
        if diagnostics or routing:
            outputs.extend(out)
    if legacy_checks:
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite gradients")
    elif not torch.isfinite(torch.stack(totals)).all():
        raise FloatingPointError("nonfinite joint objective")
    # One aggregated gradient-norm finite check, rather than a Python bool per parameter.
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), t["gradient_clip"], error_if_nonfinite=not legacy_checks
    )
    legacy_norm = float(norm) if legacy_checks else None
    optimizer.step()
    keys = list(logs[0])
    scalars = torch.stack([torch.stack([r[k] for r in logs]).mean() for k in keys] + [norm.detach()])
    if legacy_checks:
        values = [float(torch.stack([r[k] for r in logs]).mean()) for k in keys] + [legacy_norm]
    else:
        values = scalars.detach().cpu().tolist()
    result = dict(zip(keys, values[:-1]))
    result.update(
        step=index,
        cost_lambda=cost_lambda(c, index),
        learning_rate=lr,
        target_tokens=index * t["effective_batch"] * t["sequence_length"],
        rollout_count=(t["effective_batch"] // t["microbatch"]) * len(generators),
        rollout_target_tokens=index * t["effective_batch"] * t["sequence_length"] * len(generators),
        gradient_norm_before_clip=values[-1],
        observed_training_gemm_flops=gemm,
        training_flops_status="INCOMPLETE_ACCOUNTING" if dispatch else "OFFLINE_ONLY",
        training_backward_flops=None,
        measurement_scope="instrumented supported GEMMs" if dispatch else "not measured during training",
        diagnostics_sampled=diagnostics,
        routing_sampled=routing,
    )
    if diagnostics:
        result.update(gradients(model, before, outputs))
    if routing and model.arm != "M0":
        result["routing"] = combined_routing(outputs, model.config["cell_count"])
    return result


def probes(model, x, y, seed, index, initial_actions=None, legacy=False):
    """Private RNG; no optimizer and no training-generator consumption."""
    model.eval()
    with torch.no_grad():
        greedy = model(x)
        sampled = model(
            x, policy="sampled", generator=torch.Generator(device=x.device).manual_seed(seed + 90000 + index)
        )
        values = (
            [float(terminal_ce(greedy, y).mean()), float(terminal_ce(sampled, y).mean())]
            if legacy
            else torch.stack([terminal_ce(greedy, y).mean(), terminal_ce(sampled, y).mean()]).cpu().tolist()
        )
        result = {
            "train_probe_greedy_nll": values[0],
            "train_probe_sampled_nll": values[1],
            "sampled_greedy_nll_gap": values[1] - values[0],
            "probe_sampled": True,
        }
        if model.arm != "M0":
            actions = greedy.actions.detach().cpu()
            if not legacy or index == 1 or index % 500 == 0:
                result["probe_actions"] = actions.tolist()
            if initial_actions is not None:
                result["route_drift_from_initial"] = float((actions != initial_actions).float().mean())
    return result
