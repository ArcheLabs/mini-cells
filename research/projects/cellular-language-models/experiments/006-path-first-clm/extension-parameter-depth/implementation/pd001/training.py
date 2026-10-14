"""One joint optimizer, no component pretraining; train gate precedes all assets."""

import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from exp006.reuse import canonical_sha256, file_sha256, make_batch, write_json
from .configs import authorize, arm_config, cost_lambda
from .models import make_model
from .losses import joint_loss, sample_paths
from .diagnostics import gradients, combined_routing
from .accounting import Counter, TrainingGemmCounter
from .checkpoint import save, restore
from .assets import load_bundle
from .evaluation import evaluate_windows
from .analysis import baseline_review


def training_budget_sha(c):
    return canonical_sha256({k: v for k, v in c["training"].items() if k not in ("mechanism", "microbatch")})


def identity(c, p, arm, seed, role, source, assets):
    return {
        "experiment_id": "006-PD1",
        "arm": arm,
        "source_sha": source,
        "config_sha": canonical_sha256(c),
        "protocol_sha": canonical_sha256(p),
        "asset_sha": assets["manifest_sha256"],
        "tokenizer_sha": assets["tokenizer_sha256"],
        "seed": seed,
        "role": role,
        "target_tokens": c["training"]["tokens"],
        "training_budget_sha": training_budget_sha(c),
    }


def optimize(model, x, y, opt, c, step, generators):
    mechanism = c["arms"][model.arm]["mechanism"]
    outputs = sample_paths(model, x, generators, mechanism, capture=True)
    total, log = joint_loss(
        outputs, y, c["training"], mechanism if model.arm != "M0" else "J0", cost_lambda(c, step)
    )
    return total, log, outputs


def train(
    c, p, arm, seed, role, source, assets_path, output, device="cpu", resume=False, confirmation_assets=None
):
    authorize(c, p, arm, seed, role, "train", source, device)  # no data load, output or GPU before this
    if role == "confirmation" and confirmation_assets is None:
        raise RuntimeError("a separate sealed confirmation bundle is required")
    assets = load_bundle(assets_path, c, p, arm, seed, role, source, evaluation=False)
    return _train_registered(
        c, p, arm, seed, role, source, assets, assets_path, output, device, resume, confirmation_assets
    )


def _train_registered(
    c, p, arm, seed, role, source, assets, assets_path, output, device, resume, confirmation_assets=None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(c["evaluation"]["threads"])
    model = make_model(arm_config(c, arm), arm, seed).to(device)
    if c["training"]["precision"] != "float32":
        raise RuntimeError(
            "only audited float32 training is implemented; mixed precision needs a separately validated registration"
        )
    opt = torch.optim.AdamW(
        model.parameters(), lr=c["training"]["learning_rate"], weight_decay=c["training"]["weight_decay"]
    )
    rng = np.random.default_rng(seed)
    mechanism = c["arms"][arm]["mechanism"]
    generators = [
        torch.Generator(device=device).manual_seed(seed + 100 + i)
        for i in range(1 if mechanism == "J0" or arm == "M0" else 2)
    ]
    run_id = identity(c, p, arm, seed, role, source, assets)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("run exists; resume preserves identity")
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint.pt"
    if resume and (output / "metrics.json").exists():
        complete = json.loads((output / "metrics.json").read_text())
        if any(complete["identity"].get(k) != v for k, v in run_id.items()) or complete[
            "checkpoint_sha"
        ] != file_sha256(latest):
            raise ValueError("completed evidence identity mismatch")
        return complete
    first = 0
    history = []
    prior_flops = 0
    prior_elapsed = 0.0
    prior_training_gemm = 0
    if resume:
        first, extra = restore(latest, model, opt, run_id, rng, generators, device=device)
        history = extra["history"]
        prior_flops = extra["forward_gemm"]
        prior_elapsed = extra["elapsed_seconds"]
        prior_training_gemm = extra.get("observed_training_gemm", 0)
    t = c["training"]
    started = time.perf_counter()
    step = first
    probe_x, probe_y = make_batch(
        assets["train"], np.asarray(assets["train_starts"][:1]), t["sequence_length"], torch.device(device)
    )
    model.eval()
    with torch.no_grad():
        initial_probe = model(probe_x)
        initial_actions = initial_probe.actions.detach().cpu() if arm != "M0" else None
    if resume:
        initial_actions = extra.get("initial_probe_actions", initial_actions)
    try:
        with Counter(model) as counter:
            for step in range(first + 1, t["steps"] + 1):
                model.train()
                opt.zero_grad(set_to_none=True)
                chosen = rng.choice(assets["train_starts"], size=t["effective_batch"], replace=True)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                logs = []
                all_outputs = []
                training_gemm = 0
                lr = t["learning_rate"] * min(1.0, step / max(1, t["warmup_steps"]))
                for group in opt.param_groups:
                    group["lr"] = lr
                for start in range(0, t["effective_batch"], t["microbatch"]):
                    x, y = make_batch(
                        assets["train"],
                        chosen[start : start + t["microbatch"]],
                        t["sequence_length"],
                        torch.device(device),
                    )
                    with TrainingGemmCounter() as observed:
                        total, log, outputs = optimize(model, x, y, opt, c, step, generators)
                        if not torch.isfinite(total):
                            raise FloatingPointError("nonfinite joint objective")
                        (total / (t["effective_batch"] // t["microbatch"])).backward()
                    training_gemm += observed.flops
                    logs.append(log)
                    all_outputs.extend(outputs)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite gradients")
                preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"]))
                opt.step()
                prior_training_gemm += training_gemm
                log = {k: float(torch.stack([r[k] for r in logs]).mean()) for k in logs[0]}
                log.update(
                    step=step,
                    cost_lambda=cost_lambda(c, step),
                    learning_rate=lr,
                    forward_gemm_flops=prior_flops + counter.total,
                    training_flops_status="INCOMPLETE_ACCOUNTING",
                    training_backward_flops=None,
                    observed_training_gemm_flops=prior_training_gemm,
                    training_observed_scope="mm/addmm/bmm forward and backward; excludes fused SDPA and non-GEMM",
                    rollout_count=len(all_outputs),
                    target_tokens=step * t["effective_batch"] * t["sequence_length"],
                    rollout_target_tokens=step
                    * t["effective_batch"]
                    * t["sequence_length"]
                    * len(generators),
                    gradient_norm_before_clip=preclip,
                    **gradients(model, before, all_outputs),
                )
                if arm != "M0":
                    log["routing"] = combined_routing(all_outputs, model.config["cell_count"])
                # Fixed train-derived probe only: never touch confirmation in intermediate diagnostics.
                model.eval()
                counter.enabled = False
                with torch.no_grad():
                    probe = model(probe_x)
                    from .losses import terminal_ce

                    log["train_probe_greedy_nll"] = float(terminal_ce(probe, probe_y).mean())
                    sampled = model(
                        probe_x,
                        policy="sampled",
                        generator=torch.Generator(device=device).manual_seed(seed + 90000 + step),
                    )
                    log["train_probe_sampled_nll"] = float(terminal_ce(sampled, probe_y).mean())
                    log["sampled_greedy_nll_gap"] = (
                        log["train_probe_sampled_nll"] - log["train_probe_greedy_nll"]
                    )
                    if arm != "M0":
                        if step == 1 or step % t["evaluate_every"] == 0:
                            log["probe_actions"] = probe.actions.cpu().tolist()
                        log["route_drift_from_initial"] = float(
                            (probe.actions.cpu() != initial_actions).float().mean()
                        )
                counter.enabled = True
                history.append(log)
                if step == 1 or step % 100 == 0:
                    print(
                        json.dumps(
                            {
                                k: log[k]
                                for k in ("step", "ce", "actor_loss", "critic_loss", "forward_gemm_flops")
                            }
                        ),
                        flush=True,
                    )
                if step % t["checkpoint_every"] == 0 or step == t["steps"]:
                    save(
                        latest,
                        model,
                        opt,
                        step,
                        run_id,
                        rng,
                        generators,
                        {
                            "history": history,
                            "forward_gemm": prior_flops + counter.total,
                            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
                            "observed_training_gemm": prior_training_gemm,
                            "initial_probe_actions": initial_actions,
                        },
                    )
                    write_curve(output, history)
            if step != t["steps"]:
                raise RuntimeError("final step missing")
            checkpoint_sha = file_sha256(latest)
            frozen = {"sha": checkpoint_sha, "step": step, "complete": True}
            # Final-only loader happens after immutable checkpoint hash is fixed.
            counter.enabled = False
            eval_assets = load_bundle(
                confirmation_assets if role == "confirmation" else assets_path,
                c,
                p,
                arm,
                seed,
                role,
                source,
                evaluation=True,
                frozen_checkpoint=frozen,
            )
            greedy = evaluate_windows(
                model,
                eval_assets["tokens"],
                eval_assets["starts"],
                eval_assets["groups"],
                t["sequence_length"],
            )
            sampled = evaluate_windows(
                model,
                eval_assets["tokens"],
                eval_assets["starts"],
                eval_assets["groups"],
                t["sequence_length"],
                policy="sampled",
                seed=c["evaluation"]["sampled_seed"],
            )
            run_id["evaluation_sha"] = eval_assets["evaluation_sha"]
            result = {
                "identity": run_id,
                "status": "DEVELOPMENT_COMPLETED" if role == "development" else "CONFIRMATION_MEASURED",
                "scientific_decision": False,
                "checkpoint_sha": checkpoint_sha,
                "baseline_stable": baseline_review(history, p["thresholds"].get("baseline_stability"))
                if arm == "M0"
                else None,
                "quality": {"greedy": greedy, "sampled": sampled},
                "cost": counter.report(step * t["effective_batch"] * t["sequence_length"]),
                "routing": greedy["routing"],
                "stability": history[-1],
                "interventions": [],
                "statistics": {},
                "device": str(device),
                "runtime": {"torch": torch.__version__, "precision": "float32"},
                "authorization": {"registered": True, "role": role},
                "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
                "gpu_hours": (time.perf_counter() - started) / 3600
                if str(device).startswith("cuda")
                else 0.0,
            }
            write_json(output / "metrics.json", result)
            return result
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {
                "identity": run_id,
                "status": "INCOMPLETE",
                "scientific_decision": False,
                "step": step,
                "checkpoint_exists": latest.exists(),
                "exception": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def write_curve(output, history):
    """Bound metadata files so long training never creates oversized Git blobs."""
    output = Path(output)
    chunks = []
    for start in range(0, len(history), 100):
        name = f"curve/steps-{start + 1:06d}-{min(start + 100, len(history)):06d}.json"
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, history[start : start + 100])
        chunks.append(name)
    write_json(
        output / "learning-curve.json",
        {"format": "chunked-step-records-v1", "steps": len(history), "chunks": chunks},
    )
