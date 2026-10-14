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
from .accounting import Counter
from contextlib import nullcontext
from .performance import settings, VERSION
from .engine import step as optimization_step, probes
from .checkpoint import save, restore
from .assets import load_bundle
from .evaluation import evaluate_windows
from .analysis import baseline_review


def training_budget_sha(c):
    return canonical_sha256({k: v for k, v in c["training"].items() if k not in ("mechanism", "microbatch")})


def identity(c, p, arm, seed, role, source, assets):
    return {
        "experiment_id": "006-PD1",
        "implementation_version": VERSION,
        "performance": settings(c),
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


def optimize(model, x, y, opt, c, step, generators, capture=False):
    mechanism = c["arms"][model.arm]["mechanism"]
    outputs = sample_paths(model, x, generators, mechanism, capture=capture)
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
    perf = settings(c)
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
    prior_flops = None
    prior_elapsed = 0.0
    prior_training_gemm = None
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
        instrumented = perf["training_profile"] == "instrumented"
        with Counter(model) if instrumented else nullcontext() as counter:
            for step in range(first + 1, t["steps"] + 1):
                diagnostic = instrumented or step % perf["diagnostics_interval"] == 0
                chosen = rng.choice(assets["train_starts"], size=t["effective_batch"], replace=True)
                log = optimization_step(
                    model,
                    opt,
                    c,
                    step,
                    chosen,
                    assets["train"],
                    generators,
                    device,
                    capture=diagnostic,
                    diagnostics=diagnostic,
                    dispatch=instrumented,
                    routing=diagnostic,
                )
                log["forward_gemm_flops"] = counter.total + (prior_flops or 0) if counter else None
                if instrumented:
                    log["observed_training_gemm_flops"] += prior_training_gemm or 0
                    prior_training_gemm = log["observed_training_gemm_flops"]
                log["diagnostics_interval"] = perf["diagnostics_interval"]
                log["train_probe_interval"] = perf["train_probe_interval"]
                log.update(
                    train_probe_greedy_nll=None,
                    train_probe_sampled_nll=None,
                    sampled_greedy_nll_gap=None,
                    route_drift_from_initial=None,
                    probe_sampled=False,
                )
                if instrumented or step % perf["train_probe_interval"] == 0:
                    if counter:
                        counter.enabled = False
                    log.update(probes(model, probe_x, probe_y, seed, step, initial_actions))
                    if counter:
                        counter.enabled = True
                # Preserve the ORIGINAL optimizer-step stability window. Sampling
                # probes must not silently redefine the scientific threshold.
                stability = p["thresholds"].get("baseline_stability")
                required_baseline = (
                    arm == "M0" and stability and (step == 1 or step > t["steps"] - stability["tail_steps"])
                )
                if required_baseline and log["train_probe_greedy_nll"] is None:
                    from .losses import terminal_ce

                    model.eval()
                    if counter:
                        counter.enabled = False
                    with torch.no_grad():
                        log["train_probe_greedy_nll"] = float(terminal_ce(model(probe_x), probe_y).mean())
                    log["probe_sampled"] = True
                    log["probe_scope"] = "required original baseline stability window; greedy only"
                    if counter:
                        counter.enabled = True
                history.append(log)
                if step == 1 or step % perf["console_interval"] == 0:
                    print(
                        json.dumps(
                            {
                                k: log[k]
                                for k in ("step", "ce", "actor_loss", "critic_loss", "forward_gemm_flops")
                            }
                        ),
                        flush=True,
                    )
                pause = step < t["steps"] and (
                    (perf["pause_after_step"] is not None and first < perf["pause_after_step"] <= step)
                    or (
                        perf["max_session_seconds"] is not None
                        and time.perf_counter() - started >= perf["max_session_seconds"]
                    )
                )
                # Existing checkpoint_every remains the scientific default; the explicit
                # performance interval may request a more frequent atomic save.
                interval = min(t["checkpoint_every"], perf["checkpoint_interval"])
                if step % interval == 0 or step == t["steps"] or pause:
                    save_started = time.perf_counter()
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
                            "forward_gemm": (prior_flops or 0) + counter.total if counter else None,
                            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
                            "observed_training_gemm": prior_training_gemm,
                            "initial_probe_actions": initial_actions,
                        },
                    )
                    write_json(
                        output / "checkpoint-io.json",
                        {
                            "step": step,
                            "save_seconds": time.perf_counter() - save_started,
                            "checkpoint_bytes": latest.stat().st_size,
                            "scope": "atomic model/optimizer/RNG checkpoint write; excludes curves, hashing and portable export",
                        },
                    )
                    write_curve(output, history)
                if pause:
                    result = {
                        "identity": run_id,
                        "status": "PAUSED_RESUMABLE",
                        "step": step,
                        "checkpoint_sha": file_sha256(latest),
                        "scientific_decision": False,
                        "resume": "Use the identical source/config/assets and --resume; import a verified Kaggle Output bundle in a new session",
                    }
                    write_json(output / "pause.json", result)
                    return result
            if step != t["steps"]:
                raise RuntimeError("final step missing")
            checkpoint_sha = file_sha256(latest)
            frozen = {"sha": checkpoint_sha, "step": step, "complete": True}
            # Final-only loader happens after immutable checkpoint hash is fixed.
            if counter:
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
                "cost": counter.report(step * t["effective_batch"] * t["sequence_length"])
                if counter
                else {
                    "total_flops_status": "OFFLINE_ONLY",
                    "forward_gemm_flops": None,
                    "training_backward_flops": None,
                    "executed_parameter_work": None,
                    "measurement_scope": "not measured in training; use separate frozen-checkpoint profile",
                    "resident_parameters": sum(v.numel() for v in model.parameters()),
                },
                "routing": greedy["routing"],
                "stability": history[-1],
                "interventions": [],
                "statistics": {},
                "device": str(device),
                "runtime": {"torch": torch.__version__, "precision": "float32"},
                "authorization": {
                    "registered": p.get("status") == "REGISTERED",
                    "automatic_development": p.get("status") == "AUTOMATIC_DEVELOPMENT",
                    "engineering_scope": c.get("engineering_scope"),
                    "role": role,
                },
                "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
                "gpu_hours": (time.perf_counter() - started) / 3600
                if str(device).startswith("cuda")
                else 0.0,
            }
            write_json(output / "metrics.json", result)
            (output / "pause.json").unlink(missing_ok=True)
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
