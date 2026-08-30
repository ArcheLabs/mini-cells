from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_plastic_metrics import paired_bootstrap_delta, task_signal_null  # noqa: E402
from minicells.language_plastic_tissue import (  # noqa: E402
    ACTIVITY_BUDGET,
    PLASTICITY_RATE,
    STABILITY_WEIGHT,
    SYNAPTIC_BUDGET,
    TISSUE_HEIGHT,
    build_plastic_reaction_diffusion_model,
)
from minicells.language_skill_data import (  # noqa: E402
    ALL_TASKS,
    MODEL_LENGTH,
    VOCAB_SIZE,
    SkillCorpus,
    batch_from_indices,
    make_index_schedule,
)
from minicells.language_sparse_topology import BALANCE_WEIGHT, build_sparse_topology_model  # noqa: E402
from minicells.language_stabilization import make_depth_schedule  # noqa: E402

VARIANTS = ("B", "D")
BATCH_SIZE = 64
TRAIN_STEPS = 1000
TOKENS_PER_STEP = BATCH_SIZE * MODEL_LENGTH
BUDGET_TOKENS = TRAIN_STEPS * TOKENS_PER_STEP
CHECKPOINT_STEPS = (250, 500, 1000)
BASE_LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 50
N_REPLICATES = 3
MODEL_SEED_BASE = 71_015
SCHEDULE_SEED_BASE = 31_015
DEPTH_SEED_BASE = 41_015
NULL_SEED_BASE = 61_015
BOOTSTRAP_SEED_BASE = 81_015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 015b model.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--replicate", type=int, choices=range(N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_corpus(cache_dir: Path, prefix: str) -> SkillCorpus:
    payload = torch.load(cache_dir / f"{prefix}-corpus.pt", map_location="cpu")
    return SkillCorpus(
        sequences=payload["sequences"],
        task_ids=payload["task_ids"],
        task_names=tuple(payload["task_names"]),
        loss_mask=payload["loss_mask"],
    )


def lr_multiplier(step: int, total_steps: int) -> float:
    if step <= WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, mask, :].reshape(-1, logits.shape[-1]), targets[:, mask].reshape(-1))


def build_matched_model(variant: str, seed: int):
    torch.manual_seed(seed)
    baseline = build_sparse_topology_model(VOCAB_SIZE, variant="B")
    if variant == "B":
        return baseline, {"copied_shared_tensors": 0}
    torch.manual_seed(seed)
    model = build_plastic_reaction_diffusion_model(VOCAB_SIZE)
    baseline_state = baseline.state_dict()
    target_state = model.state_dict()
    copied = 0
    with torch.no_grad():
        for key, value in target_state.items():
            source = baseline_state.get(key)
            if source is not None and source.shape == value.shape:
                value.copy_(source)
                copied += 1
    model.load_state_dict(target_state)
    return model, {"copied_shared_tensors": copied}


def training_forward(model, variant: str, inputs: torch.Tensor, depths: tuple[int, int, int]):
    if variant == "B":
        result = model.forward_variable(inputs, stage_depths=depths)
        return result.output.logits, result.stability_loss, BALANCE_WEIGHT * result.balance_loss
    result = model.forward_variable(inputs, stage_depths=depths)
    return result.output.logits, result.stability_loss, result.output.logits.new_zeros(())


@torch.no_grad()
def evaluate_quality(
    model,
    corpus: SkillCorpus,
    *,
    variant: str,
    device: torch.device,
    depths: tuple[int, int, int] = (4, 4, 4),
    intervention: str = "normal",
    ablate_row: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    model.eval()
    mask = corpus.loss_mask.to(device)
    digit_positions = torch.nonzero(mask, as_tuple=False).flatten()[:-1]
    n_tasks = len(corpus.task_names)
    loss_sum = np.zeros(n_tasks)
    loss_tokens = np.zeros(n_tasks, dtype=np.int64)
    correct = np.zeros(n_tasks, dtype=np.int64)
    total = np.zeros(n_tasks, dtype=np.int64)
    exact = np.zeros(n_tasks, dtype=np.int64)
    per_example = np.zeros(len(corpus.sequences))

    for start in range(0, len(corpus.sequences), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(corpus.sequences))
        inputs, targets, task_ids = batch_from_indices(
            corpus, torch.arange(start, stop), device=device
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            if variant == "B":
                result = model.forward_variable(inputs, stage_depths=depths, ablate_row=ablate_row)
            else:
                result = model.forward_variable(
                    inputs,
                    stage_depths=depths,
                    intervention=intervention,
                    ablate_row=ablate_row,
                )
        logits = result.output.logits.float()
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view(targets.shape)
        predictions = logits.argmax(dim=-1)
        selected = losses[:, mask]
        per_example[start:stop] = selected.mean(dim=1).cpu().numpy()
        for local_index, task_id in enumerate(task_ids.cpu().tolist()):
            values = selected[local_index]
            loss_sum[task_id] += float(values.sum())
            loss_tokens[task_id] += int(values.numel())
            expected = targets[local_index, digit_positions]
            predicted = predictions[local_index, digit_positions]
            correct[task_id] += int((predicted == expected).sum())
            total[task_id] += int(expected.numel())
            exact[task_id] += int(bool(torch.equal(predicted, expected)))

    rows = []
    for task_id, task in enumerate(corpus.task_names):
        nll = loss_sum[task_id] / max(1, loss_tokens[task_id])
        examples = int((corpus.task_ids == task_id).sum())
        rows.append(
            {
                "task_id": task_id,
                "task": task,
                "nll": nll,
                "ppl": math.exp(min(nll, 20.0)),
                "token_accuracy": correct[task_id] / max(1, total[task_id]),
                "exact_match": exact[task_id] / max(1, examples),
                "examples": examples,
            }
        )
    return pd.DataFrame(rows), per_example


@torch.no_grad()
def evaluate_observability(model, corpus: SkillCorpus, *, device: torch.device, null_seed: int):
    model.eval()
    mask = corpus.loss_mask.to(device)
    activities, connectomes, tasks = [], [], []
    activity_trace_sum = connectome_trace_sum = scalar_trace_sum = None
    seen = 0
    for start in range(0, len(corpus.sequences), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(corpus.sequences))
        inputs, _, task_ids = batch_from_indices(corpus, torch.arange(start, stop), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(inputs, stage_depths=(4, 4, 4), collect_observability=True)
        d = result.diagnostics
        assert d is not None
        activity = d.activity[:, mask, :].float().mean(dim=1)
        connectome = d.connectome[:, mask, :, :].float().mean(dim=1)
        activities.append(activity.cpu().numpy())
        connectomes.append(connectome.cpu().numpy())
        tasks.append(task_ids.cpu().numpy())
        trace_a = d.activity_trace[:, :, mask, :].float().mean(dim=(1, 2)).cpu().numpy()
        trace_w = d.connectome_trace[:, :, mask, :, :].float().mean(dim=(1, 2)).cpu().numpy()
        scalar = np.stack(
            [
                d.reaction_rms_trace.float().cpu().numpy(),
                d.diffusion_rms_trace.float().cpu().numpy(),
                d.residual_trace.float().cpu().numpy(),
                d.activity_entropy_trace.float().cpu().numpy(),
                d.connectome_entropy_trace.float().cpu().numpy(),
                d.nonlocal_mass_trace.float().cpu().numpy(),
            ], axis=1
        )
        n = stop - start
        if activity_trace_sum is None:
            activity_trace_sum, connectome_trace_sum, scalar_trace_sum = trace_a * n, trace_w * n, scalar * n
        else:
            activity_trace_sum += trace_a * n
            connectome_trace_sum += trace_w * n
            scalar_trace_sum += scalar * n
        seen += n

    activity = np.concatenate(activities)
    connectome = np.concatenate(connectomes)
    task_ids = np.concatenate(tasks)
    activity_trace = activity_trace_sum / seen
    connectome_trace = connectome_trace_sum / seen
    scalar_trace = scalar_trace_sum / seen
    activity_signal = task_signal_null(task_ids, activity, seed=null_seed)
    connectome_signal = task_signal_null(task_ids, connectome.reshape(len(connectome), -1), seed=null_seed + 1)

    activity_rows, connectome_rows = [], []
    for task_id, task in enumerate(corpus.task_names):
        selected = task_ids == task_id
        mean_a = activity[selected].mean(axis=0)
        mean_w = connectome[selected].mean(axis=0)
        for row, value in enumerate(mean_a):
            activity_rows.append({"task": task, "row": row, "mean_activity": float(value)})
        for receiver in range(TISSUE_HEIGHT):
            for source in range(TISSUE_HEIGHT):
                connectome_rows.append(
                    {"task": task, "receiver": receiver, "source": source, "mean_weight": float(mean_w[receiver, source])}
                )

    final_a = activity.mean(axis=0)
    final_w = connectome.mean(axis=0)
    final_activity = pd.DataFrame([{"row": r, "mean_activity": float(v)} for r, v in enumerate(final_a)])
    final_connectome = pd.DataFrame(
        [{"receiver": r, "source": s, "mean_weight": float(final_w[r, s])}
         for r in range(TISSUE_HEIGHT) for s in range(TISSUE_HEIGHT)]
    )
    dynamics_rows = []
    for step in range(activity_trace.shape[0]):
        row = {
            "evolution_step": step + 1,
            "reaction_rms": float(scalar_trace[step, 0]),
            "diffusion_rms": float(scalar_trace[step, 1]),
            "diffusion_to_reaction": float(scalar_trace[step, 1] / max(1e-8, scalar_trace[step, 0])),
            "state_residual": float(scalar_trace[step, 2]),
            "activity_entropy": float(scalar_trace[step, 3]),
            "connectome_entropy": float(scalar_trace[step, 4]),
            "nonlocal_mass": float(scalar_trace[step, 5]),
        }
        for r in range(TISSUE_HEIGHT):
            row[f"activity_r{r}"] = float(activity_trace[step, r])
        dynamics_rows.append(row)
    dynamics = pd.DataFrame(dynamics_rows)
    summary = {
        "task_activity_mi": activity_signal,
        "task_connectome_mi": connectome_signal,
        "effective_active_fraction": float(np.mean(activity.sum(-1) ** 2 / np.clip((activity ** 2).sum(-1), 1e-12, None) / TISSUE_HEIGHT)),
        "final_activity_entropy": float(dynamics.iloc[-1]["activity_entropy"]),
        "final_connectome_entropy": float(dynamics.iloc[-1]["connectome_entropy"]),
        "final_nonlocal_mass": float(dynamics.iloc[-1]["nonlocal_mass"]),
        "mean_diffusion_to_reaction": float(dynamics["diffusion_to_reaction"].mean()),
        "connectome_change_l1": float(np.abs(connectome_trace[-1] - connectome_trace[0]).mean()),
    }
    return pd.DataFrame(activity_rows), pd.DataFrame(connectome_rows), final_activity, final_connectome, dynamics, summary


def ablation_analysis(model, corpus: SkillCorpus, baseline: pd.DataFrame, *, device: torch.device, run_name: str):
    base = baseline.set_index("task")
    rows, specific = [], []
    for row_id in range(1, TISSUE_HEIGHT):
        ablated, _ = evaluate_quality(model, corpus, variant="D", device=device, ablate_row=row_id)
        by_task = ablated.set_index("task")
        deltas = []
        for task in ALL_TASKS:
            delta = float(by_task.loc[task, "nll"] - base.loc[task, "nll"])
            deltas.append((task, delta))
            rows.append({"run": run_name, "row": row_id, "task": task, "baseline_nll": float(base.loc[task, "nll"]), "ablated_nll": float(by_task.loc[task, "nll"]), "delta_nll": delta})
        best_task, best_delta = max(deltas, key=lambda item: item[1])
        others = [abs(v) for task, v in deltas if task != best_task]
        specificity = best_delta / max(1e-6, float(np.mean(others)))
        if best_delta >= 0.02 and specificity >= 1.5:
            specific.append({"row": row_id, "task": best_task, "delta_nll": best_delta, "specificity": specificity})
    return pd.DataFrame(rows), {"functional_specific_rows": len(specific), "specific_rows": specific, "delta_nll_threshold": 0.02, "specificity_threshold": 1.5}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 015b worker requires CUDA")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_corpus(args.cache_dir.resolve(), "train")
    validation = load_corpus(args.cache_dir.resolve(), "validation")
    replicate = args.replicate
    model_seed = MODEL_SEED_BASE + 1000 * replicate
    schedule_seed = SCHEDULE_SEED_BASE + 1000 * replicate
    depth_seed = DEPTH_SEED_BASE + 1000 * replicate
    null_seed = NULL_SEED_BASE + 1000 * replicate
    bootstrap_seed = BOOTSTRAP_SEED_BASE + 1000 * replicate
    run = f"r{replicate}-{args.variant}"
    model, initialization = build_matched_model(args.variant, model_seed)
    parameters = count_parameters(model)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()
    schedule = make_index_schedule(len(train.sequences), steps=TRAIN_STEPS, batch_size=BATCH_SIZE, seed=schedule_seed)
    depth_schedule = make_depth_schedule(TRAIN_STEPS, seed=depth_seed)
    mask = train.loss_mask.to(device)
    checkpoints, elapsed, iterations = [], 0.0, 0
    print({"run": run, "variant": args.variant, "gpu": torch.cuda.get_device_name(0), "parameters": parameters, **initialization})

    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        lr = BASE_LR * lr_multiplier(step, TRAIN_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets, _ = batch_from_indices(train, schedule[step - 1], device=device)
        optimizer.zero_grad(set_to_none=True)
        depths = tuple(int(v) for v in depth_schedule[step - 1])
        synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits, stability, auxiliary = training_forward(model, args.variant, inputs, depths)
            main_loss = masked_loss(logits, targets, mask)
            loss = main_loss + STABILITY_WEIGHT * stability + auxiliary
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        elapsed += time.perf_counter() - started
        iterations += sum(depths)
        if step in CHECKPOINT_STEPS:
            metrics, _ = evaluate_quality(model, validation, variant=args.variant, device=device)
            consumed = step * TOKENS_PER_STEP
            checkpoints.append({"run": run, "replicate": replicate, "variant": args.variant, "step": step, "consumed_tokens": consumed, "train_loss": float(loss.detach()), "main_loss": float(main_loss.detach()), "stability_loss": float(stability.detach()), "auxiliary_loss": float(auxiliary.detach()), "grad_norm": grad_norm, "learning_rate": lr, "validation_mean_nll": float(metrics["nll"].mean()), "validation_mean_token_accuracy": float(metrics["token_accuracy"].mean()), "training_elapsed_seconds": elapsed, "seconds_per_million_tokens": elapsed / (consumed / 1_000_000), "avg_recurrent_iterations": iterations / step})
    pd.DataFrame(checkpoints).to_csv(output_dir / f"{run}-checkpoints.csv", index=False)

    task_metrics, normal_nll = evaluate_quality(model, validation, variant=args.variant, device=device)
    task_metrics.insert(0, "run", run)
    task_metrics.insert(1, "replicate", replicate)
    task_metrics.insert(2, "variant", args.variant)
    task_metrics.to_csv(output_dir / f"{run}-task-metrics.csv", index=False)
    depth_rows = []
    for depth in (2, 3, 4):
        metrics, _ = evaluate_quality(model, validation, variant=args.variant, device=device, depths=(depth, depth, depth))
        depth_rows.append({"run": run, "replicate": replicate, "variant": args.variant, "depth": depth, "mean_nll": float(metrics["nll"].mean()), "mean_token_accuracy": float(metrics["token_accuracy"].mean())})
    pd.DataFrame(depth_rows).to_csv(output_dir / f"{run}-depth-eval.csv", index=False)

    topology = {}
    interventions = {}
    ablation_summary = {}
    if args.variant == "D":
        activity, connectome, final_activity, final_connectome, dynamics, topology = evaluate_observability(model, validation, device=device, null_seed=null_seed)
        for frame in (activity, connectome, final_activity, final_connectome, dynamics):
            frame.insert(0, "run", run)
        activity.to_csv(output_dir / f"{run}-activity.csv", index=False)
        connectome.to_csv(output_dir / f"{run}-connectome.csv", index=False)
        final_activity.to_csv(output_dir / f"{run}-final-activity.csv", index=False)
        final_connectome.to_csv(output_dir / f"{run}-final-connectome.csv", index=False)
        dynamics.to_csv(output_dir / f"{run}-dynamics.csv", index=False)
        intervention_rows = []
        for offset, intervention in enumerate(("diffusion_off", "plasticity_off", "connectome_shuffled")):
            _, altered = evaluate_quality(model, validation, variant="D", device=device, intervention=intervention)
            stats = paired_bootstrap_delta(normal_nll, altered, seed=bootstrap_seed + offset)
            interventions[intervention] = stats
            intervention_rows.append({"run": run, "intervention": intervention, **stats})
        pd.DataFrame(intervention_rows).to_csv(output_dir / f"{run}-interventions.csv", index=False)
        ablation, ablation_summary = ablation_analysis(model, validation, task_metrics, device=device, run_name=run)
        ablation.to_csv(output_dir / f"{run}-ablation.csv", index=False)

    peak = int(torch.cuda.max_memory_allocated())
    worker = {
        "format": "minicells.language-plastic-reaction-diffusion-worker.v1",
        "run": run,
        "replicate": replicate,
        "variant": args.variant,
        "parameters": parameters,
        "initialization": initialization,
        "model_seed": model_seed,
        "schedule_seed": schedule_seed,
        "depth_seed": depth_seed,
        "null_seed": null_seed,
        "bootstrap_seed": bootstrap_seed,
        "tokens": BUDGET_TOKENS,
        "training_elapsed_seconds": elapsed,
        "training_tokens_per_second": BUDGET_TOKENS / elapsed,
        "seconds_per_million_tokens": elapsed / (BUDGET_TOKENS / 1_000_000),
        "peak_vram_bytes": peak,
        "avg_recurrent_iterations": iterations / TRAIN_STEPS,
        "mean_nll": float(task_metrics["nll"].mean()),
        "mean_token_accuracy": float(task_metrics["token_accuracy"].mean()),
        "mean_exact_match": float(task_metrics["exact_match"].mean()),
        "activity_budget": ACTIVITY_BUDGET if args.variant == "D" else None,
        "synaptic_budget": SYNAPTIC_BUDGET if args.variant == "D" else None,
        "plasticity_rate": PLASTICITY_RATE if args.variant == "D" else None,
        "topology": topology,
        "interventions": interventions,
        "ablation": ablation_summary,
    }
    (output_dir / f"{run}-worker.json").write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
