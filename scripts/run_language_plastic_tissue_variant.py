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
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

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
from minicells.language_sparse_topology import (  # noqa: E402
    BALANCE_WEIGHT,
    build_sparse_topology_model,
)
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


def _lr_multiplier(step: int, total_steps: int) -> float:
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


def masked_language_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    selected_logits = logits[:, loss_mask, :].reshape(-1, logits.shape[-1])
    selected_targets = targets[:, loss_mask].reshape(-1)
    return F.cross_entropy(selected_logits, selected_targets)


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


def forward_training(model, variant: str, inputs: torch.Tensor, depths: tuple[int, int, int]):
    if variant == "B":
        result = model.forward_variable(inputs, stage_depths=depths)
        return (
            result.output.logits,
            result.stability_loss,
            BALANCE_WEIGHT * result.balance_loss,
        )
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
    task_loss_sum = np.zeros(len(corpus.task_names), dtype=np.float64)
    task_loss_tokens = np.zeros(len(corpus.task_names), dtype=np.int64)
    task_correct = np.zeros(len(corpus.task_names), dtype=np.int64)
    task_total = np.zeros(len(corpus.task_names), dtype=np.int64)
    task_exact = np.zeros(len(corpus.task_names), dtype=np.int64)
    per_example = np.zeros(len(corpus.sequences), dtype=np.float64)

    for start in range(0, len(corpus.sequences), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(corpus.sequences))
        indices = torch.arange(start, stop)
        inputs, targets, task_ids = batch_from_indices(corpus, indices, device=device)
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
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        predictions = logits.argmax(dim=-1)
        selected_losses = losses[:, mask]
        per_example[start:stop] = selected_losses.mean(dim=1).detach().cpu().numpy()
        for local_index, task_id in enumerate(task_ids.detach().cpu().tolist()):
            task_losses = selected_losses[local_index]
            task_loss_sum[task_id] += float(task_losses.sum().item())
            task_loss_tokens[task_id] += int(task_losses.numel())
            expected_digits = targets[local_index, digit_positions]
            predicted_digits = predictions[local_index, digit_positions]
            task_correct[task_id] += int((predicted_digits == expected_digits).sum().item())
            task_total[task_id] += int(expected_digits.numel())
            task_exact[task_id] += int(bool(torch.equal(predicted_digits, expected_digits)))

    rows = []
    for task_id, task in enumerate(corpus.task_names):
        nll = task_loss_sum[task_id] / max(1, task_loss_tokens[task_id])
        examples = int((corpus.task_ids == task_id).sum().item())
        rows.append(
            {
                "task_id": task_id,
                "task": task,
                "nll": nll,
                "ppl": math.exp(min(nll, 20.0)),
                "token_accuracy": task_correct[task_id] / max(1, task_total[task_id]),
                "exact_match": task_exact[task_id] / max(1, examples),
                "examples": examples,
            }
        )
    return pd.DataFrame(rows), per_example


@torch.no_grad()
def evaluate_observability(
    model,
    corpus: SkillCorpus,
    *,
    device: torch.device,
    null_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    model.eval()
    mask = corpus.loss_mask.to(device)
    activities = []
    connectomes = []
    tasks = []
    activity_trace_sum = None
    connectome_trace_sum = None
    scalar_trace_sum = None
    examples_seen = 0

    for start in range(0, len(corpus.sequences), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(corpus.sequences))
        indices = torch.arange(start, stop)
        inputs, _, task_ids = batch_from_indices(corpus, indices, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(
                inputs,
                stage_depths=(4, 4, 4),
                collect_observability=True,
            )
        diagnostics = result.diagnostics
        assert diagnostics is not None
        activity = diagnostics.activity[:, mask, :].float().mean(dim=1)
        connectome = diagnostics.connectome[:, mask, :, :].float().mean(dim=1)
        activities.append(activity.cpu().numpy())
        connectomes.append(connectome.cpu().numpy())
        tasks.append(task_ids.cpu().numpy())

        trace_activity = diagnostics.activity_trace[:, :, mask, :].float().mean(dim=(1, 2)).cpu().numpy()
        trace_connectome = diagnostics.connectome_trace[:, :, mask, :, :].float().mean(dim=(1, 2)).cpu().numpy()
        scalar = np.stack(
            [
                diagnostics.reaction_rms_trace.float().cpu().numpy(),
                diagnostics.diffusion_rms_trace.float().cpu().numpy(),
                diagnostics.residual_trace.float().cpu().numpy(),
                diagnostics.activity_entropy_trace.float().cpu().numpy(),
                diagnostics.connectome_entropy_trace.float().cpu().numpy(),
                diagnostics.nonlocal_mass_trace.float().cpu().numpy(),
            ],
            axis=1,
        )
        batch_n = stop - start
        if activity_trace_sum is None:
            activity_trace_sum = trace_activity * batch_n
            connectome_trace_sum = trace_connectome * batch_n
            scalar_trace_sum = scalar * batch_n
        else:
            activity_trace_sum += trace_activity * batch_n
            connectome_trace_sum += trace_connectome * batch_n
            scalar_trace_sum += scalar * batch_n
        examples_seen += batch_n

    activity = np.concatenate(activities, axis=0)
    connectome = np.concatenate(connectomes, axis=0)
    task_ids = np.concatenate(tasks, axis=0)
    assert activity_trace_sum is not None and connectome_trace_sum is not None and scalar_trace_sum is not None
    activity_trace = activity_trace_sum / examples_seen
    connectome_trace = connectome_trace_sum / examples_seen
    scalar_trace = scalar_trace_sum / examples_seen

    region_signal = task_signal_null(task_ids, activity, seed=null_seed)
    connectome_signal = task_signal_null(
        task_ids,
        connectome.reshape(len(connectome), -1),
        seed=null_seed + 1,
    )

    activity_rows = []
    connectome_rows = []
    for task_id, task in enumerate(corpus.task_names):
        selected = task_ids == task_id
        mean_activity = activity[selected].mean(axis=0)
        mean_connectome = connectome[selected].mean(axis=0)
        for row, value in enumerate(mean_activity):
            activity_rows.append(
                {"task": task, "row": row, "mean_activity": float(value)}
            )
        for receiver in range(TISSUE_HEIGHT):
            for source in range(TISSUE_HEIGHT):
                connectome_rows.append(
                    {
                        "task": task,
                        "receiver": receiver,
                        "source": source,
                        "mean_weight": float(mean_connectome[receiver, source]),
                    }
                )

    final_activity = activity.mean(axis=0)
    final_connectome = connectome.mean(axis=0)
    final_activity_frame = pd.DataFrame(
        [{"row": row, "mean_activity": float(value)} for row, value in enumerate(final_activity)]
    )
    final_connectome_frame = pd.DataFrame(
        [
            {
                "receiver": receiver,
                "source": source,
                "mean_weight": float(final_connectome[receiver, source]),
            }
            for receiver in range(TISSUE_HEIGHT)
            for source in range(TISSUE_HEIGHT)
        ]
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
        for tissue_row in range(TISSUE_HEIGHT):
            row[f"activity_r{tissue_row}"] = float(activity_trace[step, tissue_row])
        dynamics_rows.append(row)
    dynamics = pd.DataFrame(dynamics_rows)

    summary = {
        "task_activity_mi": region_signal,
        "task_connectome_mi": connectome_signal,
        "effective_active_fraction": float(
            np.mean(
                np.square(activity.sum(axis=-1))
                / np.clip(np.square(activity).sum(axis=-1), 1e-12, None)
                / TISSUE_HEIGHT
            )
        ),
        "final_activity_entropy": float(dynamics.iloc[-1]["activity_entropy"]),
        "final_connectome_entropy": float(dynamics.iloc[-1]["connectome_entropy"]),
        "final_nonlocal_mass": float(dynamics.iloc[-1]["nonlocal_mass"]),
        "mean_diffusion_to_reaction": float(dynamics["diffusion_to_reaction"].mean()),
        "connectome_change_l1": float(np.abs(connectome_trace[-1] - connectome_trace[0]).mean()),
    }
    return (
        pd.DataFrame(activity_rows),
        pd.DataFrame(connectome_rows),
        final_activity_frame,
        final_connectome_frame,
        dynamics,
        summary,
    )


def ablation_analysis(
    model,
    corpus: SkillCorpus,
    baseline: pd.DataFrame,
    *,
    device: torch.device,
    run_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = baseline.set_index("task")
    rows = []
    specific = []
    for tissue_row in range(1, TISSUE_HEIGHT):
        ablated, _ = evaluate_quality(
            model,
            corpus,
            variant="D",
            device=device,
            ablate_row=tissue_row,
        )
        by_task = ablated.set_index("task")
        deltas = []
        for task in ALL_TASKS:
            delta = float(by_task.loc[task, "nll"] - base.loc[task, "nll"])
            deltas.append((task, delta))
            rows.append(
                {
                    "run": run_name,
                    "row": tissue_row,
                    "task": task,
                    "baseline_nll": float(base.loc[task, "nll"]),
                    "ablated_nll": float(by_task.loc[task, "nll"]),
                    "delta_nll": delta,
                }
            )
        best_task, best_delta = max(deltas, key=lambda item: item[1])
        others = [abs(value) for task, value in deltas if task != best_task]
        specificity = best_delta / max(1e-6, float(np.mean(others)))
        if best_delta >= 0.02 and specificity >= 1.5:
            specific.append(
                {
                    "row": tissue_row,
                    "task": best_task,
                    "delta_nll": best_delta,
                    "specificity": specificity,
                }
            )
    return pd.DataFrame(rows), {
        "functional_specific_rows": len(specific),
        "specific_rows": specific,
        "delta_nll_threshold": 0.02,
        "specificity_threshold": 1.5,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 015b worker requires CUDA")
    device = torch.device("cuda:0")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_corpus = load_corpus(cache_dir, "train")
    validation_corpus = load_corpus(cache_dir, "validation")
    replicate = args.replicate
    model_seed = MODEL_SEED_BASE + 1000 * replicate
    schedule_seed = SCHEDULE_SEED_BASE + 1000 * replicate
    depth_seed = DEPTH_SEED_BASE + 1000 * replicate
    null_seed = NULL_SEED_BASE + 1000 * replicate
    bootstrap_seed = BOOTSTRAP_SEED_BASE + 1000 * replicate
    run_name = f"r{replicate}-{args.variant}"

    model, initialization = build_matched_model(args.variant, model_seed)
    parameters = count_parameters(model)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )
    scaler = make_scaler()
    index_schedule = make_index_schedule(
        len(train_corpus.sequences),
        steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        seed=schedule_seed,
    )
    depth_schedule = make_depth_schedule(TRAIN_STEPS, seed=depth_seed)
    checkpoint_rows = []
    training_elapsed = 0.0
    executed_iterations = 0
    mask = train_corpus.loss_mask.to(device)

    print(
        {
            "run": run_name,
            "variant": args.variant,
            "gpu": torch.cuda.get_device_name(0),
            "parameters": parameters,
            "model_seed": model_seed,
            **initialization,
        }
    )

    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        lr = BASE_LR * _lr_multiplier(step, TRAIN_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets, _ = batch_from_indices(
            train_corpus,
            index_schedule[step - 1],
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        depths = tuple(int(value) for value in depth_schedule[step - 1])
        synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits, stability_loss, auxiliary_loss = forward_training(
                model,
                args.variant,
                inputs,
                depths,
            )
            main_loss = masked_language_loss(logits, targets, mask)
            loss = main_loss + STABILITY_WEIGHT * stability_loss + auxiliary_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        training_elapsed += time.perf_counter() - started
        executed_iterations += sum(depths)

        if step in CHECKPOINT_STEPS:
            metrics, _ = evaluate_quality(
                model,
                validation_corpus,
                variant=args.variant,
                device=device,
            )
            consumed = step * TOKENS_PER_STEP
            row = {
                "run": run_name,
                "replicate": replicate,
                "variant": args.variant,
                "step": step,
                "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()),
                "main_loss": float(main_loss.detach().item()),
                "stability_loss": float(stability_loss.detach().item()),
                "auxiliary_loss": float(auxiliary_loss.detach().item()),
                "grad_norm": grad_norm,
                "learning_rate": lr,
                "validation_mean_nll": float(metrics["nll"].mean()),
                "validation_mean_token_accuracy": float(metrics["token_accuracy"].mean()),
                "training_elapsed_seconds": training_elapsed,
                "seconds_per_million_tokens": training_elapsed / (consumed / 1_000_000),
                "avg_recurrent_iterations": executed_iterations / step,
            }
            checkpoint_rows.append(row)
            print(
                f"{run_name:5s} step={step:4d} nll={row['validation_mean_nll']:.4f} "
                f"acc={row['validation_mean_token_accuracy']:.3f}"
            )

    pd.DataFrame(checkpoint_rows).to_csv(
        output_dir / f"{run_name}-checkpoints.csv",
        index=False,
    )
    task_metrics, normal_nll = evaluate_quality(
        model,
        validation_corpus,
        variant=args.variant,
        device=device,
    )
    task_metrics.insert(0, "run", run_name)
    task_metrics.insert(1, "replicate", replicate)
    task_metrics.insert(2, "variant", args.variant)
    task_metrics.to_csv(output_dir / f"{run_name}-task-metrics.csv", index=False)

    depth_rows = []
    for depth in (2, 3, 4):
        metrics, _ = evaluate_quality(
            model,
            validation_corpus,
            variant=args.variant,
            device=device,
            depths=(depth, depth, depth),
        )
        depth_rows.append(
            {
                "run": run_name,
                "replicate": replicate,
                "variant": args.variant,
                "depth": depth,
                "mean_nll": float(metrics["nll"].mean()),
                "mean_token_accuracy": float(metrics["token_accuracy"].mean()),
            }
        )
    pd.DataFrame(depth_rows).to_csv(output_dir / f"{run_name}-depth-eval.csv", index=False)

    topology_summary: dict[str, object] = {}
    intervention_summary: dict[str, object] = {}
    ablation_summary: dict[str, object] = {}
    if args.variant == "D":
        activity, connectome, final_activity, final_connectome, dynamics, topology_summary = evaluate_observability(
            model,
            validation_corpus,
            device=device,
            null_seed=null_seed,
        )
        for frame in (activity, connectome, final_activity, final_connectome, dynamics):
            frame.insert(0, "run", run_name)
        activity.to_csv(output_dir / f"{run_name}-activity.csv", index=False)
        connectome.to_csv(output_dir / f"{run_name}-connectome.csv", index=False)
        final_activity.to_csv(output_dir / f"{run_name}-final-activity.csv", index=False)
        final_connectome.to_csv(output_dir / f"{run_name}-final-connectome.csv", index=False)
        dynamics.to_csv(output_dir / f"{run_name}-dynamics.csv", index=False)

        intervention_rows = []
        for offset, intervention in enumerate(("diffusion_off", "plasticity_off", "connectome_shuffled")):
            _, intervened_nll = evaluate_quality(
                model,
                validation_corpus,
                variant="D",
                device=device,
                intervention=intervention,
            )
            stats = paired_bootstrap_delta(
                normal_nll,
                intervened_nll,
                seed=bootstrap_seed + offset,
            )
            intervention_summary[intervention] = stats
            intervention_rows.append(
                {
                    "run": run_name,
                    "intervention": intervention,
                    **stats,
                }
            )
        pd.DataFrame(intervention_rows).to_csv(
            output_dir / f"{run_name}-interventions.csv",
            index=False,
        )

        ablation, ablation_summary = ablation_analysis(
            model,
            validation_corpus,
            task_metrics,
            device=device,
            run_name=run_name,
        )
        ablation.to_csv(output_dir / f"{run_name}-ablation.csv", index=False)

    peak_vram = int(torch.cuda.max_memory_allocated())
    worker = {
        "format": "minicells.language-plastic-reaction-diffusion-worker.v1",
        "run": run_name,
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
        "training_elapsed_seconds": training_elapsed,
        "training_tokens_per_second": BUDGET_TOKENS / training_elapsed,
        "seconds_per_million_tokens": training_elapsed / (BUDGET_TOKENS / 1_000_000),
        "peak_vram_bytes": peak_vram,
        "avg_recurrent_iterations": executed_iterations / TRAIN_STEPS,
        "mean_nll": float(task_metrics["nll"].mean()),
        "mean_token_accuracy": float(task_metrics["token_accuracy"].mean()),
        "mean_exact_match": float(task_metrics["exact_match"].mean()),
        "activity_budget": ACTIVITY_BUDGET if args.variant == "D" else None,
        "synaptic_budget": SYNAPTIC_BUDGET if args.variant == "D" else None,
        "plasticity_rate": PLASTICITY_RATE if args.variant == "D" else None,
        "topology": topology_summary,
        "interventions": intervention_summary,
        "ablation": ablation_summary,
    }
    (output_dir / f"{run_name}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
