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
from minicells.language_skill_data import (  # noqa: E402
    ALL_TASKS,
    BASE_TASKS,
    COMPOSITION_MAP,
    MODEL_LENGTH,
    VOCAB_SIZE,
    SkillCorpus,
    batch_from_indices,
    make_index_schedule,
)
from minicells.language_sparse_topology import (  # noqa: E402
    ACTIVE_LATENT,
    BALANCE_WEIGHT,
    STABILITY_WEIGHT,
    TISSUE_HEIGHT,
    VARIANT_CODES,
    build_sparse_topology_model,
)
from minicells.language_stabilization import make_depth_schedule  # noqa: E402
from minicells.language_topology_metrics import composition_reuse_scores, permutation_mi_null  # noqa: E402


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
NULL_SEED_BASE = 51_015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 015 topology cell.")
    parser.add_argument("--variant", choices=VARIANT_CODES, required=True)
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


def masked_language_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    selected_logits = logits[:, loss_mask, :].reshape(-1, logits.shape[-1])
    selected_targets = targets[:, loss_mask].reshape(-1)
    return F.cross_entropy(selected_logits, selected_targets)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    corpus: SkillCorpus,
    *,
    device: torch.device,
    batch_size: int = 64,
    collect_topology: bool = False,
    ablate_row: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float]]:
    model.eval()
    mask = corpus.loss_mask.to(device)
    digit_positions = torch.nonzero(mask, as_tuple=False).flatten()[:-1]
    task_loss_sum = np.zeros(len(corpus.task_names), dtype=np.float64)
    task_loss_tokens = np.zeros(len(corpus.task_names), dtype=np.int64)
    task_correct = np.zeros(len(corpus.task_names), dtype=np.int64)
    task_total = np.zeros(len(corpus.task_names), dtype=np.int64)
    task_exact = np.zeros(len(corpus.task_names), dtype=np.int64)
    activity_rows: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    task_rows: list[np.ndarray] = []
    active_fractions: list[float] = []

    for start in range(0, len(corpus.sequences), batch_size):
        stop = min(start + batch_size, len(corpus.sequences))
        indices = torch.arange(start, stop)
        inputs, targets, task_ids = batch_from_indices(corpus, indices, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(
                inputs,
                stage_depths=(4, 4, 4),
                collect_topology=collect_topology,
                ablate_row=ablate_row,
            )
        logits = result.output.logits.float()
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        predictions = logits.argmax(dim=-1)
        for local_index, task_id in enumerate(task_ids.detach().cpu().tolist()):
            losses = per_token[local_index, mask]
            task_loss_sum[task_id] += float(losses.sum().item())
            task_loss_tokens[task_id] += int(losses.numel())
            expected_digits = targets[local_index, digit_positions]
            predicted_digits = predictions[local_index, digit_positions]
            correct = int((predicted_digits == expected_digits).sum().item())
            task_correct[task_id] += correct
            task_total[task_id] += int(expected_digits.numel())
            task_exact[task_id] += int(bool(torch.equal(predicted_digits, expected_digits)))
        if collect_topology:
            diagnostics = result.diagnostics
            assert diagnostics is not None
            activity = diagnostics.activity[:, mask, :].float().mean(dim=1)
            edges = diagnostics.edges[:, mask, :, :].float().mean(dim=1)
            activity_rows.append(activity.detach().cpu().numpy())
            edge_rows.append(edges.detach().cpu().numpy())
            task_rows.append(task_ids.detach().cpu().numpy())
            active_fractions.append(float(diagnostics.logical_active_fraction.detach().cpu()))

    rows = []
    for task_id, task in enumerate(corpus.task_names):
        nll = task_loss_sum[task_id] / max(1, task_loss_tokens[task_id])
        rows.append(
            {
                "task_id": task_id,
                "task": task,
                "nll": nll,
                "ppl": math.exp(min(nll, 20.0)),
                "token_accuracy": task_correct[task_id] / max(1, task_total[task_id]),
                "exact_match": task_exact[task_id] / max(1, int((corpus.task_ids == task_id).sum().item())),
                "examples": int((corpus.task_ids == task_id).sum().item()),
            }
        )
    frame = pd.DataFrame(rows)
    extras: dict[str, np.ndarray | float] = {}
    if collect_topology:
        extras = {
            "activity": np.concatenate(activity_rows, axis=0),
            "edges": np.concatenate(edge_rows, axis=0),
            "task_ids": np.concatenate(task_rows, axis=0),
            "logical_active_fraction": float(np.mean(active_fractions)),
        }
    return frame, extras


def summarize_topology(
    run_name: str,
    corpus: SkillCorpus,
    task_metrics: pd.DataFrame,
    extras: dict[str, np.ndarray | float],
    *,
    null_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    activity = np.asarray(extras["activity"], dtype=np.float64)
    edges = np.asarray(extras["edges"], dtype=np.float64)
    task_ids = np.asarray(extras["task_ids"], dtype=np.int64)
    region_stats = permutation_mi_null(task_ids, activity[:, 1:], seed=null_seed)
    edge_stats = permutation_mi_null(task_ids, edges.reshape(len(edges), -1), seed=null_seed + 1)
    region_rows = []
    edge_rows = []
    task_activity: dict[str, np.ndarray] = {}
    for task_id, task in enumerate(corpus.task_names):
        selected = task_ids == task_id
        mean_activity = activity[selected].mean(axis=0)
        task_activity[task] = mean_activity[1:]
        for row, value in enumerate(mean_activity):
            region_rows.append({"run": run_name, "task": task, "row": row, "mean_activity": float(value)})
        mean_edges = edges[selected].mean(axis=0)
        for source in range(mean_edges.shape[0]):
            for receiver in range(mean_edges.shape[1]):
                if mean_edges[source, receiver] > 0:
                    edge_rows.append(
                        {
                            "run": run_name,
                            "task": task,
                            "source": source,
                            "receiver": receiver,
                            "mean_edge_usage": float(mean_edges[source, receiver]),
                        }
                    )
    reuse = pd.DataFrame(composition_reuse_scores(task_activity, COMPOSITION_MAP, BASE_TASKS))
    reuse.insert(0, "run", run_name)
    summary = {
        "task_region_mi": region_stats,
        "task_edge_mi": edge_stats,
        "logical_active_fraction": float(extras["logical_active_fraction"]),
        "composition_reuse_mean": float(reuse["true_reuse"].mean()),
        "composition_reuse_margin_mean": float(reuse["reuse_margin_vs_best_wrong"].mean()),
        "composition_reuse_positive": int((reuse["reuse_margin_vs_best_wrong"] > 0).sum()),
        "composition_reuse_total": int(len(reuse)),
        "mean_nll": float(task_metrics["nll"].mean()),
        "mean_token_accuracy": float(task_metrics["token_accuracy"].mean()),
        "mean_exact_match": float(task_metrics["exact_match"].mean()),
    }
    return pd.DataFrame(region_rows), pd.DataFrame(edge_rows), reuse, summary


def ablation_analysis(
    model: torch.nn.Module,
    corpus: SkillCorpus,
    baseline: pd.DataFrame,
    *,
    device: torch.device,
    run_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = baseline.set_index("task")
    rows = []
    specificity_rows = []
    for row in range(1, TISSUE_HEIGHT):
        ablated, _ = evaluate(model, corpus, device=device, ablate_row=row)
        by_task = ablated.set_index("task")
        deltas = []
        for task in ALL_TASKS:
            delta_nll = float(by_task.loc[task, "nll"] - base.loc[task, "nll"])
            delta_accuracy = float(base.loc[task, "token_accuracy"] - by_task.loc[task, "token_accuracy"])
            deltas.append((task, delta_nll))
            rows.append(
                {
                    "run": run_name,
                    "row": row,
                    "task": task,
                    "baseline_nll": float(base.loc[task, "nll"]),
                    "ablated_nll": float(by_task.loc[task, "nll"]),
                    "delta_nll": delta_nll,
                    "delta_token_accuracy": delta_accuracy,
                }
            )
        best_task, best_delta = max(deltas, key=lambda item: item[1])
        other = [abs(value) for task, value in deltas if task != best_task]
        specificity = best_delta / max(1e-6, float(np.mean(other)))
        specificity_rows.append((row, best_task, best_delta, specificity))
    frame = pd.DataFrame(rows)
    functional = [item for item in specificity_rows if item[2] >= 0.02 and item[3] >= 1.5]
    summary = {
        "functional_specific_rows": len(functional),
        "functional_specificity_threshold": 1.5,
        "functional_delta_nll_threshold": 0.02,
        "best_rows": [
            {"row": row, "task": task, "delta_nll": delta, "specificity": specificity}
            for row, task, delta, specificity in sorted(specificity_rows, key=lambda item: item[2], reverse=True)
        ],
    }
    return frame, summary


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 015 worker requires CUDA")
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
    run_name = f"r{replicate}-{args.variant}"

    torch.manual_seed(model_seed)
    model = build_sparse_topology_model(VOCAB_SIZE, variant=args.variant)
    parameters = count_parameters(model)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()
    index_schedule = make_index_schedule(
        len(train_corpus.sequences), steps=TRAIN_STEPS, batch_size=BATCH_SIZE, seed=schedule_seed
    )
    depth_schedule = make_depth_schedule(TRAIN_STEPS, seed=depth_seed)
    checkpoint_set = set(CHECKPOINT_STEPS)
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
            "schedule_seed": schedule_seed,
            "depth_seed": depth_seed,
        }
    )

    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        lr = BASE_LR * _lr_multiplier(step, TRAIN_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets, _ = batch_from_indices(train_corpus, index_schedule[step - 1], device=device)
        optimizer.zero_grad(set_to_none=True)
        depths = tuple(int(value) for value in depth_schedule[step - 1])
        synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            result = model.forward_variable(inputs, stage_depths=depths)
            main_loss = masked_language_loss(result.output.logits, targets, mask)
            loss = main_loss + STABILITY_WEIGHT * result.stability_loss + BALANCE_WEIGHT * result.balance_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        training_elapsed += time.perf_counter() - started
        executed_iterations += sum(depths)
        if step in checkpoint_set:
            metrics, _ = evaluate(model, validation_corpus, device=device)
            consumed = step * TOKENS_PER_STEP
            row = {
                "run": run_name,
                "replicate": replicate,
                "variant": args.variant,
                "step": step,
                "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()),
                "main_loss": float(main_loss.detach().item()),
                "stability_loss": float(result.stability_loss.detach().item()),
                "balance_loss": float(result.balance_loss.detach().item()),
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
                f"acc={row['validation_mean_token_accuracy']:.3f} iters={row['avg_recurrent_iterations']:.2f}"
            )

    pd.DataFrame(checkpoint_rows).to_csv(output_dir / f"{run_name}-checkpoints.csv", index=False)
    task_metrics, extras = evaluate(model, validation_corpus, device=device, collect_topology=True)
    task_metrics.insert(0, "run", run_name)
    task_metrics.insert(1, "replicate", replicate)
    task_metrics.insert(2, "variant", args.variant)
    task_metrics.to_csv(output_dir / f"{run_name}-task-metrics.csv", index=False)
    regions, edges, reuse, topology_summary = summarize_topology(
        run_name, validation_corpus, task_metrics, extras, null_seed=null_seed
    )
    regions.to_csv(output_dir / f"{run_name}-task-region.csv", index=False)
    edges.to_csv(output_dir / f"{run_name}-task-edge.csv", index=False)
    reuse.to_csv(output_dir / f"{run_name}-composition-reuse.csv", index=False)
    ablation, ablation_summary = ablation_analysis(
        model, validation_corpus, task_metrics, device=device, run_name=run_name
    )
    ablation.to_csv(output_dir / f"{run_name}-ablation.csv", index=False)

    peak_vram = int(torch.cuda.max_memory_allocated())
    worker = {
        "format": "minicells.language-sparse-topology-worker.v1",
        "run": run_name,
        "replicate": replicate,
        "variant": args.variant,
        "parameters": parameters,
        "model_seed": model_seed,
        "schedule_seed": schedule_seed,
        "depth_seed": depth_seed,
        "null_seed": null_seed,
        "tokens": BUDGET_TOKENS,
        "training_elapsed_seconds": training_elapsed,
        "training_tokens_per_second": BUDGET_TOKENS / training_elapsed,
        "seconds_per_million_tokens": training_elapsed / (BUDGET_TOKENS / 1_000_000),
        "peak_vram_bytes": peak_vram,
        "avg_recurrent_iterations": executed_iterations / TRAIN_STEPS,
        "tissue_height": TISSUE_HEIGHT,
        "active_latent": ACTIVE_LATENT,
        "stability_weight": STABILITY_WEIGHT,
        "balance_weight": BALANCE_WEIGHT,
        "topology": topology_summary,
        "ablation": ablation_summary,
    }
    (output_dir / f"{run_name}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
