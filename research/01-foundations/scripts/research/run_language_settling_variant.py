from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_2d import build_minicells_2d  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    load_tokenizer,
    make_training_schedule,
)
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_scaling import build_minicells_v2  # noqa: E402
from minicells.language_settling import relaxation_forward, settling_forward  # noqa: E402
from minicells.language_stabilization import make_depth_schedule  # noqa: E402

MODELS = ("minicells-v2-settling", "minicells-2d-k4-settling")
CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000)
BUDGET_TOKENS = 2_000_000
BATCH_SIZE = 8
SEQUENCE_LENGTH = 125
SCHEDULE_SEED = 11_011
DEPTH_SEED = 21_011
BASE_SEED = 61_011
WARMUP_STEPS = 100
BASE_LR = 3e-4
WEIGHT_DECAY = 0.1
STATE_STABILITY_WEIGHT = 0.10
LOGIT_CONSISTENCY_WEIGHT = 0.05
VALIDATION_DEPTH = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 012 settling model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def zero_absolute_step_identity(model: torch.nn.Module) -> None:
    """Keep parameter shapes intact while removing iteration-number information."""

    with torch.no_grad():
        for stage in model.stages:
            stage.step_embedding.zero_()


def build_model(name: str, vocab_size: int) -> tuple[torch.nn.Module, int, dict[str, object]]:
    if name == "minicells-v2-settling":
        seed = BASE_SEED
        torch.manual_seed(seed)
        model = build_minicells_v2(vocab_size)
        zero_absolute_step_identity(model)
        return model, seed, {
            "architecture": "1D MiniCells-v2 shared-rule settling",
            "seed": seed,
            "absolute_step_embedding_used": False,
        }
    if name == "minicells-2d-k4-settling":
        seed = BASE_SEED + 4
        torch.manual_seed(seed)
        model = build_minicells_2d(vocab_size, tissue_height=4)
        zero_absolute_step_identity(model)
        return model, seed, {
            "architecture": "2D MiniCells K=4 shared-rule settling",
            "seed": seed,
            "tissue_height": 4,
            "absolute_step_embedding_used": False,
        }
    raise ValueError(name)


def _make_scaler(amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


@torch.no_grad()
def evaluate_at_depth(
    model: torch.nn.Module,
    token_stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    depth: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    residual_sum = [0.0, 0.0, 0.0]
    batches = 0
    for batch_starts in starts:
        inputs, targets = batch_from_starts(token_stream, batch_starts, 128, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            relaxed = relaxation_forward(model, inputs, stage_depths=(depth, depth, depth))
            loss = F.cross_entropy(
                relaxed.output.logits.reshape(-1, relaxed.output.logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
        for index, residual in enumerate(relaxed.stage_last_residuals):
            residual_sum[index] += float(residual.detach().cpu())
        batches += 1
    nll = total_loss / total_tokens
    return {
        "validation_nll": nll,
        "validation_ppl": math.exp(min(nll, 20.0)),
        "validation_tokens": total_tokens,
        "stage1_last_relative_residual": residual_sum[0] / batches,
        "stage2_last_relative_residual": residual_sum[1] / batches,
        "stage3_last_relative_residual": residual_sum[2] / batches,
        "mean_last_relative_residual": sum(residual_sum) / (3.0 * batches),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 012 worker requires CUDA")
    device = torch.device("cuda:0")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = cache_dir / "tokenizer.json"
    train_path = cache_dir / "train-tokens.pt"
    validation_path = cache_dir / "validation-tokens.pt"
    if not tokenizer_path.is_file() or not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Experiment 012 cache is incomplete")

    tokenizer = load_tokenizer(tokenizer_path)
    train_stream = torch.load(train_path, map_location="cpu")
    validation_stream = torch.load(validation_path, map_location="cpu")
    schedule = make_training_schedule(
        int(train_stream.numel()),
        seed=SCHEDULE_SEED,
        budget_tokens=BUDGET_TOKENS,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )
    validation_starts = fixed_validation_starts(
        int(validation_stream.numel()),
        batches=48,
        batch_size=8,
        sequence_length=128,
        seed=41_012,
    )
    depth_schedule = make_depth_schedule(
        schedule.steps,
        seed=DEPTH_SEED,
        min_depth=2,
        max_depth=4,
        stages=3,
    )

    model, seed, model_metadata = build_model(args.model, tokenizer.get_vocab_size())
    parameters = count_parameters(model)
    print({
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": parameters,
        "budget_tokens": schedule.consumed_tokens,
        "depth_range": (2, 4),
        "probe_steps_per_training_step": 3,
        "state_stability_weight": STATE_STABILITY_WEIGHT,
        "logit_consistency_weight": LOGIT_CONSISTENCY_WEIGHT,
    })

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )
    scaler = _make_scaler(True)
    checkpoint_set = set(CHECKPOINTS)
    rows: list[dict[str, object]] = []
    training_elapsed = 0.0
    main_iterations = 0
    probe_iterations = 0

    for step, starts in enumerate(schedule.starts, start=1):
        model.train()
        lr = BASE_LR * _lr_multiplier(step, schedule.steps, WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = batch_from_starts(train_stream, starts, schedule.sequence_length, device)
        optimizer.zero_grad(set_to_none=True)
        depths = depth_schedule[step - 1]

        synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            settled = settling_forward(model, inputs, stage_depths=depths)
            lm_loss = F.cross_entropy(
                settled.output.logits.reshape(-1, settled.output.logits.shape[-1]),
                targets.reshape(-1),
            )
            loss = (
                lm_loss
                + STATE_STABILITY_WEIGHT * settled.state_stability_loss
                + LOGIT_CONSISTENCY_WEIGHT * settled.logit_consistency_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        training_elapsed += time.perf_counter() - started
        main_iterations += sum(depths)
        probe_iterations += 3

        consumed = step * schedule.tokens_per_step
        if consumed in checkpoint_set:
            validation = evaluate_at_depth(
                model,
                validation_stream,
                validation_starts,
                depth=VALIDATION_DEPTH,
                device=device,
            )
            avg_main = main_iterations / step
            avg_total = (main_iterations + probe_iterations) / step
            row = {
                "model": args.model,
                "step": step,
                "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()),
                "lm_loss": float(lm_loss.detach().item()),
                "state_stability_loss": float(settled.state_stability_loss.detach().item()),
                "logit_consistency_loss": float(settled.logit_consistency_loss.detach().item()),
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "training_elapsed_seconds": training_elapsed,
                "training_tokens_per_second": consumed / training_elapsed,
                "seconds_per_million_tokens": training_elapsed / (consumed / 1_000_000),
                "avg_main_recurrent_iterations": avg_main,
                "avg_total_rule_applications_including_probes": avg_total,
                **validation,
            }
            rows.append(row)
            print(
                f"{args.model:28s} tokens={consumed:9d} val_ppl={row['validation_ppl']:.2f} "
                f"resid={row['mean_last_relative_residual']:.5f} "
                f"main_iters={avg_main:.2f} train_tok/s={row['training_tokens_per_second']:.0f}"
            )

    metrics = pd.DataFrame(rows)
    metrics["parameters"] = parameters
    peak_vram = int(torch.cuda.max_memory_allocated())
    metrics["peak_vram_bytes"] = peak_vram
    metrics.to_csv(output_dir / f"{args.model}-checkpoints.csv", index=False)

    checkpoint_path = output_dir / f"{args.model}-2m.pt"
    torch.save(
        {
            "format": "minicells.language-checkpoint.v1",
            "model_name": args.model,
            "consumed_tokens": schedule.consumed_tokens,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_recipe": {
                "shared_rule_no_absolute_step_identity": True,
                "random_depth_range": [2, 4],
                "depth_seed": DEPTH_SEED,
                "state_stability_weight": STATE_STABILITY_WEIGHT,
                "logit_consistency_weight": LOGIT_CONSISTENCY_WEIGHT,
                "probe_steps_per_training_step": 3,
            },
        },
        checkpoint_path,
    )

    worker = {
        "format": "minicells.language-settling-worker.v1",
        "model": args.model,
        "parameters": parameters,
        "model_metadata": model_metadata,
        "seed": seed,
        "schedule_seed": SCHEDULE_SEED,
        "depth_seed": DEPTH_SEED,
        "training_elapsed_seconds": training_elapsed,
        "training_tokens_per_second": schedule.consumed_tokens / training_elapsed,
        "seconds_per_million_tokens": training_elapsed / (schedule.consumed_tokens / 1_000_000),
        "peak_vram_bytes": peak_vram,
        "avg_main_recurrent_iterations": main_iterations / schedule.steps,
        "avg_total_rule_applications_including_probes": (
            main_iterations + probe_iterations
        ) / schedule.steps,
        "state_stability_weight": STATE_STABILITY_WEIGHT,
        "logit_consistency_weight": LOGIT_CONSISTENCY_WEIGHT,
        "absolute_step_embedding_used": False,
        "checkpoint_path": checkpoint_path.name,
    }
    (output_dir / f"{args.model}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
