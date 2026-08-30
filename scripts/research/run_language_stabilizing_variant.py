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
from minicells.language_halting import adaptive_forward  # noqa: E402
from minicells.language_models import (  # noqa: E402
    build_parameter_matched_transformer,
    count_parameters,
)
from minicells.language_scaling import build_minicells_v2  # noqa: E402
from minicells.language_stabilization import (  # noqa: E402
    make_depth_schedule,
    scale_step_embeddings,
    stabilizing_forward,
)
from minicells.language_training import evaluate_language_model  # noqa: E402


MODELS = (
    "transformer-s",
    "minicells-v2-fixed",
    "minicells-v2-stable",
    "minicells-2d-k4-fixed",
    "minicells-2d-k4-stable",
)
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
STABILITY_WEIGHT = 0.10
STEP_EMBEDDING_SCALE = 0.25
HALTING_THRESHOLDS = (0.0075, 0.0100, 0.0125, 0.0150, 0.0200)
HALTING_PREFIXES = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 011 model on one visible GPU.")
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


def build_model(name: str, vocab_size: int) -> tuple[torch.nn.Module, int, dict[str, object]]:
    if name.startswith("minicells-v2"):
        seed = BASE_SEED
        torch.manual_seed(seed)
        model = build_minicells_v2(vocab_size)
        metadata: dict[str, object] = {"architecture": "1D MiniCells-v2", "seed": seed}
        if name.endswith("stable"):
            scale_step_embeddings(model, STEP_EMBEDDING_SCALE)
            metadata["step_embedding_scale"] = STEP_EMBEDDING_SCALE
        return model, seed, metadata

    if name.startswith("minicells-2d-k4"):
        seed = BASE_SEED + 4
        torch.manual_seed(seed)
        model = build_minicells_2d(vocab_size, tissue_height=4)
        metadata = {"architecture": "2D MiniCells K=4", "seed": seed, "tissue_height": 4}
        if name.endswith("stable"):
            scale_step_embeddings(model, STEP_EMBEDDING_SCALE)
            metadata["step_embedding_scale"] = STEP_EMBEDDING_SCALE
        return model, seed, metadata

    if name == "transformer-s":
        seed = BASE_SEED + 1
        torch.manual_seed(seed)
        target = count_parameters(build_minicells_v2(vocab_size))
        model, config = build_parameter_matched_transformer(vocab_size, target)
        return model, seed, {"architecture": "parameter-matched Transformer-S", **config, "seed": seed}

    raise ValueError(name)


def _make_scaler(amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


@torch.no_grad()
def probe_halting(
    model: torch.nn.Module,
    validation_stream: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    if not hasattr(model, "stages"):
        return pd.DataFrame(), None
    starts = fixed_validation_starts(
        int(validation_stream.numel()),
        batches=HALTING_PREFIXES,
        batch_size=1,
        sequence_length=128,
        seed=31_011,
    )
    amp = device.type == "cuda"
    rows: list[dict[str, object]] = []
    configs: list[float | None] = [None, *HALTING_THRESHOLDS]
    for threshold in configs:
        total_loss = 0.0
        total_steps = 0
        count = 0
        for one_start in starts:
            inputs, targets = batch_from_starts(validation_stream, one_start, 128, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if threshold is None:
                    output = model(inputs)
                    steps = 12
                else:
                    adaptive = adaptive_forward(model, inputs, threshold=threshold, min_iterations=1)
                    output = adaptive.output
                    steps = adaptive.total_steps
                loss = F.cross_entropy(output.logits[:, -1, :].float(), targets[:, -1])
            total_loss += float(loss.item())
            total_steps += int(steps)
            count += 1
        nll = total_loss / count
        rows.append(
            {
                "mode": "fixed" if threshold is None else "adaptive",
                "threshold": math.nan if threshold is None else threshold,
                "last_token_ppl": math.exp(min(nll, 20.0)),
                "avg_total_steps": total_steps / count,
                "iteration_fraction": (total_steps / count) / 12.0,
            }
        )
    frame = pd.DataFrame(rows)
    fixed = frame.loc[frame["mode"] == "fixed"].iloc[0]
    frame["ppl_ratio_to_fixed"] = frame["last_token_ppl"] / float(fixed["last_token_ppl"])
    adaptive = frame.loc[frame["mode"] == "adaptive"].copy()
    viable = adaptive.loc[adaptive["ppl_ratio_to_fixed"] <= 1.01]
    best = None
    if not viable.empty:
        selected = viable.sort_values(["iteration_fraction", "ppl_ratio_to_fixed"]).iloc[0]
        best = {
            "threshold": float(selected["threshold"]),
            "ppl_ratio_to_fixed": float(selected["ppl_ratio_to_fixed"]),
            "avg_total_steps": float(selected["avg_total_steps"]),
            "iteration_fraction": float(selected["iteration_fraction"]),
            "theoretical_iteration_saving": float(1.0 - selected["iteration_fraction"]),
        }
    return frame, best


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 011 worker requires CUDA")
    device = torch.device("cuda:0")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = cache_dir / "tokenizer.json"
    train_path = cache_dir / "train-tokens.pt"
    validation_path = cache_dir / "validation-tokens.pt"
    if not tokenizer_path.is_file() or not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Experiment 011 cache is incomplete")

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
        seed=41_011,
    )
    depth_schedule = make_depth_schedule(schedule.steps, seed=DEPTH_SEED)

    model, seed, model_metadata = build_model(args.model, tokenizer.get_vocab_size())
    parameters = count_parameters(model)
    stable = args.model.endswith("stable")
    recurrent = hasattr(model, "stages")
    print(
        {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "parameters": parameters,
            "stable_training": stable,
            "budget_tokens": schedule.consumed_tokens,
            "checkpoints": CHECKPOINTS,
        }
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    amp = True
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )
    scaler = _make_scaler(amp)
    checkpoint_set = set(CHECKPOINTS)
    rows: list[dict[str, object]] = []
    training_elapsed = 0.0
    executed_recurrent_steps = 0
    stable_steps_seen = 0

    for step, starts in enumerate(schedule.starts, start=1):
        model.train()
        lr = BASE_LR * _lr_multiplier(step, schedule.steps, WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = batch_from_starts(
            train_stream,
            starts,
            schedule.sequence_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)

        synchronize(device)
        train_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            stability_value = None
            if stable:
                depths = depth_schedule[step - 1]
                stabilized = stabilizing_forward(model, inputs, stage_depths=depths)
                output = stabilized.output
                main_loss = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1),
                )
                loss = main_loss + STABILITY_WEIGHT * stabilized.stability_loss
                stability_value = stabilized.stability_loss
                executed_recurrent_steps += sum(depths)
                stable_steps_seen += 1
            else:
                output = model(inputs)
                loss = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1),
                )
                if recurrent:
                    executed_recurrent_steps += 12
                    stable_steps_seen += 1
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        training_elapsed += time.perf_counter() - train_started

        consumed = step * schedule.tokens_per_step
        if consumed in checkpoint_set:
            validation = evaluate_language_model(
                model,
                validation_stream,
                validation_starts,
                sequence_length=128,
                device=device,
                amp=True,
            )
            avg_recurrent = (
                executed_recurrent_steps / stable_steps_seen if recurrent and stable_steps_seen else math.nan
            )
            row = {
                "model": args.model,
                "step": step,
                "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()),
                "stability_loss": (
                    float(stability_value.detach().item()) if stability_value is not None else math.nan
                ),
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "training_elapsed_seconds": training_elapsed,
                "training_tokens_per_second": consumed / training_elapsed,
                "seconds_per_million_tokens": training_elapsed / (consumed / 1_000_000),
                "avg_recurrent_iterations": avg_recurrent,
                "recurrent_iteration_fraction": avg_recurrent / 12.0 if recurrent else math.nan,
                **validation,
            }
            rows.append(row)
            print(
                f"{args.model:24s} tokens={consumed:9d} val_ppl={row['validation_ppl']:.2f} "
                f"train_tok/s={row['training_tokens_per_second']:.0f} "
                f"iters={avg_recurrent if recurrent else float('nan'):.2f}"
            )

    metrics = pd.DataFrame(rows)
    metrics["parameters"] = parameters
    peak_vram = int(torch.cuda.max_memory_allocated())
    metrics["peak_vram_bytes"] = peak_vram
    metrics.to_csv(output_dir / f"{args.model}-checkpoints.csv", index=False)

    halting_frame, best_halting = probe_halting(model, validation_stream, device=device)
    if not halting_frame.empty:
        halting_frame.insert(0, "model", args.model)
        halting_frame.to_csv(output_dir / f"{args.model}-halting.csv", index=False)

    checkpoint_path = output_dir / f"{args.model}-2m.pt"
    torch.save(
        {
            "format": "minicells.language-checkpoint.v1",
            "model_name": args.model,
            "consumed_tokens": schedule.consumed_tokens,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_recipe": {
                "stable": stable,
                "stability_weight": STABILITY_WEIGHT if stable else 0.0,
                "step_embedding_scale": STEP_EMBEDDING_SCALE if stable else 1.0,
                "random_depth_range": [2, 4] if stable else [4, 4],
                "depth_seed": DEPTH_SEED if stable else None,
            },
        },
        checkpoint_path,
    )

    worker = {
        "format": "minicells.language-stabilizing-cost-worker.v1",
        "model": args.model,
        "parameters": parameters,
        "model_metadata": model_metadata,
        "seed": seed,
        "schedule_seed": SCHEDULE_SEED,
        "depth_seed": DEPTH_SEED if stable else None,
        "stable_training": stable,
        "stability_weight": STABILITY_WEIGHT if stable else 0.0,
        "step_embedding_scale": STEP_EMBEDDING_SCALE if stable else 1.0,
        "training_elapsed_seconds": training_elapsed,
        "training_tokens_per_second": schedule.consumed_tokens / training_elapsed,
        "seconds_per_million_tokens": training_elapsed / (schedule.consumed_tokens / 1_000_000),
        "peak_vram_bytes": peak_vram,
        "avg_recurrent_iterations": (
            executed_recurrent_steps / stable_steps_seen if recurrent and stable_steps_seen else None
        ),
        "best_adaptive_halting_within_1pct_ppl": best_halting,
        "checkpoint_path": checkpoint_path.name,
    }
    (output_dir / f"{args.model}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
