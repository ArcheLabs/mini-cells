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
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_2d import build_minicells_2d  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    load_tokenizer,
    make_training_schedule,
)
from minicells.language_depth_ablation import (  # noqa: E402
    resolve_stage_depths,
    step_embedding_rms,
    variant_by_code,
)
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_scaling import build_minicells_v2  # noqa: E402
from minicells.language_stabilization import (  # noqa: E402
    make_depth_schedule,
    scale_step_embeddings,
    stabilizing_forward,
)
from minicells.language_training import evaluate_language_model  # noqa: E402


CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000)
BUDGET_TOKENS = 2_000_000
BATCH_SIZE = 8
SEQUENCE_LENGTH = 125
SCHEDULE_SEED = 11_011
DEPTH_SEED = 21_011
BASE_SEED_1D = 61_011
BASE_SEED_2D = 61_015
WARMUP_STEPS = 100
BASE_LR = 3e-4
WEIGHT_DECAY = 0.1
DEPTH_EVALS = (2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 013 factorial cell on one visible GPU.")
    parser.add_argument("--topology", choices=("1d", "2d"), required=True)
    parser.add_argument("--variant", choices=tuple("ABCDEFGH"), required=True)
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


def _make_scaler(amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


def build_model(topology: str, vocab_size: int, step_scale: float) -> tuple[torch.nn.Module, int, dict[str, object]]:
    if topology == "1d":
        seed = BASE_SEED_1D
        torch.manual_seed(seed)
        model = build_minicells_v2(vocab_size)
        metadata: dict[str, object] = {"architecture": "1D MiniCells-v2", "seed": seed}
    elif topology == "2d":
        seed = BASE_SEED_2D
        torch.manual_seed(seed)
        model = build_minicells_2d(vocab_size, tissue_height=4)
        metadata = {"architecture": "2D MiniCells K=4", "seed": seed, "tissue_height": 4}
    else:
        raise ValueError(topology)
    if step_scale != 1.0:
        scale_step_embeddings(model, step_scale)
    metadata["step_embedding_init_scale"] = step_scale
    return model, seed, metadata


@torch.no_grad()
def evaluate_depth_robustness(
    model: torch.nn.Module,
    validation_stream: torch.Tensor,
    *,
    device: torch.device,
) -> pd.DataFrame:
    starts = fixed_validation_starts(
        int(validation_stream.numel()),
        batches=24,
        batch_size=8,
        sequence_length=128,
        seed=51_013,
    )
    rows: list[dict[str, object]] = []
    for depth in DEPTH_EVALS:
        total_loss = 0.0
        total_tokens = 0
        for batch_starts in starts:
            inputs, targets = batch_from_starts(validation_stream, batch_starts, 128, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = stabilizing_forward(model, inputs, stage_depths=(depth, depth, depth))
                loss = F.cross_entropy(
                    result.output.logits.reshape(-1, result.output.logits.shape[-1]).float(),
                    targets.reshape(-1),
                    reduction="sum",
                )
            total_loss += float(loss.item())
            total_tokens += int(targets.numel())
        nll = total_loss / total_tokens
        rows.append(
            {
                "depth_per_stage": depth,
                "total_recurrent_iterations": depth * 3,
                "validation_nll": nll,
                "validation_ppl": math.exp(min(nll, 20.0)),
                "validation_tokens": total_tokens,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 013 worker requires CUDA")
    device = torch.device("cuda:0")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = variant_by_code(args.variant)
    run_name = f"{args.topology}-{variant.code}"

    tokenizer_path = cache_dir / "tokenizer.json"
    train_path = cache_dir / "train-tokens.pt"
    validation_path = cache_dir / "validation-tokens.pt"
    if not tokenizer_path.is_file() or not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Experiment 013 cache is incomplete")

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

    model, seed, model_metadata = build_model(
        args.topology,
        tokenizer.get_vocab_size(),
        variant.step_embedding_init_scale,
    )
    parameters = count_parameters(model)
    initial_step_rms = step_embedding_rms(model)
    print(
        {
            "run": run_name,
            "gpu": torch.cuda.get_device_name(0),
            "parameters": parameters,
            "random_depth": variant.random_depth,
            "step_embedding_init_scale": variant.step_embedding_init_scale,
            "stability_weight": variant.stability_weight,
            "initial_step_embedding_rms": initial_step_rms,
        }
    )

    # Match Experiment 011's stochastic stream after model construction.
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
    executed_recurrent_steps = 0

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
        scheduled_depths = depth_schedule[step - 1]
        depths = resolve_stage_depths(variant, scheduled_depths)

        synchronize(device)
        train_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            stability_value = None
            use_stabilizing_path = variant.random_depth or variant.uses_stability_loss
            if use_stabilizing_path:
                result = stabilizing_forward(model, inputs, stage_depths=depths)
                output = result.output
                main_loss = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1),
                )
                stability_value = result.stability_loss
                loss = main_loss + variant.stability_weight * result.stability_loss
            else:
                output = model(inputs)
                main_loss = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1),
                )
                loss = main_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        training_elapsed += time.perf_counter() - train_started
        executed_recurrent_steps += sum(depths)

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
            avg_recurrent = executed_recurrent_steps / step
            row = {
                "run": run_name,
                "topology": args.topology,
                "variant": variant.code,
                "random_depth": variant.random_depth,
                "step_embedding_init_scale": variant.step_embedding_init_scale,
                "stability_weight": variant.stability_weight,
                "step": step,
                "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()),
                "main_loss": float(main_loss.detach().item()),
                "stability_loss": float(stability_value.detach().item()) if stability_value is not None else math.nan,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "training_elapsed_seconds": training_elapsed,
                "training_tokens_per_second": consumed / training_elapsed,
                "seconds_per_million_tokens": training_elapsed / (consumed / 1_000_000),
                "avg_recurrent_iterations": avg_recurrent,
                "recurrent_iteration_fraction": avg_recurrent / 12.0,
                **validation,
            }
            rows.append(row)
            print(
                f"{run_name:6s} tokens={consumed:9d} ppl={row['validation_ppl']:.2f} "
                f"tok/s={row['training_tokens_per_second']:.0f} iters={avg_recurrent:.2f}"
            )

    metrics = pd.DataFrame(rows)
    peak_vram = int(torch.cuda.max_memory_allocated())
    final_step_rms = step_embedding_rms(model)
    metrics["parameters"] = parameters
    metrics["peak_vram_bytes"] = peak_vram
    metrics.to_csv(output_dir / f"{run_name}-checkpoints.csv", index=False)

    model.eval()
    depth_frame = evaluate_depth_robustness(model, validation_stream, device=device)
    depth_frame.insert(0, "run", run_name)
    depth_frame.insert(1, "topology", args.topology)
    depth_frame.insert(2, "variant", variant.code)
    depth_frame.to_csv(output_dir / f"{run_name}-depth-eval.csv", index=False)
    by_depth = depth_frame.set_index("depth_per_stage")
    robustness_ratio = float(depth_frame["validation_ppl"].max() / depth_frame["validation_ppl"].min())

    worker = {
        "format": "minicells.language-depth-ablation-worker.v1",
        "run": run_name,
        "topology": args.topology,
        "variant": variant.code,
        "parameters": parameters,
        "model_metadata": model_metadata,
        "seed": seed,
        "schedule_seed": SCHEDULE_SEED,
        "depth_seed": DEPTH_SEED if variant.random_depth else None,
        "random_depth": variant.random_depth,
        "step_embedding_init_scale": variant.step_embedding_init_scale,
        "stability_weight": variant.stability_weight,
        "initial_step_embedding_rms": initial_step_rms,
        "final_step_embedding_rms": final_step_rms,
        "step_embedding_rms_growth_ratio": final_step_rms / initial_step_rms if initial_step_rms > 0 else None,
        "training_elapsed_seconds": training_elapsed,
        "training_tokens_per_second": schedule.consumed_tokens / training_elapsed,
        "seconds_per_million_tokens": training_elapsed / (schedule.consumed_tokens / 1_000_000),
        "peak_vram_bytes": peak_vram,
        "avg_recurrent_iterations": executed_recurrent_steps / schedule.steps,
        "depth_robustness_ratio_2_to_4": robustness_ratio,
        "ppl_depth2": float(by_depth.loc[2, "validation_ppl"]),
        "ppl_depth3": float(by_depth.loc[3, "validation_ppl"]),
        "ppl_depth4": float(by_depth.loc[4, "validation_ppl"]),
    }
    (output_dir / f"{run_name}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
