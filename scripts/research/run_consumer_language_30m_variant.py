from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_30m import (  # noqa: E402
    BASE_LR,
    BATCH_SIZE,
    CHECKPOINT_TOKENS,
    CONTEXT_LENGTH,
    MODEL_NAME,
    MODEL_SEED,
    RESUME_INTERVAL_TOKENS,
    SCHEDULE_SEED,
    TARGET_TOKENS,
    TOKENS_PER_STEP,
    TRAIN_SEQUENCE_LENGTH,
    TRANSFORMER_NAME,
    TRANSFORMER_SEED,
    VALIDATION_SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    build_minicells_30m,
    build_transformer_30m,
    memmap_batch,
    open_memmap,
)
from minicells.language_data import fixed_validation_starts, load_tokenizer  # noqa: E402
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_training import FIXED_PROMPTS, GENERATION_SEED, generate_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Experiment 007 model on one visible GPU.")
    parser.add_argument("--model", choices=(MODEL_NAME, TRANSFORMER_NAME), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def lr_multiplier(step: int, total_steps: int) -> float:
    if step <= WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def model_config(name: str, vocab_size: int) -> dict[str, object]:
    if name == MODEL_NAME:
        return {
            "name": MODEL_NAME,
            "vocab_size": vocab_size,
            "context_length": CONTEXT_LENGTH,
            "dim": 720,
            "heads": 8,
            "ffn_dim": 2880,
            "windows": [8, 32, 128],
            "iterations": [4, 4, 4],
            "normalization": "LayerNorm",
            "gru_carry_bias": 2.0,
            "auxiliary_stage_losses": None,
        }
    return {
        "name": TRANSFORMER_NAME,
        "vocab_size": vocab_size,
        "context_length": CONTEXT_LENGTH,
        "dim": 512,
        "heads": 8,
        "ffn_dim": 2048,
        "layers": 9,
        "normalization": "RMSNorm",
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    validation,
    validation_starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for starts in validation_starts:
        inputs, targets = memmap_batch(validation, starts, CONTEXT_LENGTH, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(inputs).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
    nll = total_loss / total_tokens
    return {
        "validation_nll": nll,
        "validation_ppl": math.exp(min(nll, 20.0)),
        "validation_tokens": total_tokens,
    }


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fp16_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        tensor = value.detach().cpu()
        result[key] = tensor.half() if tensor.is_floating_point() else tensor
    return result


def save_inference_artifact(
    model: torch.nn.Module,
    *,
    name: str,
    config: dict[str, object],
    consumed_tokens: int,
    output_path: Path,
) -> None:
    atomic_torch_save(
        {
            "format": "minicells.language-inference.v1",
            "model_name": name,
            "consumed_tokens": consumed_tokens,
            "model_config": config,
            "state_dict": fp16_state_dict(model),
        },
        output_path,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 007 worker requires CUDA")
    device = torch.device("cuda")
    if args.stop_after_tokens <= 0 or args.stop_after_tokens > TARGET_TOKENS:
        raise ValueError("--stop-after-tokens must be in (0, 100M]")
    if args.stop_after_tokens % TOKENS_PER_STEP:
        raise ValueError("--stop-after-tokens must be divisible by tokens per step")

    tokenizer_path = args.cache_dir / "tokenizer.json"
    train_path = args.cache_dir / "train.u16"
    validation_path = args.cache_dir / "validation.u16"
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    train = open_memmap(train_path)
    validation = open_memmap(validation_path)

    seed = MODEL_SEED if args.model == MODEL_NAME else TRANSFORMER_SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.model == MODEL_NAME:
        model = build_minicells_30m(vocab_size)
        match_config: dict[str, object] | None = None
    else:
        model, match_config = build_transformer_30m(vocab_size)
    config = model_config(args.model, vocab_size)
    parameters = count_parameters(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )
    amp = True
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    batch_rng = random.Random(SCHEDULE_SEED)
    metrics: list[dict[str, object]] = []
    generations: list[dict[str, object]] = []
    start_step = 0
    elapsed_before = 0.0
    peak_before = 0

    resume_path = args.resume_from
    if resume_path is None:
        candidate = args.output_dir / "resume" / f"{args.model}-latest.pt"
        resume_path = candidate if candidate.is_file() else None

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        if checkpoint.get("format") != "minicells.language-30m-resume.v1":
            raise RuntimeError(f"unexpected resume checkpoint format: {resume_path}")
        if checkpoint.get("model_name") != args.model:
            raise RuntimeError("resume checkpoint model does not match worker model")
        if checkpoint.get("target_tokens") != TARGET_TOKENS:
            raise RuntimeError("resume checkpoint was created for a different training target")
        if checkpoint.get("model_config") != config:
            raise RuntimeError("resume checkpoint architecture does not match current Experiment 007")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        peak_before = int(checkpoint.get("peak_vram_bytes", 0))
        metrics = list(checkpoint.get("metrics", []))
        generations = list(checkpoint.get("generations", []))
        batch_rng.setstate(checkpoint["batch_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(
            f"resumed {args.model} at {start_step * TOKENS_PER_STEP:,} tokens "
            f"from {resume_path}"
        )

    stop_step = args.stop_after_tokens // TOKENS_PER_STEP
    total_steps = TARGET_TOKENS // TOKENS_PER_STEP
    if start_step > stop_step:
        raise RuntimeError("resume checkpoint is already beyond requested stop point")

    validation_starts = fixed_validation_starts(
        len(validation),
        batches=48,
        batch_size=BATCH_SIZE,
        sequence_length=CONTEXT_LENGTH,
        seed=VALIDATION_SEED,
    )
    high = len(train) - TRAIN_SEQUENCE_LENGTH - 1
    if high <= 0:
        raise RuntimeError("training corpus is too short")

    torch.cuda.reset_peak_memory_stats()
    session_started = time.perf_counter()
    resume_output = args.output_dir / "resume" / f"{args.model}-latest.pt"

    def current_elapsed() -> float:
        return elapsed_before + (time.perf_counter() - session_started)

    def current_peak() -> int:
        return max(peak_before, int(torch.cuda.max_memory_allocated()))

    def save_resume(step: int) -> None:
        atomic_torch_save(
            {
                "format": "minicells.language-30m-resume.v1",
                "model_name": args.model,
                "model_config": config,
                "target_tokens": TARGET_TOKENS,
                "step": step,
                "consumed_tokens": step * TOKENS_PER_STEP,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "batch_rng_state": batch_rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "metrics": metrics,
                "generations": generations,
                "elapsed_seconds": current_elapsed(),
                "peak_vram_bytes": current_peak(),
            },
            resume_output,
        )

    checkpoint_set = set(CHECKPOINT_TOKENS)
    for step in range(start_step + 1, stop_step + 1):
        model.train()
        starts = tuple(batch_rng.randrange(high) for _ in range(BATCH_SIZE))
        lr = BASE_LR * lr_multiplier(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = memmap_batch(train, starts, TRAIN_SEQUENCE_LENGTH, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(inputs)
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()

        consumed = step * TOKENS_PER_STEP
        if consumed in checkpoint_set:
            validation_metrics = evaluate(
                model,
                validation,
                validation_starts,
                device=device,
                amp=True,
            )
            elapsed = current_elapsed()
            row = {
                "model": args.model,
                "step": step,
                "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()),
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "elapsed_seconds": elapsed,
                "tokens_per_second": consumed / elapsed,
                "parameters": parameters,
                "peak_vram_bytes": current_peak(),
                **validation_metrics,
            }
            metrics.append(row)
            print(
                f"{args.model:18s} tokens={consumed:10d} "
                f"train={row['train_loss']:.4f} val_ppl={row['validation_ppl']:.2f} "
                f"tok/s={row['tokens_per_second']:.0f}"
            )
            for prompt_index, prompt in enumerate(FIXED_PROMPTS):
                text = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    device=device,
                    max_context=CONTEXT_LENGTH,
                    max_new_tokens=64,
                    seed=GENERATION_SEED + consumed + prompt_index,
                    amp=True,
                )
                generations.append(
                    {
                        "model": args.model,
                        "consumed_tokens": consumed,
                        "prompt": prompt,
                        "text": text,
                    }
                )

        should_resume_save = (
            consumed % RESUME_INTERVAL_TOKENS == 0
            or consumed == args.stop_after_tokens
            or consumed in checkpoint_set
        )
        if should_resume_save:
            save_resume(step)

    elapsed = current_elapsed()
    peak = current_peak()
    complete = stop_step == total_steps

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / f"{args.model}-checkpoints.csv", index=False)
    (args.output_dir / f"{args.model}-generations.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    worker_summary = {
        "format": "minicells.language-30m-worker.v1",
        "model": args.model,
        "parameters": parameters,
        "model_config": config,
        "transformer_match": match_config,
        "consumed_tokens": stop_step * TOKENS_PER_STEP,
        "target_tokens": TARGET_TOKENS,
        "complete": complete,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (stop_step * TOKENS_PER_STEP) / elapsed if elapsed > 0 else 0.0,
        "peak_vram_bytes": peak,
        "resume_checkpoint": str(resume_output),
    }
    (args.output_dir / f"{args.model}-worker.json").write_text(
        json.dumps(worker_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if complete:
        inference_name = (
            "minicells-30m-v0-fp16.pt"
            if args.model == MODEL_NAME
            else "transformer-30m-fp16.pt"
        )
        save_inference_artifact(
            model,
            name=args.model,
            config=config,
            consumed_tokens=TARGET_TOKENS,
            output_path=args.output_dir / inference_name,
        )
        print(f"saved final FP16 inference artifact: {inference_name}")
    else:
        print(
            f"partial run complete at {stop_step * TOKENS_PER_STEP:,} tokens; "
            f"resume checkpoint: {resume_output}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
