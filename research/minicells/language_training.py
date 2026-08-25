from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .language_data import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, BatchSchedule, batch_from_starts
from .language_models import LanguageModelOutput


CHECKPOINT_TOKENS = (125_000, 250_000, 500_000)
GENERATION_SEED = 65005
FIXED_PROMPTS = (
    "Once upon a time",
    "There was a little girl",
    "The dog was afraid because",
    "One day Tom",
    "Lily wanted to",
)


@dataclass
class TrainingResult:
    name: str
    metrics: pd.DataFrame
    generations: list[dict[str, object]]
    elapsed_seconds: float
    peak_vram_bytes: int
    checkpoint_path: Path


def language_loss(
    output: LanguageModelOutput,
    targets: torch.Tensor,
    *,
    auxiliary_weights: tuple[float, float] | None,
) -> torch.Tensor:
    main = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1))
    if auxiliary_weights is None:
        return main
    if len(output.stage_logits) < 3:
        raise ValueError("auxiliary loss requires three NCA stage outputs")
    total = main
    for weight, logits in zip(auxiliary_weights, output.stage_logits[:2]):
        total = total + weight * F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    return total


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _token_label(tokens: int) -> str:
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    if tokens % 1_000 == 0:
        return f"{tokens // 1_000}k"
    return str(tokens)


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    token_stream: torch.Tensor,
    validation_starts: tuple[tuple[int, ...], ...],
    *,
    sequence_length: int,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for starts in validation_starts:
        inputs, targets = batch_from_starts(token_stream, starts, sequence_length, device)
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


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: str,
    *,
    device: torch.device,
    max_context: int = 128,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 0,
    amp: bool,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError(f"prompt tokenized to an empty sequence: {prompt!r}")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sequence = list(ids)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    forbidden = {
        token_id
        for token in (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN)
        if (token_id := tokenizer.token_to_id(token)) is not None
    }
    for _ in range(max_new_tokens):
        context = sequence[-max_context:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(input_ids).logits[0, -1].float()
        for token_id in forbidden:
            logits[token_id] = -torch.inf
        logits = logits / temperature
        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k=k)
        probabilities = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probabilities, 1, generator=generator)
        next_id = int(indices[choice].item())
        if eos_id is not None and next_id == eos_id:
            break
        sequence.append(next_id)
    return tokenizer.decode(sequence, skip_special_tokens=True)


def train_language_model(
    *,
    name: str,
    model: torch.nn.Module,
    train_stream: torch.Tensor,
    validation_stream: torch.Tensor,
    schedule: BatchSchedule,
    validation_starts: tuple[tuple[int, ...], ...],
    tokenizer: object,
    output_dir: Path,
    device: torch.device,
    auxiliary_weights: tuple[float, float] | None,
    seed: int,
    base_lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 50,
    checkpoint_tokens: tuple[int, ...] = CHECKPOINT_TOKENS,
    final_checkpoint_label: str | None = None,
) -> TrainingResult:
    if not checkpoint_tokens:
        raise ValueError("checkpoint_tokens must not be empty")
    if tuple(sorted(set(checkpoint_tokens))) != checkpoint_tokens:
        raise ValueError("checkpoint_tokens must be unique and strictly increasing")
    for tokens in checkpoint_tokens:
        if tokens <= 0 or tokens > schedule.consumed_tokens:
            raise ValueError(f"checkpoint {tokens} is outside the training budget")
        if tokens % schedule.tokens_per_step != 0:
            raise ValueError(
                f"checkpoint {tokens} must be divisible by tokens_per_step={schedule.tokens_per_step}"
            )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    amp = device.type == "cuda"
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total_steps = schedule.steps
    checkpoint_set = set(checkpoint_tokens)
    rows: list[dict[str, object]] = []
    generations: list[dict[str, object]] = []
    started = time.perf_counter()

    for step, starts in enumerate(schedule.starts, start=1):
        model.train()
        lr = base_lr * _lr_multiplier(step, total_steps, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = batch_from_starts(
            train_stream,
            starts,
            schedule.sequence_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(inputs)
            loss = language_loss(output, targets, auxiliary_weights=auxiliary_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()

        consumed = step * schedule.tokens_per_step
        if consumed in checkpoint_set:
            validation = evaluate_language_model(
                model,
                validation_stream,
                validation_starts,
                sequence_length=128,
                device=device,
                amp=amp,
            )
            elapsed = time.perf_counter() - started
            throughput = consumed / elapsed
            row = {
                "model": name,
                "step": step,
                "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()),
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "elapsed_seconds": elapsed,
                "tokens_per_second": throughput,
                **validation,
            }
            rows.append(row)
            print(
                f"{name:18s} tokens={consumed:9d} "
                f"train={row['train_loss']:.4f} val_ppl={row['validation_ppl']:.2f} "
                f"tok/s={throughput:.0f}"
            )
            for prompt_index, prompt in enumerate(FIXED_PROMPTS):
                text = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    device=device,
                    seed=GENERATION_SEED + consumed + prompt_index,
                    amp=amp,
                )
                generations.append(
                    {
                        "model": name,
                        "consumed_tokens": consumed,
                        "prompt": prompt,
                        "text": text,
                    }
                )

    elapsed = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = final_checkpoint_label or _token_label(schedule.consumed_tokens)
    checkpoint_path = output_dir / f"{name}-{suffix}.pt"
    torch.save(
        {
            "format": "minicells.language-checkpoint.v1",
            "model_name": name,
            "consumed_tokens": schedule.consumed_tokens,
            "checkpoint_tokens": list(checkpoint_tokens),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return TrainingResult(
        name=name,
        metrics=pd.DataFrame(rows),
        generations=generations,
        elapsed_seconds=elapsed,
        peak_vram_bytes=peak_vram,
        checkpoint_path=checkpoint_path,
    )


def estimate_learning_slope(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return 0.0
    x = torch.log(torch.tensor(frame["consumed_tokens"].to_numpy(), dtype=torch.float64))
    y = torch.log(torch.tensor(frame["validation_nll"].to_numpy(), dtype=torch.float64))
    x_centered = x - x.mean()
    denominator = float((x_centered.square()).sum().item())
    if denominator == 0:
        return 0.0
    slope = float((x_centered * (y - y.mean())).sum().item() / denominator)
    return -slope
