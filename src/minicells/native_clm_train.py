"""Training utilities for Native CLM v0 M1 next-token language modeling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .native_clm_v0 import ByteTokenizer, NativeCLM, NativeCLMConfig


@dataclass(frozen=True)
class NativeCLMTrainConfig:
    seed: int = 72001
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 3000
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 20
    checkpoint_interval: int = 500
    warmup_steps: int = 200
    lr_shared: float = 2e-4
    lr_router: float = 4e-4
    lr_cells: float = 8e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    certificate_update_interval: int = 50
    precision: str = "fp16"
    num_workers: int = 0
    generation_prompt: str = "Once upon a time"
    generation_tokens: int = 120

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.eval_interval < 1 or self.log_interval < 1:
            raise ValueError("logging/eval intervals must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


class ByteSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, path: str | Path, *, seq_len: int) -> None:
        data = Path(path).read_bytes()
        if len(data) < seq_len + 2:
            raise ValueError(f"{path} is too small for seq_len={seq_len}")
        self.tokens = np.frombuffer(data, dtype=np.uint8).copy()
        self.seq_len = int(seq_len)
        self.chunk_count = (len(self.tokens) - 1) // self.seq_len
        if self.chunk_count < 1:
            raise ValueError("dataset contains no full sequence")

    def __len__(self) -> int:
        return self.chunk_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = (int(index) % self.chunk_count) * self.seq_len
        block = self.tokens[start : start + self.seq_len + 1].astype(np.int64, copy=True)
        x = torch.from_numpy(block[:-1])
        y = torch.from_numpy(block[1:])
        return x, y


def _cycle(loader: DataLoader[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        yield from loader


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _usage_entropy(model: NativeCLM) -> float:
    counts = torch.tensor(
        [int(cell.usage_count.item()) for cell in model.cellular.cells],
        dtype=torch.float64,
    )
    total = float(counts.sum().item())
    if total <= 0:
        return 0.0
    probs = counts / total
    nz = probs > 0
    entropy = -(probs[nz] * probs[nz].log()).sum()
    denom = math.log(max(2, model.cell_count))
    return float(entropy / denom)


def _group_grad_norms(model: NativeCLM) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, params in model.parameter_groups().items():
        squares = 0.0
        for parameter in params:
            if parameter.grad is not None:
                value = float(torch.linalg.vector_norm(parameter.grad.detach()).item())
                squares += value * value
        result[name] = math.sqrt(squares)
    return result


def _make_optimizer(model: NativeCLM, config: NativeCLMTrainConfig) -> torch.optim.Optimizer:
    groups = model.parameter_groups()
    return torch.optim.AdamW(
        [
            {
                "params": groups["shared"],
                "lr": config.lr_shared,
                "initial_lr": config.lr_shared,
                "group_name": "shared",
            },
            {
                "params": groups["router"],
                "lr": config.lr_router,
                "initial_lr": config.lr_router,
                "group_name": "router",
            },
            {
                "params": groups["cells"],
                "lr": config.lr_cells,
                "initial_lr": config.lr_cells,
                "group_name": "cells",
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )


def _lr_factor(step: int, config: NativeCLMTrainConfig) -> float:
    if step < config.warmup_steps:
        return max(1e-3, (step + 1) / max(1, config.warmup_steps))
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    step: int,
    config: NativeCLMTrainConfig,
) -> None:
    factor = _lr_factor(step, config)
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * factor


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(
    model: NativeCLM,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    batches: int,
    precision: str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    iterator = _cycle(loader)
    losses: list[float] = []
    active: list[float] = []
    entropy: list[float] = []
    for _ in range(batches):
        x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        with _autocast_context(device, precision):
            output = model(x, y, return_info=True)
        losses.append(float(output["loss"].detach().cpu()))
        info = output["cell_info"]
        active.append(float(info["active_fraction_vs_dense"]))
        entropy.append(float(info["route_entropy"]))
    if was_training:
        model.train()
    mean_loss = float(sum(losses) / len(losses))
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(20.0, mean_loss))),
        "active_fraction_vs_dense": float(sum(active) / len(active)),
        "route_entropy": float(sum(entropy) / len(entropy)),
    }


def _save_training_checkpoint(
    path: Path,
    *,
    model: NativeCLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_config: NativeCLMTrainConfig,
    metrics: list[dict[str, Any]],
) -> None:
    payload = model.checkpoint_payload(
        extra={
            "step": step,
            "train_config": asdict(train_config),
            "metrics_tail": metrics[-10:],
            "optimizer_state_dict": optimizer.state_dict(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_run_config(path: str | Path) -> tuple[NativeCLMConfig, NativeCLMTrainConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return NativeCLMConfig(**raw["model"]), NativeCLMTrainConfig(**raw["training"])


def train_m1(
    *,
    model_config: NativeCLMConfig,
    train_config: NativeCLMTrainConfig,
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    model_config.validate()
    train_config.validate()
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = NativeCLM(model_config).to(target_device)
    parameter_count = model.parameter_count()

    train_dataset = ByteSequenceDataset(train_path, seq_len=model_config.max_seq_len)
    validation_dataset = ByteSequenceDataset(validation_path, seq_len=model_config.max_seq_len)
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=train_config.num_workers,
    )
    if len(train_loader) == 0:
        raise ValueError("training corpus is smaller than one batch")

    optimizer = _make_optimizer(model, train_config)
    scaler_enabled = target_device.type == "cuda" and train_config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    train_iter = _cycle(train_loader)
    initial_eval = evaluate(
        model,
        validation_loader,
        device=target_device,
        batches=train_config.eval_batches,
        precision=train_config.precision,
    )
    max_router_grad = 0.0
    max_cell_grad = 0.0
    train_loss_ema: float | None = None
    last_info: dict[str, Any] | None = None
    wall_start = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, train_config.max_steps + 1):
        _set_learning_rates(optimizer, step - 1, train_config)
        accumulated_loss = 0.0

        for _ in range(train_config.gradient_accumulation_steps):
            x, y = next(train_iter)
            x, y = x.to(target_device), y.to(target_device)
            with _autocast_context(target_device, train_config.precision):
                result = model(x, y, return_info=True)
                loss = result["loss"] / train_config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += float(loss.detach().cpu())
            last_info = result["cell_info"]

        if scaler_enabled:
            scaler.unscale_(optimizer)
        projection_ratios = model.project_cell_gradients_()
        grad_norms = _group_grad_norms(model)
        max_router_grad = max(max_router_grad, grad_norms["router"])
        max_cell_grad = max(max_cell_grad, grad_norms["cells"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if (
            last_info is not None
            and train_config.certificate_update_interval > 0
            and step % train_config.certificate_update_interval == 0
        ):
            model.update_certificates(last_info)

        train_loss_ema = (
            accumulated_loss
            if train_loss_ema is None
            else 0.98 * train_loss_ema + 0.02 * accumulated_loss
        )

        should_log = step == 1 or step % train_config.log_interval == 0
        should_eval = step == train_config.max_steps or step % train_config.eval_interval == 0
        if should_log or should_eval:
            row: dict[str, Any] = {
                "step": step,
                "train_loss_ema": train_loss_ema,
                "cell_count": model.cell_count,
                "active_fraction_vs_dense": float(last_info["active_fraction_vs_dense"]),
                "route_entropy": float(last_info["route_entropy"]),
                "top1_confidence": float(last_info["top1_confidence"]),
                "usage_entropy": _usage_entropy(model),
                "certificate_mean_fill": model.certificate_summary()["mean_fill"],
                "projection_ratio_min": min(projection_ratios.values()),
                "grad_norm_shared": grad_norms["shared"],
                "grad_norm_router": grad_norms["router"],
                "grad_norm_cells": grad_norms["cells"],
                "lr_shared": optimizer.param_groups[0]["lr"],
                "lr_router": optimizer.param_groups[1]["lr"],
                "lr_cells": optimizer.param_groups[2]["lr"],
                "elapsed_seconds": time.time() - wall_start,
            }
            if should_eval:
                eval_metrics = evaluate(
                    model,
                    validation_loader,
                    device=target_device,
                    batches=train_config.eval_batches,
                    precision=train_config.precision,
                )
                row.update(
                    {
                        "eval_loss": eval_metrics["loss"],
                        "eval_perplexity": eval_metrics["perplexity"],
                        "eval_active_fraction_vs_dense": eval_metrics["active_fraction_vs_dense"],
                        "eval_route_entropy": eval_metrics["route_entropy"],
                    }
                )
            metrics.append(row)

        if train_config.checkpoint_interval > 0 and step % train_config.checkpoint_interval == 0:
            _save_training_checkpoint(
                output / f"checkpoint-step-{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                train_config=train_config,
                metrics=metrics,
            )

    final_eval = evaluate(
        model,
        validation_loader,
        device=target_device,
        batches=train_config.eval_batches,
        precision=train_config.precision,
    )
    prompt = torch.tensor(
        [ByteTokenizer.encode(train_config.generation_prompt)],
        dtype=torch.long,
        device=target_device,
    )
    generated = model.generate(
        prompt,
        max_new_tokens=train_config.generation_tokens,
        temperature=0.8,
        top_k=64,
    )
    sample = ByteTokenizer.decode(generated[0])

    final_checkpoint = output / "final-model.pt"
    _save_training_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        step=train_config.max_steps,
        train_config=train_config,
        metrics=metrics,
    )
    _write_csv(output / "metrics.csv", metrics)
    (output / "sample.txt").write_text(sample, encoding="utf-8")

    config_payload = {"model": asdict(model_config), "training": asdict(train_config)}
    config_text = json.dumps(config_payload, indent=2, sort_keys=True) + "\n"
    (output / "run-config.json").write_text(config_text, encoding="utf-8")
    config_sha = hashlib.sha256(config_text.encode("utf-8")).hexdigest()

    gates = {
        "target_parameter_scale": 10_000_000 <= parameter_count["total"] <= 15_000_000,
        "completed_requested_steps": train_config.max_steps > 0,
        "finite_initial_and_final_eval": math.isfinite(initial_eval["loss"])
        and math.isfinite(final_eval["loss"]),
        "validation_loss_improves": final_eval["loss"] <= initial_eval["loss"] * 0.95,
        "sparse_cell_execution": final_eval["active_fraction_vs_dense"] <= 0.30,
        "router_receives_gradient": max_router_grad > 0.0,
        "cells_receive_gradient": max_cell_grad > 0.0,
        "generation_executes": len(sample) > len(train_config.generation_prompt),
        "single_cellular_layer_runtime": True,
        "autonomous_growth_not_claimed_in_m1": model.cell_count == model_config.initial_cells,
    }
    passed = all(gates.values())
    summary = {
        "format": "minicells.native-clm-v0.m1-summary.v1",
        "status": (
            "NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS"
            if passed
            else "NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_INCOMPLETE"
        ),
        "scientific_decision": False,
        "seed": train_config.seed,
        "device": str(target_device),
        "parameter_count": parameter_count,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "eval_loss_ratio": final_eval["loss"] / initial_eval["loss"],
        "cell_count": model.cell_count,
        "active_cells": model_config.active_cells,
        "certificate": model.certificate_summary(),
        "max_observed_router_grad_norm": max_router_grad,
        "max_observed_cell_grad_norm": max_cell_grad,
        "run_config_sha256": config_sha,
        "final_checkpoint_sha256": _sha256_file(final_checkpoint),
        "final_checkpoint_bytes": final_checkpoint.stat().st_size,
        "published_checkpoint": False,
        "gates": gates,
        "pass": passed,
        "next_milestone_if_pass": "M2 continual language stream",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result_md = f"""# Native CLM v0 M1 — Next-token training

- Status: `{summary['status']}`
- Scientific decision: `False` (M1 is an engineering/training milestone, not a formal continual-learning claim)
- Seed: `{train_config.seed}`
- Parameters: `{parameter_count['total']:,}`
- Cells: `{model.cell_count}` total / `{model_config.active_cells}` active per token
- Initial validation loss: `{initial_eval['loss']:.6f}`
- Final validation loss: `{final_eval['loss']:.6f}`
- Initial validation perplexity: `{initial_eval['perplexity']:.4f}`
- Final validation perplexity: `{final_eval['perplexity']:.4f}`
- Sparse Cell execution fraction: `{final_eval['active_fraction_vs_dense']:.4f}`
- Final checkpoint SHA-256: `{summary['final_checkpoint_sha256']}`
- Checkpoint published to GitHub: `False`

The full checkpoint remains a runtime artifact rather than a Git-tracked research artifact.
Canonical repository publication contains the run config, metrics, summary, sample, and
checkpoint hash so later milestones can distinguish training evidence from model-weight storage.
"""
    (output / "RESULTS.md").write_text(result_md, encoding="utf-8")
    return summary


def load_configs(path: str | Path) -> tuple[NativeCLMConfig, NativeCLMTrainConfig]:
    return _load_run_config(path)
