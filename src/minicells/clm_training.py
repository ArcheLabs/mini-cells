from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .language_models import LanguageModelOutput


@dataclass
class ComputeDual:
    target: float
    learning_rate: float = 1e-2
    value: float = 0.0

    def penalty(self, observed: torch.Tensor) -> torch.Tensor:
        return observed * self.value - self.target * self.value

    def update(self, observed: float) -> None:
        self.value = max(0.0, self.value + self.learning_rate * (observed - self.target))


@dataclass(frozen=True)
class CLMContinuationPhase:
    name: str
    target_program_ratio: float
    target_cell_ratio: float
    routing_mode: str
    program_top_k: int | None = None


DEFAULT_CONTINUATION = (
    CLMContinuationPhase("dense", 1.0, 1.0, "dense"),
    CLMContinuationPhase("program-75", 0.75, 1.0, "soft_program"),
    CLMContinuationPhase("program-50", 0.50, 1.0, "soft_program"),
    CLMContinuationPhase("hard-program-75", 0.75, 1.0, "hard_program", 6),
    CLMContinuationPhase("hard-program-50", 0.50, 1.0, "hard_program", 4),
    CLMContinuationPhase("cell-75", 0.50, 0.75, "soft_cell_hard_program", 4),
    CLMContinuationPhase("cell-50", 0.50, 0.50, "hard", 4),
)


def distillation_loss(
    student: LanguageModelOutput,
    teacher: LanguageModelOutput,
    targets: torch.Tensor,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ce = F.cross_entropy(student.logits.reshape(-1, student.logits.shape[-1]), targets.reshape(-1))
    student_logits = student.logits.reshape(-1, student.logits.shape[-1]) / temperature
    teacher_logits = teacher.logits.detach().reshape(-1, teacher.logits.shape[-1]) / temperature
    kl = F.kl_div(
        student_logits.log_softmax(-1), teacher_logits.softmax(-1), reduction="batchmean"
    )
    return ce + beta * temperature**2 * kl


def dead_program_survival_loss(
    program_usage_ema: torch.Tensor, *, minimum_usage: float = 0.01
) -> torch.Tensor:
    return (minimum_usage - program_usage_ema).clamp_min(0).square().mean()


def save_clm_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    program_dual: ComputeDual,
    cell_dual: ComputeDual,
    phase: int,
    step: int,
    teacher_source: str,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "dual_lambda_program": program_dual.__dict__,
        "dual_lambda_cell": cell_dual.__dict__,
        "phase": phase,
        "step": step,
        "target_program_ratio": program_dual.target,
        "target_cell_ratio": cell_dual.target,
        "teacher_source": teacher_source,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def load_clm_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload["scheduler_state"] is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    torch.random.set_rng_state(payload["rng_state"].cpu())
    if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return payload


def quality_gate(
    sparse_perplexity: float,
    dense_perplexity: float,
    *,
    maximum_ratio: float = 1.03,
) -> bool:
    if dense_perplexity <= 0:
        raise ValueError("dense_perplexity must be positive")
    return sparse_perplexity / dense_perplexity <= maximum_ratio


def compute_constraint_loss(
    program_ratio: torch.Tensor,
    cell_ratio: torch.Tensor,
    *,
    program_dual: ComputeDual,
    cell_dual: ComputeDual,
) -> torch.Tensor:
    return program_dual.penalty(program_ratio) + cell_dual.penalty(cell_ratio)
