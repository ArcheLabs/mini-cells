from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .clm_v2_compute import CLMv2Stats
from .language_models import LanguageModelOutput


BALANCE_WEIGHT = 0.01
ROUTER_Z_WEIGHT = 1e-4
DISTILLATION_WEIGHT = 0.5


@dataclass(frozen=True)
class HandoffStage:
    name: str
    alpha: float
    tokens: int
    local_weight: float


HANDOFF_STAGES = (
    HandoffStage("alpha-075", 0.75, 250_000, 1.0),
    HandoffStage("alpha-050", 0.50, 250_000, 0.75),
    HandoffStage("alpha-025", 0.25, 250_000, 0.50),
    HandoffStage("alpha-000", 0.00, 250_000, 0.25),
)
K_STAGES = ((5, 375_000), (4, 375_000), (3, 375_000))


def normalized_local_loss(stats: CLMv2Stats) -> torch.Tensor:
    return (
        stats.local_relative_mse
        + BALANCE_WEIGHT * stats.balance_loss
        + ROUTER_Z_WEIGHT * stats.router_z_loss
    )


def handoff_loss(
    student: LanguageModelOutput,
    teacher: LanguageModelOutput,
    targets: torch.Tensor,
    stats: CLMv2Stats,
    *,
    local_weight: float,
) -> torch.Tensor:
    student_logits = student.logits.reshape(-1, student.logits.shape[-1])
    teacher_logits = teacher.logits.detach().reshape(-1, teacher.logits.shape[-1])
    ce = F.cross_entropy(student_logits, targets.reshape(-1))
    kl = F.kl_div(student_logits.log_softmax(-1), teacher_logits.softmax(-1), reduction="batchmean")
    return (
        ce + DISTILLATION_WEIGHT * kl
        + local_weight * stats.local_relative_mse
        + BALANCE_WEIGHT * stats.balance_loss
        + ROUTER_Z_WEIGHT * stats.router_z_loss
    )


def save_v2_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def latest_stage_checkpoint(output_dir: Path, replicate: int) -> Path | None:
    paths = sorted(
        output_dir.glob(f"r{replicate}-stage-*.pt"),
        key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
    )
    return paths[-1] if paths else None
