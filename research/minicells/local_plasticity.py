"""Cell-local plasticity controller used by Experiment 026.

The controller changes no forward parameter and has no trainable parameter of
its own. Every micro-cell starts at plasticity 1.0, so all granularity arms have
identical age-zero functions and parameter counts. After each backward pass,
local gradient pressure relative to immediate tissue neighbours determines a
bounded learning-rate multiplier for the next optimizer step.

The multipliers are renormalized to mean 1.0 inside every tissue. This prevents
a finer tissue from receiving a systematically larger global learning rate and
makes the granularity comparison about the resolution of local adaptation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .developmental_tissue import TissueFFN


@dataclass(frozen=True)
class LocalPlasticityConfig:
    ema_decay: float = 0.95
    pressure_exponent: float = 0.5
    minimum: float = 0.5
    maximum: float = 2.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.pressure_exponent <= 0.0:
            raise ValueError("pressure_exponent must be positive")
        if self.minimum <= 0.0 or self.maximum < self.minimum:
            raise ValueError("plasticity bounds must be positive and ordered")
        if not self.minimum <= 1.0 <= self.maximum:
            raise ValueError("plasticity bounds must contain 1.0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class CellPlasticityTelemetry:
    stage: int
    expert_id: str
    cell_index: int
    gradient_rms: float
    local_pressure: float
    target_plasticity: float
    plasticity: float


def _cell_gradient_rms(cell: nn.Module) -> float:
    square_sum = 0.0
    elements = 0
    for parameter in cell.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach().float()
        square_sum += float(detached.square().sum().item())
        elements += detached.numel()
    if elements == 0:
        return 0.0
    return math.sqrt(square_sum / elements)


def _iter_tissues(model: nn.Module):
    for stage_index, stage in enumerate(getattr(model, "stages", ())):
        bank = getattr(stage, "program_bank", None)
        experts = getattr(bank, "experts", None)
        if experts is None:
            continue
        items = experts.items() if hasattr(experts, "items") else enumerate(experts)
        for expert_id, expert in items:
            if isinstance(expert, TissueFFN):
                yield stage_index, str(expert_id), expert


def build_local_plasticity_optimizer(
    model: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> tuple[torch.optim.AdamW, dict[tuple[int, str, int], int]]:
    """Build AdamW groups with one group per micro-cell plus one shared group."""
    if lr <= 0.0:
        raise ValueError("lr must be positive")

    cell_parameter_ids: set[int] = set()
    groups: list[dict[str, Any]] = []
    group_index: dict[tuple[int, str, int], int] = {}
    for stage, expert_id, tissue in _iter_tissues(model):
        for cell_index, cell in enumerate(tissue.cells):
            parameters = list(cell.parameters())
            if not parameters:
                raise RuntimeError("micro-cell unexpectedly has no parameters")
            cell_parameter_ids.update(id(parameter) for parameter in parameters)
            group_index[(stage, expert_id, cell_index)] = len(groups)
            groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "base_lr": lr,
                    "plasticity": 1.0,
                    "stage": stage,
                    "expert_id": expert_id,
                    "cell_index": cell_index,
                }
            )

    shared = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in cell_parameter_ids
    ]
    if shared:
        groups.append(
            {
                "params": shared,
                "lr": lr,
                "base_lr": lr,
                "plasticity": 1.0,
                "stage": None,
                "expert_id": None,
                "cell_index": None,
            }
        )

    optimizer = torch.optim.AdamW(
        groups,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    return optimizer, group_index


def set_global_lr(
    optimizer: torch.optim.Optimizer,
    group_index: dict[tuple[int, str, int], int],
    *,
    base_lr: float,
) -> None:
    """Apply a scheduled base LR while preserving each cell's multiplier."""
    if base_lr < 0.0:
        raise ValueError("base_lr must be non-negative")
    cell_groups = set(group_index.values())
    for index, group in enumerate(optimizer.param_groups):
        multiplier = float(group.get("plasticity", 1.0)) if index in cell_groups else 1.0
        group["base_lr"] = base_lr
        group["lr"] = base_lr * multiplier


@torch.no_grad()
def update_local_plasticity(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    group_index: dict[tuple[int, str, int], int],
    *,
    base_lr: float,
    config: LocalPlasticityConfig | None = None,
) -> list[CellPlasticityTelemetry]:
    """Update local LR multipliers from the current backward-pass gradients.

    Pressure is the cell gradient RMS divided by the mean RMS of its immediate
    neighbours. A single-cell tissue uses itself as the local reference and
    therefore remains exactly at multiplier 1.0. Targets are bounded and then
    renormalized to mean 1.0 within each tissue before EMA smoothing.
    """
    config = config or LocalPlasticityConfig()
    telemetry: list[CellPlasticityTelemetry] = []

    for stage, expert_id, tissue in _iter_tissues(model):
        gradients = [_cell_gradient_rms(cell) for cell in tissue.cells]
        if len(gradients) == 1:
            targets = [1.0]
            pressures = [1.0]
        else:
            pressures = []
            raw_targets = []
            for cell_index, gradient in enumerate(gradients):
                neighbours = tissue.neighbors(cell_index)
                reference_values = [gradients[index] for index in neighbours]
                reference = (
                    sum(reference_values) / len(reference_values)
                    if reference_values
                    else sum(gradients) / len(gradients)
                )
                if gradient <= config.epsilon and reference <= config.epsilon:
                    pressure = 1.0
                else:
                    pressure = (gradient + config.epsilon) / (reference + config.epsilon)
                pressures.append(float(pressure))
                raw_targets.append(
                    min(
                        config.maximum,
                        max(config.minimum, pressure**config.pressure_exponent),
                    )
                )
            target_mean = sum(raw_targets) / len(raw_targets)
            targets = [target / max(target_mean, config.epsilon) for target in raw_targets]
            targets = [min(config.maximum, max(config.minimum, target)) for target in targets]
            second_mean = sum(targets) / len(targets)
            targets = [target / max(second_mean, config.epsilon) for target in targets]

        smoothed = []
        for cell_index, target in enumerate(targets):
            cell = tissue.cells[cell_index]
            previous = float(cell.plasticity.item())
            value = config.ema_decay * previous + (1.0 - config.ema_decay) * float(target)
            smoothed.append(value)
        mean_smoothed = sum(smoothed) / len(smoothed)
        smoothed = [value / max(mean_smoothed, config.epsilon) for value in smoothed]

        for cell_index, value in enumerate(smoothed):
            cell = tissue.cells[cell_index]
            cell.plasticity.fill_(float(value))
            normalized_stress = max(0.0, min(1.0, (pressures[cell_index] - 0.5) / 1.5))
            cell.stress.mul_(config.ema_decay).add_(
                (1.0 - config.ema_decay) * normalized_stress
            )
            optimizer_group = optimizer.param_groups[
                group_index[(stage, expert_id, cell_index)]
            ]
            optimizer_group["plasticity"] = float(value)
            optimizer_group["base_lr"] = base_lr
            optimizer_group["lr"] = base_lr * float(value)
            telemetry.append(
                CellPlasticityTelemetry(
                    stage=stage,
                    expert_id=expert_id,
                    cell_index=cell_index,
                    gradient_rms=float(gradients[cell_index]),
                    local_pressure=float(pressures[cell_index]),
                    target_plasticity=float(targets[cell_index]),
                    plasticity=float(value),
                )
            )
    return telemetry


def plasticity_summary(rows: list[CellPlasticityTelemetry]) -> dict[str, float]:
    if not rows:
        return {
            "mean_plasticity": 1.0,
            "min_plasticity": 1.0,
            "max_plasticity": 1.0,
            "mean_local_pressure": 1.0,
        }
    values = [row.plasticity for row in rows]
    pressures = [row.local_pressure for row in rows]
    return {
        "mean_plasticity": float(sum(values) / len(values)),
        "min_plasticity": float(min(values)),
        "max_plasticity": float(max(values)),
        "mean_local_pressure": float(sum(pressures) / len(pressures)),
    }
