"""Cell-local plasticity controller used by Experiment 026.

The controller changes no forward parameter and has no trainable parameter of
its own. Every micro-cell starts at plasticity 1.0, so all granularity arms have
identical age-zero functions and parameter counts. At selected backward passes,
local gradient pressure relative to immediate tissue neighbours determines a
bounded learning-rate multiplier.

Multipliers remain inside configured bounds and have mean exactly 1.0 inside
every tissue. Finer tissues therefore receive finer local adaptation control,
not a systematically larger global learning rate.
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


def _bounded_mean_one(
    values: list[float],
    *,
    minimum: float,
    maximum: float,
) -> list[float]:
    """Scale positive values to bounded mean one using monotone bisection."""
    if not values:
        return []
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("plasticity targets must be positive finite values")
    if not minimum <= 1.0 <= maximum:
        raise ValueError("bounds must contain 1.0")
    if len(values) == 1:
        return [1.0]

    def mean_at(scale: float) -> float:
        return sum(
            min(maximum, max(minimum, scale * value))
            for value in values
        ) / len(values)

    low = 0.0
    high = 1.0
    while mean_at(high) < 1.0:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if mean_at(middle) < 1.0:
            low = middle
        else:
            high = middle
    scale = 0.5 * (low + high)
    result = [
        min(maximum, max(minimum, scale * value))
        for value in values
    ]
    residual = len(values) - sum(result)
    if abs(residual) > 1e-12:
        adjustable = [
            index
            for index, value in enumerate(result)
            if minimum < value < maximum
        ]
        if adjustable:
            correction = residual / len(adjustable)
            for index in adjustable:
                result[index] += correction
    return result


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
        multiplier = (
            float(group.get("plasticity", 1.0))
            if index in cell_groups
            else 1.0
        )
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
    """Update bounded local LR multipliers from current unscaled gradients."""
    config = config or LocalPlasticityConfig()
    telemetry: list[CellPlasticityTelemetry] = []

    for stage, expert_id, tissue in _iter_tissues(model):
        gradients = [_cell_gradient_rms(cell) for cell in tissue.cells]
        if len(gradients) == 1:
            pressures = [1.0]
            targets = [1.0]
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
                    pressure = (
                        gradient + config.epsilon
                    ) / (
                        reference + config.epsilon
                    )
                pressures.append(float(pressure))
                raw_targets.append(pressure**config.pressure_exponent)
            targets = _bounded_mean_one(
                raw_targets,
                minimum=config.minimum,
                maximum=config.maximum,
            )

        previous_values = [float(cell.plasticity.item()) for cell in tissue.cells]
        smoothed_raw = [
            config.ema_decay * previous
            + (1.0 - config.ema_decay) * target
            for previous, target in zip(previous_values, targets, strict=True)
        ]
        smoothed = _bounded_mean_one(
            smoothed_raw,
            minimum=config.minimum,
            maximum=config.maximum,
        )

        for cell_index, value in enumerate(smoothed):
            cell = tissue.cells[cell_index]
            cell.plasticity.fill_(float(value))
            normalized_stress = max(
                0.0,
                min(1.0, (pressures[cell_index] - 0.5) / 1.5),
            )
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


def plasticity_summary(
    rows: list[CellPlasticityTelemetry],
) -> dict[str, float]:
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
