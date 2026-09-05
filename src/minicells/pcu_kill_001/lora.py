"""Parameter-matched LoRA control on the same parent experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cellular import CellProjection, CellularExpert


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float = 1.0


class LoRACell(nn.Module):
    def __init__(self, parent: CellProjection, config: LoRAConfig, trainable: bool = True) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("parent_gate_weight", parent.gate_weight.detach().clone())
        self.register_buffer("parent_up_weight", parent.up_weight.detach().clone())
        self.register_buffer("parent_down_weight", parent.down_weight.detach().clone())
        hidden, width = parent.gate_weight.shape[1], parent.gate_weight.shape[0]
        self.gate_a = nn.Parameter(torch.empty(config.rank, hidden), requires_grad=trainable)
        self.gate_b = nn.Parameter(torch.zeros(width, config.rank), requires_grad=trainable)
        self.up_a = nn.Parameter(torch.empty(config.rank, hidden), requires_grad=trainable)
        self.up_b = nn.Parameter(torch.zeros(width, config.rank), requires_grad=trainable)
        self.down_a = nn.Parameter(torch.empty(config.rank, width), requires_grad=trainable)
        self.down_b = nn.Parameter(torch.zeros(hidden, config.rank), requires_grad=trainable)
        for parameter in (self.gate_a, self.up_a, self.down_a):
            nn.init.normal_(parameter, std=0.02)
        self.scale = float(config.alpha) / max(1, config.rank)

    def _adapt(self, parent: Tensor, left: Tensor, right: Tensor) -> Tensor:
        return parent + (right @ left) * self.scale

    def forward(self, hidden_states: Tensor, activation: Any = F.silu) -> Tensor:
        gate = F.linear(hidden_states, self._adapt(self.parent_gate_weight, self.gate_a, self.gate_b))
        up = F.linear(hidden_states, self._adapt(self.parent_up_weight, self.up_a, self.up_b))
        down = self._adapt(self.parent_down_weight, self.down_a, self.down_b)
        return F.linear(activation(gate) * up, down)

    def state_delta(self) -> dict[str, Tensor]:
        return {
            "gate_a": self.gate_a.detach().clone(), "gate_b": self.gate_b.detach().clone(),
            "up_a": self.up_a.detach().clone(), "up_b": self.up_b.detach().clone(),
            "down_a": self.down_a.detach().clone(), "down_b": self.down_b.detach().clone(),
        }


class MatchedLoRAExpert(nn.Module):
    def __init__(self, parent: CellularExpert, selected_indices: Iterable[int], config: LoRAConfig) -> None:
        super().__init__()
        selected = {int(index) for index in selected_indices}
        self.cells = nn.ModuleList(
            LoRACell(cell, config, index in selected) if index in selected else CellProjection(
                _projection_from_cell(cell), 0, cell.end - cell.start
            ) for index, cell in enumerate(parent.cells)
        )
        self.activation = parent.activation
        if parent.down_bias is not None:
            self.register_buffer("down_bias", parent.down_bias.detach().clone())
        else:
            self.register_buffer("down_bias", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = None
        for cell in self.cells:
            value = cell(hidden_states, self.activation)
            output = value if output is None else output + value
        if self.down_bias is not None:
            output = output + self.down_bias
        return output


def _projection_from_cell(cell: CellProjection):
    # Local import avoids a public constructor dependency cycle.
    from .cellular import ExpertProjections

    gate_bias = cell.gate_bias.detach().clone() if cell.gate_bias is not None else None
    up_bias = cell.up_bias.detach().clone() if cell.up_bias is not None else None
    return ExpertProjections(
        cell.gate_weight.detach(), cell.up_weight.detach(), cell.down_weight.detach(), gate_bias, up_bias
    )


def lora_parameter_count(hidden_size: int, cell_width: int, selected_cells: int, rank: int) -> int:
    per_cell = 3 * int(rank) * (int(hidden_size) + int(cell_width))
    return per_cell * int(selected_cells)


def choose_matched_rank(target_parameters: int, hidden_size: int, cell_width: int, selected_cells: int, tolerance: float = 0.10) -> int:
    if target_parameters <= 0 or selected_cells <= 0:
        raise ValueError("target_parameters and selected_cells must be positive")
    ideal = target_parameters / (3 * (hidden_size + cell_width) * selected_cells)
    candidates = [max(1, int(ideal) + offset) for offset in (-1, 0, 1, 2)]
    valid = [rank for rank in candidates if abs(lora_parameter_count(hidden_size, cell_width, selected_cells, rank) - target_parameters) / target_parameters <= tolerance]
    if not valid:
        return max(1, round(ideal))
    return min(valid, key=lambda rank: abs(lora_parameter_count(hidden_size, cell_width, selected_cells, rank) - target_parameters))


def merge_lora_state(base: dict[str, Tensor], branch_a: dict[str, Tensor], branch_b: dict[str, Tensor]) -> dict[str, Tensor]:
    if set(base) != set(branch_a) or set(base) != set(branch_b):
        raise ValueError("LoRA branches do not share a parameter schema")
    # Add deltas relative to the common zero-initialized fork, never average.
    return {key: base[key] + (branch_a[key] - base[key]) + (branch_b[key] - base[key]) for key in base}
