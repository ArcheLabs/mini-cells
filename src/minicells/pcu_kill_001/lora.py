"""Parameter-matched LoRA controls with exact factor composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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
        self.start, self.end = parent.start, parent.end
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

    def effective_deltas(self) -> dict[str, Tensor]:
        """Return the actual ``Delta W`` matrices applied by this adapter."""
        return {
            "gate": (self.gate_b @ self.gate_a) * self.scale,
            "up": (self.up_b @ self.up_a) * self.scale,
            "down": (self.down_b @ self.down_a) * self.scale,
        }


class ComposedLoRACell(nn.Module):
    """Functionally compose two trained LoRA Cells without cross terms."""

    def __init__(self, parent: CellProjection, adapters: Iterable[LoRACell]) -> None:
        super().__init__()
        self.parent = parent
        self.adapters = nn.ModuleList(adapters)

    def forward(self, hidden_states: Tensor, activation: Any = F.silu) -> Tensor:
        value = self.parent(hidden_states, activation)
        for adapter in self.adapters:
            value = value + adapter(hidden_states, activation) - self.parent(hidden_states, activation)
        return value


class MatchedLoRAExpert(nn.Module):
    def __init__(self, parent: CellularExpert, selected_indices: Iterable[int], config: LoRAConfig) -> None:
        super().__init__()
        selected = {int(index) for index in selected_indices}
        cells = []
        for index, cell in enumerate(parent.cells):
            if index in selected:
                cells.append(LoRACell(cell, config, True))
            else:
                frozen = CellProjection(_projection_from_cell(cell), 0, cell.end - cell.start)
                for parameter in frozen.parameters():
                    parameter.requires_grad_(False)
                cells.append(frozen)
        self.cells = nn.ModuleList(cells)
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


class MatchedLoRAExperts(nn.Module):
    """Router-preserving collection of independently trainable LoRA experts."""

    def __init__(self, parent: Any, selected: Mapping[int, Iterable[int]], config: LoRAConfig) -> None:
        super().__init__()
        self.parent = parent
        self.num_experts = parent.num_experts
        self.hidden_dim = parent.hidden_dim
        self.intermediate_dim = parent.intermediate_dim
        self.partition = parent.partition
        self.cells = nn.ModuleList(
            MatchedLoRAExpert(expert, selected.get(index, ()), config)
            for index, expert in enumerate(parent.cells)
        )

    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0])
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = self.cells[expert_idx](hidden_states[token_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


def selected_lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    # ``MatchedLoRAExperts`` retains the parent expert collection for identity
    # and runtime provenance.  It is an nn.Module, so an unrestricted
    # ``module.parameters()`` walk would accidentally optimize the frozen
    # foundation as well as the adapter factors.
    return [
        value
        for child in module.modules()
        if isinstance(child, LoRACell)
        for value in child.parameters()
        if value.requires_grad and value.ndim > 0
    ]


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


def merge_lora_factors(
    branch_a: Mapping[str, Tensor],
    branch_b: Mapping[str, Tensor],
    *,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
) -> dict[str, Tensor]:
    """Rank-concatenate two LoRA states so ``B@A`` is exactly additive.

    The factors are stored as ``A[r, in]`` and ``B[out, r]``.  Scaling the
    right factors before concatenation gives ``B_merged @ A_merged =
    scale_a*(B_a@A_a) + scale_b*(B_b@A_b)``.  In particular, this never uses
    the invalid ``(B_a+B_b) @ (A_a+A_b)`` construction, which creates cross
    terms.
    """
    prefixes = ("gate", "up", "down")
    expected = {f"{prefix}_{suffix}" for prefix in prefixes for suffix in ("a", "b")}
    if set(branch_a) != expected or set(branch_b) != expected:
        raise ValueError("LoRA branches do not share the expected factor schema")
    result: dict[str, Tensor] = {}
    for prefix in prefixes:
        left_a, left_b = branch_a[f"{prefix}_a"], branch_b[f"{prefix}_a"]
        right_a, right_b = branch_a[f"{prefix}_b"], branch_b[f"{prefix}_b"]
        if left_a.shape[1:] != left_b.shape[1:] or right_a.shape[:1] != right_b.shape[:1]:
            raise ValueError(f"LoRA branches have incompatible {prefix} factor shapes")
        if left_a.shape[0] != right_a.shape[1] or left_b.shape[0] != right_b.shape[1]:
            raise ValueError(f"LoRA {prefix} factors are not matrix-compatible")
        result[f"{prefix}_a"] = torch.cat((left_a, left_b), dim=0)
        result[f"{prefix}_b"] = torch.cat((right_a * float(scale_a), right_b * float(scale_b)), dim=1)
    return result


def merged_effective_deltas(
    branch_a: Mapping[str, Tensor],
    branch_b: Mapping[str, Tensor],
    *,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
) -> dict[str, Tensor]:
    """Compute merged matrices directly; useful for an exact numeric audit."""
    state = merge_lora_factors(branch_a, branch_b, scale_a=scale_a, scale_b=scale_b)
    return {
        prefix: state[f"{prefix}_b"] @ state[f"{prefix}_a"]
        for prefix in ("gate", "up", "down")
    }


def merge_lora_state(base: dict[str, Tensor], branch_a: dict[str, Tensor], branch_b: dict[str, Tensor]) -> dict[str, Tensor]:
    """Backward-compatible exact merge for the historical state API.

    ``base`` is accepted to keep callers honest about a common parent schema;
    LoRA factor tensors themselves are branch-local and must be concatenated,
    not added element-wise.
    """
    if set(base) != set(branch_a) or set(base) != set(branch_b):
        raise ValueError("LoRA branches do not share a parameter schema")
    return merge_lora_factors(branch_a, branch_b)
