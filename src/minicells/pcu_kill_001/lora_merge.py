"""Exact additive LoRA weight composition for the matched baseline.

PCU branches compose *functions*.  The matched LoRA control instead composes
its independently trained low-rank weight deltas exactly and evaluates one
SwiGLU function with ``W + dW_A + dW_B``.  Keeping these two semantics separate
prevents the baseline from accidentally inheriting PCU's function-delta merge.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cellular import CellProjection, CellularExpert, CellularExperts
from .lora import LoRACell, MatchedLoRAExpert


class ExactMergedLoRACell(nn.Module):
    """Apply the sum of independent LoRA weight deltas in one Cell forward."""

    def __init__(self, parent: CellProjection, adapters: Iterable[LoRACell] = ()) -> None:
        super().__init__()
        self.parent = parent
        self.adapters = nn.ModuleList(tuple(adapters))
        for adapter in self.adapters:
            if (adapter.start, adapter.end) != (parent.start, parent.end):
                raise ValueError("LoRA adapter and parent have different Cell ranges")

    def _weight(self, name: str) -> Tensor:
        parent = getattr(self.parent, f"{name}_weight")
        if not self.adapters:
            return parent
        delta = None
        for adapter in self.adapters:
            value = adapter.effective_deltas()[name]
            delta = value if delta is None else delta + value
        return parent + delta.to(parent.device, dtype=parent.dtype)

    def forward(self, hidden_states: Tensor, activation: Any = F.silu) -> Tensor:
        gate = F.linear(hidden_states, self._weight("gate"), self.parent.gate_bias)
        up = F.linear(hidden_states, self._weight("up"), self.parent.up_bias)
        return F.linear(activation(gate) * up, self._weight("down"))


class ExactMergedLoRAExpert(nn.Module):
    """One expert whose selected Cells use exact additive LoRA weight deltas."""

    def __init__(self, parent: CellularExpert, branch_experts: Iterable[MatchedLoRAExpert]) -> None:
        super().__init__()
        self.partition = parent.partition
        self.activation = parent.activation
        branches = tuple(branch_experts)
        cells = []
        for index, parent_cell in enumerate(parent.cells):
            adapters = [
                branch.cells[index]
                for branch in branches
                if isinstance(branch.cells[index], LoRACell)
            ]
            cells.append(ExactMergedLoRACell(parent_cell, adapters))
        self.cells = nn.ModuleList(cells)
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


class ExactMergedLoRAExperts(nn.Module):
    """Router-facing exact LoRA union with branch-level activation control."""

    def __init__(
        self,
        parent: CellularExperts,
        branches: Mapping[str, Mapping[int, MatchedLoRAExpert]],
        active_branches: Iterable[str] = ("A", "B"),
    ) -> None:
        super().__init__()
        self.parent = parent
        self.num_experts = parent.num_experts
        self.hidden_dim = parent.hidden_dim
        self.intermediate_dim = parent.intermediate_dim
        self.partition = parent.partition
        self.active_branches = tuple(str(value) for value in active_branches)
        self._branches = branches
        unknown = sorted(set(self.active_branches) - set(branches))
        if unknown:
            raise ValueError(f"active LoRA branches are missing: {unknown}")
        experts = []
        for expert_index, parent_expert in enumerate(parent.cells):
            active = []
            for branch in self.active_branches:
                branch_expert = branches[branch].get(expert_index)
                if branch_expert is not None:
                    if not isinstance(branch_expert, MatchedLoRAExpert):
                        raise TypeError("LoRA exact union received a non-LoRA branch expert")
                    active.append(branch_expert)
            experts.append(ExactMergedLoRAExpert(parent_expert, active))
        self.cells = nn.ModuleList(experts)

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

    def rollback(self, branch: str) -> "ExactMergedLoRAExperts":
        branch = str(branch)
        if branch == "all":
            active: tuple[str, ...] = ()
        else:
            active = tuple(value for value in self.active_branches if value != branch)
        if branch != "all" and len(active) == len(self.active_branches):
            raise ValueError(f"branch is not active: {branch}")
        return ExactMergedLoRAExperts(self.parent, self._branches, active)
