"""Functional PCU composition over an immutable Cellular foundation.

The important distinction in this module is between *parameter* composition
and *function* composition.  A fork contains a complete Cell function whose
non-linearity has already been applied.  Consequently an overlap is composed
as ``C0(x) + (CA(x) - C0(x)) + (CB(x) - C0(x))`` rather than by adding the
three SwiGLU parameter tensors and evaluating one new non-linearity.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cellular import CellProjection, CellularExpert, CellularExperts
from .training import ForkedCell, ForkedCellularExpert


def _cell_forward(cell: nn.Module, hidden_states: Tensor, activation: Any) -> Tensor:
    """Call both the parent Cell and fork Cell contracts uniformly."""
    return cell(hidden_states, activation)


class FunctionalCellDelta(nn.Module):
    """The exact runtime function ``fork(x) - parent(x)``."""

    def __init__(self, parent: CellProjection, fork: ForkedCell | CellProjection) -> None:
        super().__init__()
        if (parent.start, parent.end) != (fork.start, fork.end):
            raise ValueError("functional fork and parent have different Cell ranges")
        self.parent = parent
        self.fork = fork

    def forward(self, hidden_states: Tensor, activation: Any = F.silu) -> Tensor:
        return _cell_forward(self.fork, hidden_states, activation) - _cell_forward(
            self.parent, hidden_states, activation
        )


class ComposedCell(nn.Module):
    """Compose one or more independently trained fork functions exactly."""

    def __init__(self, parent: CellProjection, forks: Iterable[ForkedCell | CellProjection] = ()) -> None:
        super().__init__()
        self.parent = parent
        self.deltas = nn.ModuleList(FunctionalCellDelta(parent, fork) for fork in forks)

    @property
    def fork_count(self) -> int:
        return len(self.deltas)

    def forward(self, hidden_states: Tensor, activation: Any = F.silu) -> Tensor:
        value = self.parent(hidden_states, activation)
        for delta in self.deltas:
            value = value + delta(hidden_states, activation)
        return value


class ComposedCellularExpert(nn.Module):
    """A router-independent expert made from functional Cell deltas."""

    def __init__(
        self,
        parent: CellularExpert,
        forks: Mapping[int, Iterable[ForkedCell | CellProjection]] | None = None,
    ) -> None:
        super().__init__()
        forks = forks or {}
        self.partition = parent.partition
        self.activation = parent.activation
        self.cells = nn.ModuleList(
            ComposedCell(parent_cell, forks.get(index, ()))
            for index, parent_cell in enumerate(parent.cells)
        )
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


class ComposedCellularExperts(nn.Module):
    """Router-facing functional composition with explicit rollback support.

    ``branches`` maps a branch label (``A``/``B``/``JOINT``) to its forked
    expert modules.  The parent router and all routing coefficients are copied
    by reference and are never recomputed by this wrapper.
    """

    def __init__(
        self,
        parent: CellularExperts,
        branches: Mapping[str, Mapping[int, ForkedCellularExpert]],
        active_branches: Iterable[str] = ("A", "B"),
    ) -> None:
        super().__init__()
        self.parent = parent
        self.num_experts = parent.num_experts
        self.hidden_dim = parent.hidden_dim
        self.intermediate_dim = parent.intermediate_dim
        self.partition = parent.partition
        self.active_branches = tuple(str(item) for item in active_branches)
        self._branches = branches
        unknown = sorted(set(self.active_branches) - set(branches))
        if unknown:
            raise ValueError(f"active composition branches are missing: {unknown}")
        self.cells = nn.ModuleList(
            self._compose_expert(parent.cells[index], branches, index)
            for index in range(parent.num_experts)
        )

    def _compose_expert(
        self,
        parent: CellularExpert,
        branches: Mapping[str, Mapping[int, ForkedCellularExpert]],
        expert_index: int,
    ) -> ComposedCellularExpert:
        forks: dict[int, list[ForkedCell]] = {}
        for branch in self.active_branches:
            branch_expert = branches[branch].get(expert_index)
            if branch_expert is None:
                continue
            for cell_index, cell in enumerate(branch_expert.cells):
                # Unselected Cells have zero deltas and would merely duplicate
                # the parent function.  Omitting them keeps the algebra clear.
                if any(parameter.requires_grad for parameter in cell.parameters()):
                    forks.setdefault(cell_index, []).append(cell)
        return ComposedCellularExpert(parent, forks)

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

    def rollback(self, branch: str) -> "ComposedCellularExperts":
        """Return a new functional runtime with one branch removed."""
        branch = str(branch)
        if branch == "all":
            active: tuple[str, ...] = ()
        else:
            active = tuple(item for item in self.active_branches if item != branch)
        if branch != "all" and len(active) == len(self.active_branches):
            raise ValueError(f"branch is not active: {branch}")
        # The original fork modules are retained in the private maps so that
        # rollback is still a functional reconstruction, not parameter reset.
        return ComposedCellularExperts(self.parent, self._branches, active)

    @classmethod
    def from_branches(
        cls,
        parent: CellularExperts,
        branches: Mapping[str, Mapping[int, ForkedCellularExpert]],
        active_branches: Iterable[str] = ("A", "B"),
    ) -> "ComposedCellularExperts":
        return cls(parent, branches, active_branches)


def compose_cellular_experts(
    parent: CellularExperts,
    branches: Mapping[str, Mapping[int, ForkedCellularExpert]],
    active_branches: Iterable[str] = ("A", "B"),
) -> ComposedCellularExperts:
    """Build a functional runtime from independently trained branch forks."""
    return ComposedCellularExperts.from_branches(parent, branches, active_branches)
