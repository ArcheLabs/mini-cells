"""Fork-minus-parent Cells, deterministic allocation, and bounded branch workers."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cellular import CellProjection, CellularExpert, CellularExperts
from .registry import module_tensor_hash


@dataclass(frozen=True)
class Allocation:
    scores: dict[str, float]
    selected: tuple[str, ...]
    topk_mass: dict[int, float]
    effective_count: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "selected": list(self.selected),
            "topk_mass": {str(key): value for key, value in self.topk_mass.items()},
            "effective_count": self.effective_count,
        }


def allocate_topk(scores: Mapping[str, float], ks: Iterable[int] = (1, 2, 4, 8)) -> Allocation:
    clean = {str(key): max(0.0, float(value)) for key, value in scores.items()}
    ordered = sorted(clean, key=lambda key: (-clean[key], key))
    total = sum(clean.values())
    topk_mass = {int(k): (sum(clean[key] for key in ordered[: int(k)]) / total if total else 0.0) for k in ks}
    square_total = sum(value * value for value in clean.values())
    effective = (total * total / square_total) if square_total else 0.0
    return Allocation(clean, tuple(ordered), topk_mass, effective)


def score_cell_gradients(cell_gradients: Mapping[str, Mapping[str, Tensor]]) -> Allocation:
    scores = {
        cell_id: sum(float(gradient.detach().float().pow(2).sum()) for gradient in values.values())
        / max(1, sum(int(gradient.numel()) for gradient in values.values()))
        for cell_id, values in cell_gradients.items()
    }
    return allocate_topk(scores)


class ForkedCell(nn.Module):
    """A Cell with frozen parent buffers and zero-initialized trainable deltas."""

    _NAMES = ("gate_weight", "up_weight", "down_weight", "gate_bias", "up_bias", "down_bias")

    def __init__(self, parent: CellProjection, trainable: bool) -> None:
        super().__init__()
        self.start, self.end = parent.start, parent.end
        for name in self._NAMES:
            value = getattr(parent, name, None)
            if value is not None:
                self.register_buffer(f"parent_{name}", value.detach().clone())
                self.register_parameter(
                    f"delta_{name}", nn.Parameter(torch.zeros_like(value.detach()), requires_grad=trainable)
                )
            else:
                self.register_buffer(f"parent_{name}", None)
                self.register_parameter(f"delta_{name}", None)

    def _value(self, name: str) -> Tensor | None:
        parent = getattr(self, f"parent_{name}")
        delta = getattr(self, f"delta_{name}")
        return parent + delta if parent is not None and delta is not None else parent

    def forward(self, hidden_states: Tensor, activation: Any = F.silu) -> Tensor:
        gate = F.linear(hidden_states, self._value("gate_weight"), self._value("gate_bias"))
        up = F.linear(hidden_states, self._value("up_weight"), self._value("up_bias"))
        return F.linear(activation(gate) * up, self._value("down_weight"), None)

    def delta_state(self) -> dict[str, Tensor]:
        return {
            name: value.detach().clone()
            for name in self._NAMES
            if (value := getattr(self, f"delta_{name}", None)) is not None
        }


class ForkedCellularExpert(nn.Module):
    def __init__(self, parent: CellularExpert, selected_indices: Iterable[int]) -> None:
        super().__init__()
        selected = {int(index) for index in selected_indices}
        self.partition = parent.partition
        self.activation = parent.activation
        self.cells = nn.ModuleList(
            ForkedCell(cell, index in selected) for index, cell in enumerate(parent.cells)
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

    def delta_state(self) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        for index, cell in enumerate(self.cells):
            for name, value in cell.delta_state().items():
                values[f"cells.{index}.delta_{name}"] = value
        return values


class ForkedCellularExperts(nn.Module):
    """Router-facing collection of fork-minus-parent experts."""

    def __init__(self, parent: CellularExperts, selected: Mapping[int, Iterable[int]]) -> None:
        super().__init__()
        selected = {int(expert): {int(index) for index in indices} for expert, indices in selected.items()}
        self.num_experts = parent.num_experts
        self.hidden_dim = parent.hidden_dim
        self.intermediate_dim = parent.intermediate_dim
        self.partition = parent.partition
        self.cells = nn.ModuleList(
            ForkedCellularExpert(expert, selected.get(index, ()))
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


def selected_delta_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [value for name, value in module.named_parameters() if name.startswith("cells") and ".delta_" in name and value.requires_grad]


def fork_expert(parent: CellularExpert, selected_indices: Iterable[int]) -> ForkedCellularExpert:
    return ForkedCellularExpert(parent, selected_indices)


def fork_initial_delta_norm(fork: ForkedCellularExpert) -> float:
    return float(sum(value.float().pow(2).sum() for value in fork.delta_state().values())) ** 0.5


def foundation_tensor_hashes(module: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in sorted(module.state_dict().items())
    }


def assert_foundation_unchanged(before: Mapping[str, str], module: nn.Module) -> None:
    after = foundation_tensor_hashes(module)
    if dict(before) != after:
        changed = sorted(set(before) | set(after))
        changed = [name for name in changed if before.get(name) != after.get(name)]
        raise RuntimeError(f"FOUNDATION_MUTATION_DETECTED: {changed[:8]}")


@dataclass(frozen=True)
class BranchTrainingConfig:
    learning_rate: float = 1e-3
    max_optimizer_steps: int = 128
    max_training_tokens: int = 500_000
    batch_size: int = 8
    seed: int = 26090501


def train_fork(
    fork: ForkedCellularExpert,
    batches: Iterable[tuple[Tensor, Tensor]],
    config: BranchTrainingConfig,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device | str = "cpu",
) -> list[dict[str, float]]:
    """Train only selected delta parameters under a hard token/step budget."""
    torch.manual_seed(int(config.seed))
    fork.to(device)
    parameters = [value for value in fork.parameters() if value.requires_grad]
    if not parameters:
        raise ValueError("fork has no trainable Cell deltas")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    rows: list[dict[str, float]] = []
    tokens = 0
    for step, (inputs, targets) in enumerate(batches):
        if step >= config.max_optimizer_steps:
            break
        if tokens + int(inputs.numel()) > config.max_training_tokens:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(fork(inputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite branch loss")
        loss.backward()
        optimizer.step()
        tokens += int(inputs.numel())
        rows.append({"step": float(step + 1), "tokens": float(tokens), "loss": float(loss.detach())})
    return rows


def write_training_csv(path: Path, rows: Iterable[Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "tokens", "loss"])
        writer.writeheader()
        writer.writerows(rows)
