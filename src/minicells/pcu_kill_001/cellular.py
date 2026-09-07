"""Exact SwiGLU expert partitioning and router-preserving cellularization."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class UnsupportedFoundationArchitecture(RuntimeError):
    """Raised when the loaded checkpoint cannot be safely cellularized."""


@dataclass(frozen=True)
class CellPartition:
    """A contiguous, exhaustive partition of an expert intermediate axis."""

    intermediate_size: int
    cells: int = 4

    def __post_init__(self) -> None:
        if self.intermediate_size <= 0 or self.cells <= 0:
            raise ValueError("intermediate_size and cells must be positive")
        if self.intermediate_size % self.cells:
            raise ValueError("cell partition must divide intermediate_size exactly")

    @property
    def cell_size(self) -> int:
        return self.intermediate_size // self.cells

    def ranges(self) -> tuple[tuple[int, int], ...]:
        width = self.cell_size
        return tuple((index * width, (index + 1) * width) for index in range(self.cells))

    def validate(self) -> bool:
        ranges = self.ranges()
        return (
            ranges[0][0] == 0
            and ranges[-1][1] == self.intermediate_size
            and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        )


@dataclass(frozen=True)
class ExpertProjections:
    """One expert's SwiGLU projection matrices in PyTorch linear convention."""

    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    gate_bias: Tensor | None = None
    up_bias: Tensor | None = None
    down_bias: Tensor | None = None
    fused_order: str = "gate_up"

    @property
    def hidden_size(self) -> int:
        return int(self.gate_weight.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_weight.shape[0])

    def validate(self) -> None:
        if self.gate_weight.shape != self.up_weight.shape:
            raise UnsupportedFoundationArchitecture("gate/up projection shapes differ")
        expected_down = (self.hidden_size, self.intermediate_size)
        if tuple(self.down_weight.shape) != expected_down:
            raise UnsupportedFoundationArchitecture(
                f"down projection has shape {tuple(self.down_weight.shape)}, expected {expected_down}"
            )
        for name, bias in (("gate", self.gate_bias), ("up", self.up_bias)):
            if bias is not None and tuple(bias.shape) != (self.intermediate_size,):
                raise UnsupportedFoundationArchitecture(f"{name} bias shape is not intermediate-sized")
        if self.down_bias is not None and tuple(self.down_bias.shape) != (self.hidden_size,):
            raise UnsupportedFoundationArchitecture("down bias shape is not hidden-sized")


def _weight(value: Any) -> Tensor:
    tensor = value if isinstance(value, Tensor) else getattr(value, "weight", None)
    if not isinstance(tensor, Tensor):
        raise UnsupportedFoundationArchitecture("projection has no tensor weight")
    return tensor


def _bias(value: Any) -> Tensor | None:
    # A bare Parameter is a weight, never a bias.  Biases are read only from
    # projection modules (or an explicitly supplied bias Tensor by callers).
    tensor = getattr(value, "bias", None) if not isinstance(value, Tensor) else None
    return tensor if isinstance(tensor, Tensor) else None


def _expert_item(experts: nn.Module, expert_index: int) -> Any:
    if isinstance(experts, (nn.ModuleList, nn.Sequential, list, tuple)):
        return experts[expert_index]
    return expert_index


def extract_expert_projections(experts: nn.Module, expert_index: int) -> ExpertProjections:
    """Extract one Granite expert without guessing fused channel order.

    Granite-MoE stores all experts in ``gate_up_proj[E, 2I, H]`` and
    ``down_proj[E, H, I]``.  The model implementation explicitly chunks the
    first axis as ``gate, up``.  Separate gate/up ``nn.Linear`` modules are
    also accepted for test doubles and older checkpoints.
    """

    fused = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if isinstance(fused, Tensor) or isinstance(getattr(fused, "weight", None), Tensor):
        fused_weight = _weight(fused)
        if fused_weight.ndim == 3:
            fused_weight = fused_weight[int(expert_index)]
        if fused_weight.ndim != 2 or fused_weight.shape[0] % 2:
            raise UnsupportedFoundationArchitecture("fused gate_up projection is not [2I,H]")
        gate_weight, up_weight = fused_weight.chunk(2, dim=0)
        fused_bias = _bias(fused)
        if fused_bias is not None and fused_bias.ndim == 2:
            fused_bias = fused_bias[int(expert_index)]
        gate_bias, up_bias = fused_bias.chunk(2, dim=0) if fused_bias is not None else (None, None)
        down_weight = _weight(down)
        if down_weight.ndim == 3:
            down_weight = down_weight[int(expert_index)]
        down_bias = _bias(down)
        if down_bias is not None and down_bias.ndim == 2:
            down_bias = down_bias[int(expert_index)]
        result = ExpertProjections(
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            fused_order="gate_up",
        )
        result.validate()
        return result

    item = _expert_item(experts, expert_index)
    gate = getattr(item, "gate_proj", None)
    up = getattr(item, "up_proj", None)
    down_module = getattr(item, "down_proj", None)
    if gate is None or up is None or down_module is None:
        raise UnsupportedFoundationArchitecture(
            "expert must expose gate_up_proj or separate gate_proj/up_proj/down_proj"
        )
    result = ExpertProjections(
        gate_weight=_weight(gate),
        up_weight=_weight(up),
        down_weight=_weight(down_module),
        gate_bias=_bias(gate),
        up_bias=_bias(up),
        down_bias=_bias(down_module),
    )
    result.validate()
    return result


class CellProjection(nn.Module):
    """A trainable logical Cell, initialized as an exact parent slice."""

    def __init__(self, projections: ExpertProjections, start: int, end: int) -> None:
        super().__init__()
        if not 0 <= start < end <= projections.intermediate_size:
            raise ValueError("invalid Cell range")
        self.start, self.end = int(start), int(end)
        self.gate_weight = nn.Parameter(projections.gate_weight[start:end].detach().clone())
        self.up_weight = nn.Parameter(projections.up_weight[start:end].detach().clone())
        self.down_weight = nn.Parameter(projections.down_weight[:, start:end].detach().clone())
        if projections.gate_bias is not None:
            self.gate_bias = nn.Parameter(projections.gate_bias[start:end].detach().clone())
        else:
            self.register_parameter("gate_bias", None)
        if projections.up_bias is not None:
            self.up_bias = nn.Parameter(projections.up_bias[start:end].detach().clone())
        else:
            self.register_parameter("up_bias", None)
        if projections.down_bias is not None:
            self.down_bias = nn.Parameter(projections.down_bias.detach().clone())
        else:
            self.register_parameter("down_bias", None)

    def forward(self, hidden_states: Tensor, activation: Any = F.silu) -> Tensor:
        gate = F.linear(hidden_states, self.gate_weight, self.gate_bias)
        up = F.linear(hidden_states, self.up_weight, self.up_bias)
        return F.linear(activation(gate) * up, self.down_weight, None)


class CellularExpert(nn.Module):
    """Exact sum of contiguous Cell projections for one parent expert."""

    def __init__(
        self,
        projections: ExpertProjections,
        partition: CellPartition,
        activation: Any = F.silu,
    ) -> None:
        super().__init__()
        projections.validate()
        if partition.intermediate_size != projections.intermediate_size:
            raise ValueError("partition width does not match expert")
        self.partition = partition
        self.activation = activation
        self.cells = nn.ModuleList(
            CellProjection(projections, start, end) for start, end in partition.ranges()
        )
        if projections.down_bias is not None:
            # A down bias belongs to the expert, not to every Cell.  Add it once.
            self.down_bias = nn.Parameter(projections.down_bias.detach().clone())
        else:
            self.register_parameter("down_bias", None)

    @classmethod
    def from_experts(
        cls, experts: nn.Module, expert_index: int, partition: CellPartition, activation: Any = F.silu
    ) -> "CellularExpert":
        return cls(extract_expert_projections(experts, expert_index), partition, activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = None
        for cell in self.cells:
            value = cell(hidden_states, self.activation)
            output = value if output is None else output + value
        if self.down_bias is not None:
            output = output + self.down_bias
        return output

    def cell_parameters(self, indices: Iterable[int] | None = None) -> Iterator[nn.Parameter]:
        selected = range(len(self.cells)) if indices is None else indices
        for index in selected:
            yield from self.cells[int(index)].parameters()

    def fork(self, indices: Iterable[int]) -> "CellularExpert":
        """Return a storage-independent copy; caller decides which cells train."""
        copy = CellularExpert.from_state(self)
        selected = {int(index) for index in indices}
        for index, cell in enumerate(copy.cells):
            for parameter in cell.parameters():
                parameter.requires_grad_(index in selected)
        copy.down_bias.requires_grad_(False) if copy.down_bias is not None else None
        return copy

    @classmethod
    def from_state(cls, source: "CellularExpert") -> "CellularExpert":
        # Build via a private constructor-free clone so no parent tensor storage is shared.
        clone = object.__new__(cls)
        nn.Module.__init__(clone)
        clone.partition = source.partition
        clone.activation = source.activation
        clone.cells = nn.ModuleList()
        for source_cell in source.cells:
            cell = object.__new__(CellProjection)
            nn.Module.__init__(cell)
            cell.start, cell.end = source_cell.start, source_cell.end
            for name, value in (
                ("gate_weight", source_cell.gate_weight),
                ("up_weight", source_cell.up_weight),
                ("down_weight", source_cell.down_weight),
                ("gate_bias", source_cell.gate_bias),
                ("up_bias", source_cell.up_bias),
                ("down_bias", source_cell.down_bias),
            ):
                cell.register_parameter(name, nn.Parameter(value.detach().clone(), requires_grad=value.requires_grad) if value is not None else None)
            clone.cells.append(cell)
        clone.register_parameter(
            "down_bias",
            nn.Parameter(source.down_bias.detach().clone(), requires_grad=False)
            if source.down_bias is not None
            else None,
        )
        return clone


class CellularExperts(nn.Module):
    """Drop-in replacement for GraniteMoeExperts with unchanged dispatch."""

    def __init__(self, source: nn.Module, partition: CellPartition, activation: Any = F.silu) -> None:
        super().__init__()
        self.num_experts = int(getattr(source, "num_experts"))
        self.hidden_dim = int(getattr(source, "hidden_dim", getattr(source, "hidden_size", 0)))
        self.intermediate_dim = int(
            getattr(source, "intermediate_dim", getattr(source, "intermediate_size", partition.intermediate_size))
        )
        self.partition = partition
        self.cells = nn.ModuleList(
            CellularExpert.from_experts(source, index, partition, activation)
            for index in range(self.num_experts)
        )
        self.source_class = type(source).__name__

    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        # This is intentionally the same parent-expert dispatch contract: no
        # Cell softmax, no child router, and no change to routing coefficients.
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0])
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current = self.cells[expert_idx](current_state)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


def _activation_for(source: nn.Module) -> Any:
    return getattr(source, "act_fn", F.silu)


def patch_moe_block(block_sparse_moe: nn.Module, partition: CellPartition | None = None) -> CellularExperts:
    """Replace only the expert compute module and preserve the parent router."""
    source = getattr(block_sparse_moe, "experts", None)
    router = getattr(block_sparse_moe, "router", None)
    if source is None or router is None:
        raise UnsupportedFoundationArchitecture("target module must expose router and experts")
    intermediate = int(getattr(source, "intermediate_dim", getattr(source, "intermediate_size", 0)))
    partition = partition or CellPartition(intermediate, 4)
    cellular = CellularExperts(source, partition, _activation_for(source))
    block_sparse_moe.experts = cellular
    if block_sparse_moe.router is not router:
        raise AssertionError("parent router storage changed during cellularization")
    return cellular


@dataclass(frozen=True)
class GraniteArchitectureInspector:
    """Resolved target architecture and its fail-closed invariants."""

    target_path: str
    target_layer: int
    hidden_size: int
    intermediate_size: int
    local_experts: int
    experts_per_token: int
    fused_projection: bool
    fused_order: str
    cells: int = 4

    @property
    def partition(self) -> CellPartition:
        return CellPartition(self.intermediate_size, self.cells)

    @property
    def logical_cells(self) -> int:
        return self.local_experts * self.cells

    @property
    def decoder_layer_path(self) -> str:
        suffix = ".block_sparse_moe"
        return self.target_path[:-len(suffix)] if self.target_path.endswith(suffix) else self.target_path

    @classmethod
    def inspect(cls, model: nn.Module, require_granite: bool = True) -> "GraniteArchitectureInspector":
        config = getattr(model, "config", None)
        local = int(getattr(config, "num_local_experts", getattr(config, "num_experts", 0)))
        per_token = int(getattr(config, "num_experts_per_tok", getattr(config, "num_experts_per_token", 0)))
        hidden = int(getattr(config, "hidden_size", 0))
        intermediate = int(getattr(config, "intermediate_size", 0))
        candidates: list[tuple[str, int, nn.Module, nn.Module]] = []
        for name, module in model.named_modules():
            experts = getattr(module, "experts", None)
            router = getattr(module, "router", None)
            if experts is None or router is None:
                continue
            try:
                count = int(getattr(experts, "num_experts"))
                projection = extract_expert_projections(experts, 0)
            except (AttributeError, TypeError, UnsupportedFoundationArchitecture):
                continue
            layer_numbers = [int(item) for item in re.findall(r"layers\.(\d+)", name)]
            layer = layer_numbers[-1] if layer_numbers else len(candidates)
            candidates.append((name, layer, module, experts))
            if not local:
                local = count
            if not hidden:
                hidden = projection.hidden_size
            if not intermediate:
                intermediate = projection.intermediate_size
        if not candidates:
            raise UnsupportedFoundationArchitecture("no router-preserving MoE block found")
        target_path, target_layer, target, experts = max(candidates, key=lambda row: row[1])
        projection = extract_expert_projections(experts, 0)
        fused = isinstance(getattr(experts, "gate_up_proj", None), Tensor)
        invariant_errors: list[str] = []
        if require_granite:
            if local != 32:
                invariant_errors.append(f"local experts={local}, expected 32")
            if projection.intermediate_size != 512:
                invariant_errors.append(
                    f"intermediate size={projection.intermediate_size}, expected 512"
                )
            if projection.hidden_size != 1024:
                invariant_errors.append(f"hidden size={projection.hidden_size}, expected 1024")
        if projection.fused_order != "gate_up":
            invariant_errors.append("fused gate/up order is not the verified gate,up order")
        if invariant_errors:
            raise UnsupportedFoundationArchitecture("; ".join(invariant_errors))
        return cls(
            target_path=target_path,
            target_layer=target_layer,
            hidden_size=projection.hidden_size,
            intermediate_size=projection.intermediate_size,
            local_experts=local,
            experts_per_token=per_token,
            fused_projection=fused,
            fused_order=projection.fused_order,
        )
