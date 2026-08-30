"""Fine-grained developmental tissues for MiniCells.

A full MoE FFN expert is treated as a tissue rather than one cell. The dense
FFN is decomposed exactly along its hidden dimension into smaller micro-cells
whose summed output is identical to the original expert. This makes cell
granularity independently controllable while keeping initial parameters,
active compute, and function fixed.

Micro-cells also carry non-functional developmental state: age, stress and
plasticity. Mitosis is function-preserving: the child clones the parent and
both outgoing projections are halved, so parent + child exactly reproduce the
pre-division contribution at age zero. The juvenile child can then receive a
higher learning coefficient and differentiate during later training.

Version 0 deliberately uses a fixed radius-one local chain inside each tissue.
More general learned tissue morphology is deferred until granularity itself is
validated experimentally.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StressWeights:
    usage: float = 0.15
    residual_loss: float = 0.25
    novelty: float = 0.20
    gradient_conflict: float = 0.40
    neighbor_relief: float = 0.25

    def __post_init__(self) -> None:
        positive = self.usage + self.residual_loss + self.novelty + self.gradient_conflict
        if not math.isclose(positive, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("positive stress weights must sum to 1")
        if min(
            self.usage,
            self.residual_loss,
            self.novelty,
            self.gradient_conflict,
            self.neighbor_relief,
        ) < 0:
            raise ValueError("stress weights must be non-negative")


@dataclass(frozen=True)
class StressObservation:
    usage: float
    residual_loss: float
    novelty: float
    gradient_conflict: float
    neighbor_capacity: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be normalized to [0, 1]")


@dataclass(frozen=True)
class TissueConfig:
    cells_per_tissue: int = 4
    mature_plasticity: float = 1.0
    juvenile_plasticity: float = 4.0
    plasticity_half_life_tokens: int = 2_000_000
    stress_ema_decay: float = 0.9
    mitosis_threshold: float = 0.75
    minimum_overload_steps: int = 3
    stress_weights: StressWeights = StressWeights()

    def __post_init__(self) -> None:
        if self.cells_per_tissue < 1:
            raise ValueError("cells_per_tissue must be positive")
        if self.mature_plasticity <= 0 or self.juvenile_plasticity <= 0:
            raise ValueError("plasticity coefficients must be positive")
        if self.juvenile_plasticity < self.mature_plasticity:
            raise ValueError("juvenile plasticity must be >= mature plasticity")
        if self.plasticity_half_life_tokens <= 0:
            raise ValueError("plasticity half-life must be positive")
        if not 0.0 <= self.stress_ema_decay < 1.0:
            raise ValueError("stress_ema_decay must be in [0, 1)")
        if not 0.0 <= self.mitosis_threshold <= 1.0:
            raise ValueError("mitosis_threshold must be in [0, 1]")
        if self.minimum_overload_steps < 1:
            raise ValueError("minimum_overload_steps must be positive")


@dataclass(frozen=True)
class TissueTelemetry:
    cell_contribution_rms: tuple[float, ...]
    cell_stress: tuple[float, ...]
    cell_plasticity: tuple[float, ...]
    cell_age_tokens: tuple[int, ...]


class MicroCell(nn.Module):
    """One fine-grained FFN contribution plus persistent developmental state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        activation: nn.Module,
        mature_plasticity: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("micro-cell hidden_dim must be positive")
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = copy.deepcopy(activation)
        self.out_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.register_buffer("age_tokens", torch.zeros((), dtype=torch.long))
        self.register_buffer("stress", torch.zeros((), dtype=torch.float32))
        self.register_buffer("plasticity", torch.tensor(float(mature_plasticity)))
        self.register_buffer("birth_plasticity", torch.tensor(float(mature_plasticity)))
        self.register_buffer("overload_steps", torch.zeros((), dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.in_proj(inputs)))

    @torch.no_grad()
    def reset_as_juvenile(self, plasticity: float) -> None:
        self.age_tokens.zero_()
        self.stress.zero_()
        self.overload_steps.zero_()
        self.birth_plasticity.fill_(float(plasticity))
        self.plasticity.fill_(float(plasticity))

    @torch.no_grad()
    def advance_age(
        self,
        tokens: int,
        *,
        mature_plasticity: float,
        half_life_tokens: int,
    ) -> None:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        self.age_tokens.add_(int(tokens))
        age = float(self.age_tokens.item())
        decay = 0.5 ** (age / float(half_life_tokens))
        value = (
            mature_plasticity
            + (float(self.birth_plasticity.item()) - mature_plasticity) * decay
        )
        self.plasticity.fill_(value)


class TissueFFN(nn.Module):
    """An exact additive decomposition of one dense FFN expert into micro-cells."""

    FORMAT = "minicells.developmental-tissue.v0"

    def __init__(
        self,
        cells: Iterable[MicroCell],
        output_bias: torch.Tensor | None,
        *,
        config: TissueConfig,
    ) -> None:
        super().__init__()
        cells = list(cells)
        if not cells:
            raise ValueError("a tissue requires at least one cell")
        self.cells = nn.ModuleList(cells)
        if output_bias is None:
            self.register_parameter("output_bias", None)
        else:
            self.output_bias = nn.Parameter(output_bias.detach().clone())
        self.config = config
        self.division_history: list[dict[str, object]] = []
        self.register_buffer("adjacency", self._initial_adjacency(len(cells)))

    @classmethod
    def from_dense_ffn(
        cls,
        dense_ffn: nn.Module,
        *,
        config: TissueConfig | None = None,
    ) -> TissueFFN:
        config = config or TissueConfig()
        if not isinstance(dense_ffn, nn.Sequential) or len(dense_ffn) != 3:
            raise TypeError("expected Sequential(Linear, elementwise activation, Linear)")
        first, activation, second = dense_ffn
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise TypeError("expected Sequential(Linear, elementwise activation, Linear)")
        if not isinstance(activation, (nn.GELU, nn.ReLU, nn.SiLU)):
            raise TypeError("developmental tissue currently supports GELU/ReLU/SiLU FFNs")
        if second.in_features != first.out_features:
            raise ValueError("FFN hidden dimensions do not match")
        if config.cells_per_tissue > first.out_features:
            raise ValueError("cells_per_tissue cannot exceed FFN hidden width")

        widths = _balanced_partition(first.out_features, config.cells_per_tissue)
        cells: list[MicroCell] = []
        offset = 0
        for width in widths:
            cell = MicroCell(
                first.in_features,
                width,
                second.out_features,
                activation=activation,
                mature_plasticity=config.mature_plasticity,
            ).to(device=first.weight.device, dtype=first.weight.dtype)
            with torch.no_grad():
                cell.in_proj.weight.copy_(first.weight[offset : offset + width])
                if first.bias is not None:
                    cell.in_proj.bias.copy_(first.bias[offset : offset + width])
                else:
                    cell.in_proj.bias.zero_()
                cell.out_proj.weight.copy_(second.weight[:, offset : offset + width])
            cells.append(cell)
            offset += width

        tissue = cls(cells, second.bias, config=config)
        return tissue.to(device=first.weight.device, dtype=first.weight.dtype)

    @staticmethod
    def _initial_adjacency(count: int) -> torch.Tensor:
        adjacency = torch.zeros((count, count), dtype=torch.bool)
        for index in range(count - 1):
            adjacency[index, index + 1] = True
            adjacency[index + 1, index] = True
        return adjacency

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_telemetry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TissueTelemetry]:
        contributions = [cell(inputs) for cell in self.cells]
        output = torch.stack(contributions, dim=0).sum(dim=0)
        if self.output_bias is not None:
            output = output + self.output_bias
        if not return_telemetry:
            return output
        rms = tuple(
            float(value.detach().float().square().mean().sqrt().item())
            for value in contributions
        )
        telemetry = TissueTelemetry(
            rms,
            tuple(float(cell.stress.item()) for cell in self.cells),
            tuple(float(cell.plasticity.item()) for cell in self.cells),
            tuple(int(cell.age_tokens.item()) for cell in self.cells),
        )
        return output, telemetry

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def cell_hidden_widths(self) -> tuple[int, ...]:
        return tuple(cell.in_proj.out_features for cell in self.cells)

    def neighbors(self, cell_index: int) -> tuple[int, ...]:
        self._validate_cell_index(cell_index)
        return tuple(self.adjacency[cell_index].nonzero(as_tuple=False).flatten().tolist())

    def neighbor_capacity(self, cell_index: int, capacities: Sequence[float]) -> float:
        if len(capacities) != self.cell_count:
            raise ValueError("one capacity value is required per cell")
        neighbors = self.neighbors(cell_index)
        if not neighbors:
            return 0.0
        values = [float(capacities[index]) for index in neighbors]
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("capacities must be normalized to [0, 1]")
        return sum(values) / len(values)

    def instantaneous_stress(self, observation: StressObservation) -> float:
        weights = self.config.stress_weights
        pressure = (
            weights.usage * observation.usage
            + weights.residual_loss * observation.residual_loss
            + weights.novelty * observation.novelty
            + weights.gradient_conflict * observation.gradient_conflict
            - weights.neighbor_relief * observation.neighbor_capacity
        )
        return max(0.0, min(1.0, pressure))

    @torch.no_grad()
    def observe_stress(self, cell_index: int, observation: StressObservation) -> float:
        self._validate_cell_index(cell_index)
        cell = self.cells[cell_index]
        instant = self.instantaneous_stress(observation)
        decay = self.config.stress_ema_decay
        cell.stress.mul_(decay).add_((1.0 - decay) * instant)
        if float(cell.stress.item()) >= self.config.mitosis_threshold:
            cell.overload_steps.add_(1)
        else:
            cell.overload_steps.zero_()
        return float(cell.stress.item())

    def should_divide(self, cell_index: int) -> bool:
        self._validate_cell_index(cell_index)
        cell = self.cells[cell_index]
        return (
            float(cell.stress.item()) >= self.config.mitosis_threshold
            and int(cell.overload_steps.item()) >= self.config.minimum_overload_steps
        )

    @torch.no_grad()
    def divide_cell(self, parent_index: int) -> dict[str, object]:
        """Function-preserving fission followed by asymmetric developmental state."""
        self._validate_cell_index(parent_index)
        parent = self.cells[parent_index]
        child = copy.deepcopy(parent)
        parent.out_proj.weight.mul_(0.5)
        child.out_proj.weight.mul_(0.5)
        child.reset_as_juvenile(self.config.juvenile_plasticity)
        child_index = parent_index + 1
        self.cells.insert(child_index, child)
        self._insert_child_adjacency(parent_index, child_index)
        event = {
            "type": "cell_division",
            "parent_index": parent_index,
            "child_index": child_index,
            "cell_count": self.cell_count,
            "parent_plasticity": float(parent.plasticity.item()),
            "child_plasticity": float(child.plasticity.item()),
            "function_preserving_rule": "clone_parent_and_halve_both_outgoing_projections",
        }
        self.division_history.append(event)
        return event

    @torch.no_grad()
    def advance_age(self, tokens: int) -> None:
        for cell in self.cells:
            cell.advance_age(
                tokens,
                mature_plasticity=self.config.mature_plasticity,
                half_life_tokens=self.config.plasticity_half_life_tokens,
            )

    def optimizer_param_groups(self, base_lr: float) -> list[dict[str, object]]:
        if base_lr <= 0:
            raise ValueError("base_lr must be positive")
        groups: list[dict[str, object]] = []
        for index, cell in enumerate(self.cells):
            scale = float(cell.plasticity.item())
            groups.append(
                {
                    "params": list(cell.parameters()),
                    "lr": base_lr * scale,
                    "cell_index": index,
                    "plasticity": scale,
                }
            )
        if self.output_bias is not None:
            groups.append(
                {
                    "params": [self.output_bias],
                    "lr": base_lr,
                    "cell_index": None,
                    "plasticity": 1.0,
                }
            )
        return groups

    def _validate_cell_index(self, cell_index: int) -> None:
        if not 0 <= cell_index < self.cell_count:
            raise IndexError(f"cell index out of range: {cell_index}")

    @torch.no_grad()
    def _insert_child_adjacency(self, parent_index: int, child_index: int) -> None:
        old = self.adjacency
        old_count = old.shape[0]
        new = torch.zeros(
            (old_count + 1, old_count + 1),
            dtype=torch.bool,
            device=old.device,
        )

        def shifted(index: int) -> int:
            return index if index <= parent_index else index + 1

        for left in range(old_count):
            for right in range(old_count):
                if bool(old[left, right]):
                    new[shifted(left), shifted(right)] = True
        old_neighbors = old[parent_index].nonzero(as_tuple=False).flatten().tolist()
        new[parent_index, child_index] = True
        new[child_index, parent_index] = True
        for neighbor in old_neighbors:
            mapped = shifted(int(neighbor))
            new[child_index, mapped] = True
            new[mapped, child_index] = True
        self.adjacency = new


def _balanced_partition(total: int, parts: int) -> tuple[int, ...]:
    if parts < 1 or parts > total:
        raise ValueError("parts must be in [1, total]")
    base, remainder = divmod(total, parts)
    return tuple(base + (1 if index < remainder else 0) for index in range(parts))


def count_module_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def convert_model_experts_to_tissues(
    model: nn.Module,
    *,
    config: TissueConfig | None = None,
    inplace: bool = False,
) -> nn.Module:
    """Replace current FFN experts with exact fine-grained tissues.

    CLM-0.1 ModuleList banks and CLM-0.3 ModuleDict banks are both supported.
    Existing root/hierarchical routing is deliberately unchanged: in v0 the
    router selects a tissue, while micro-cells are the units inside it.
    """
    config = config or TissueConfig()
    target = model if inplace else copy.deepcopy(model)
    stages = getattr(target, "stages", None)
    if stages is None:
        raise TypeError("model does not expose MiniCells stages")
    replaced = 0
    for stage in stages:
        bank = getattr(stage, "program_bank", None)
        if bank is None or not hasattr(bank, "experts"):
            raise TypeError("stage does not expose a program_bank.experts collection")
        experts = bank.experts
        if isinstance(experts, nn.ModuleList):
            bank.experts = nn.ModuleList(
                [
                    expert
                    if isinstance(expert, TissueFFN)
                    else TissueFFN.from_dense_ffn(expert, config=config)
                    for expert in experts
                ]
            )
            replaced += len(bank.experts)
        elif isinstance(experts, nn.ModuleDict):
            bank.experts = nn.ModuleDict(
                {
                    expert_id: (
                        expert
                        if isinstance(expert, TissueFFN)
                        else TissueFFN.from_dense_ffn(expert, config=config)
                    )
                    for expert_id, expert in experts.items()
                }
            )
            replaced += len(bank.experts)
        else:
            raise TypeError("unsupported expert collection; expected ModuleList or ModuleDict")

    provenance = dict(getattr(target, "provenance", {}))
    provenance["developmental_tissue"] = {
        "format": TissueFFN.FORMAT,
        "cells_per_tissue": config.cells_per_tissue,
        "expert_to_tissue": "exact_hidden_dimension_partition",
        "routing_level": "tissue",
        "cell_level": "micro_ffn_contribution",
        "neighborhood": "fixed_radius_one_chain",
        "replaced_experts": replaced,
    }
    target.provenance = provenance
    return target
