from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


RoutingMode = Literal[
    "dense",
    "soft_program",
    "soft",
    "hard_program",
    "soft_cell_hard_program",
    "hard",
]
ExecutionBackend = Literal["masked_dense", "sparse_dispatch"]


@dataclass(frozen=True)
class CLMRoutingConfig:
    num_programs: int = 8
    receptor_dim: int = 32
    routing_mode: RoutingMode = "dense"
    execution_backend: ExecutionBackend = "masked_dense"
    program_top_k: int | None = None
    cell_threshold: float = 0.5
    phenotype_dim: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.num_programs <= 16:
            raise ValueError("num_programs must be in [1, 16]")
        if self.receptor_dim < 1:
            raise ValueError("receptor_dim must be positive")
        if self.routing_mode not in (
            "dense",
            "soft_program",
            "soft",
            "hard_program",
            "soft_cell_hard_program",
            "hard",
        ):
            raise ValueError(f"unknown routing_mode: {self.routing_mode}")
        if self.execution_backend not in ("masked_dense", "sparse_dispatch"):
            raise ValueError(f"unknown execution_backend: {self.execution_backend}")
        if self.program_top_k is not None and not 1 <= self.program_top_k <= self.num_programs:
            raise ValueError("program_top_k must be in [1, num_programs]")
        if not 0.0 <= self.cell_threshold <= 1.0:
            raise ValueError("cell_threshold must be in [0, 1]")
        if self.phenotype_dim < 0:
            raise ValueError("phenotype_dim must be non-negative")


def straight_through_topk(probabilities: torch.Tensor, k: int) -> torch.Tensor:
    indices = probabilities.topk(k, dim=-1).indices
    hard = torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)
    return hard + probabilities - probabilities.detach()


def straight_through_threshold(
    probabilities: torch.Tensor, threshold: float
) -> torch.Tensor:
    hard = (probabilities >= threshold).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()
