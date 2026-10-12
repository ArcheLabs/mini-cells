"""Residual Cell wrapper using the existing HybridCLM FFN implementation."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from experiments.common.models import FeedForward


class ResidualCell(nn.Module):
    def __init__(self, width: int, hidden: int, residual_scale: float = 0.25):
        super().__init__()
        self.compute = FeedForward(width, hidden)
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

    def forward(self, state: Tensor, normalized_state: Tensor) -> Tensor:
        return self.compute(normalized_state) * self.residual_scale.to(state.dtype)
