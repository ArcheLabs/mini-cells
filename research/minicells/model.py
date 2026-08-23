from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EchoModel(nn.Module):
    def __init__(self, *, vocab_size: int, num_cells: int = 64, embedding_dim: int = 8,
                 hidden_dim: int = 16, radius: int = 2, iterations: int = 4,
                 mlp_width: int = 32, residual_scale: float = 1.0, **_: object) -> None:
        super().__init__()
        self.vocab_size, self.num_cells = vocab_size, num_cells
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.radius, self.iterations, self.mlp_width = radius, iterations, mlp_width
        self.residual_scale = residual_scale
        update_input = (2 * radius + 1) * hidden_dim + embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.update_in = nn.Linear(update_input, mlp_width)
        self.update_out = nn.Linear(mlp_width, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def initial_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*input_ids.shape, self.hidden_dim), device=input_ids.device,
                           dtype=self.embedding.weight.dtype)

    def _neighborhood(self, state: torch.Tensor) -> torch.Tensor:
        padded = F.pad(state, (0, 0, self.radius, self.radius))
        views = [padded[:, offset:offset + self.num_cells] for offset in range(2 * self.radius + 1)]
        return torch.cat(views, dim=-1)

    def forward(self, input_ids: torch.Tensor, return_state: bool = False):
        if input_ids.ndim != 2 or input_ids.shape[1] != self.num_cells:
            raise ValueError(f"input_ids must have shape [batch, {self.num_cells}]")
        embedded = self.embedding(input_ids)
        state = self.initial_state(input_ids)
        for _ in range(self.iterations):
            update_input = torch.cat((self._neighborhood(state), embedded), dim=-1)
            delta = self.update_out(F.relu(self.update_in(update_input)))
            state = torch.clamp(state + self.residual_scale * delta, -1.0, 1.0)
        logits = self.output(state)
        return (logits, state) if return_state else logits
