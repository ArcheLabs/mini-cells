"""State- and predecessor-aware routing with optional hard candidate constraints."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RouteScores:
    candidate_ids: Tensor
    logits: Tensor


class StateAwareRouter(nn.Module):
    def __init__(self, width: int, cell_count: int, router_width: int = 32):
        super().__init__()
        self.width = int(width)
        self.cell_count = int(cell_count)
        self.router_width = int(router_width)
        # One scorer is shared by all sources in a block; source/target biases are
        # factorized, avoiding an N-by-N-by-width controller.
        self.source_address = nn.Embedding(cell_count + 1, router_width)
        self.target_address = nn.Embedding(cell_count, router_width)
        self.edge_source = nn.Embedding(cell_count + 1, router_width)
        self.edge_target = nn.Embedding(cell_count, router_width)
        self.query = nn.Linear(width + router_width, router_width, bias=False)

    def forward(self, state: Tensor, previous: Tensor, adjacency: Tensor | None = None) -> RouteScores:
        if state.ndim != 2 or previous.shape != (state.shape[0],):
            raise ValueError("state must be (tokens, width) and previous must contain one id per token")
        if torch.any(previous < 0) or torch.any(previous > self.cell_count):
            raise ValueError("previous Cell id is outside the registered address space")
        if adjacency is None:
            candidate_ids = torch.arange(self.cell_count, device=state.device)
            candidate_ids = candidate_ids.expand(state.shape[0], -1)
        else:
            if adjacency.shape != (self.cell_count + 1, self.cell_count):
                raise ValueError("adjacency has the wrong shape")
            if adjacency.dtype != torch.bool or not torch.all(adjacency.sum(-1) == adjacency.sum(-1)[0]):
                raise ValueError("adjacency must be boolean with fixed out-degree")
            candidate_ids = torch.where(adjacency.to(state.device)[previous])[1]
            # torch.where flattens rows; reshape is safe because all rows have equal degree.
            candidate_ids = candidate_ids.reshape(state.shape[0], -1)
        source_address = self.source_address(previous)
        query = self.query(torch.cat((state, source_address.to(state.dtype)), dim=-1)).float()
        keys = self.target_address(candidate_ids).float()
        edge_u = self.edge_source(previous).float()
        edge_v = self.edge_target(candidate_ids).float()
        scale = self.router_width ** -0.5
        logits = (keys * query[:, None, :]).sum(-1) * scale
        logits = logits + (edge_u[:, None, :] * edge_v).sum(-1) * scale
        return RouteScores(candidate_ids, logits)


def hard_with_top2_backup(hard_value: Tensor, backup_values: Tensor, top2_logits: Tensor) -> Tensor:
    """Hard forward value with the selected top-2 soft surrogate gradient."""
    if backup_values.ndim != hard_value.ndim + 1 or backup_values.shape[1] != 2:
        raise ValueError("backup_values must have a leading top-2 axis")
    weights = top2_logits.float().softmax(-1).to(backup_values.dtype)
    shape = (weights.shape[0], 2) + (1,) * (backup_values.ndim - 2)
    surrogate = (backup_values * weights.reshape(shape)).sum(1)
    return surrogate + (hard_value - surrogate).detach()
