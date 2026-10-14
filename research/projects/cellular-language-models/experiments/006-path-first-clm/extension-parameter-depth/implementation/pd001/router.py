"""State policy plus optional learned predecessor prior, including real STOP."""

import torch
from torch import nn


class Router(nn.Module):
    def __init__(self, c, state_only=False):
        super().__init__()
        self.actions = c["cell_count"] + 2
        self.step = nn.Embedding(c["step_capacity"], c["step_width"])
        self.scorer = nn.Sequential(
            nn.Linear(c["width"] + c["step_width"], c["router_hidden"]),
            nn.Tanh(),
            nn.Linear(c["router_hidden"], self.actions),
        )
        if not state_only:
            self.source = nn.Embedding(self.actions + 1, c["edge_rank"])
            self.target = nn.Embedding(self.actions, c["edge_rank"])
        self.critic = nn.Sequential(nn.Linear(c["width"] + c["step_width"], 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, h, previous, t, edge_mode="normal", permutation=None):
        step = self.step.weight[t].expand(h.shape[0], -1)
        policy_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        scores = self.scorer(torch.cat((h, step), -1)).to(policy_dtype)
        if hasattr(self, "source") and edge_mode != "zero":
            ids = previous
            if edge_mode == "shuffle":
                if permutation is None or sorted(permutation.cpu().tolist()) != list(range(self.actions + 1)):
                    raise ValueError("a full predecessor row permutation is required")
                ids = permutation[previous]
            elif edge_mode != "normal":
                raise ValueError("unknown edge mode")
            scores = (
                scores
                + (self.source(ids).to(policy_dtype) @ self.target.weight.to(policy_dtype).T)
                * self.source.embedding_dim**-0.5
            )
        value = (
            self.critic(torch.cat((h.detach(), step.detach()), -1)).squeeze(-1)
            if self.training
            else h.new_zeros(h.shape[0])
        )
        return scores, value
