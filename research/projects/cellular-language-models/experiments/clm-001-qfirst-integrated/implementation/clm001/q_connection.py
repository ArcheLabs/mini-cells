"""One all-action cost head; no separate Actor. Detached value baseline on both arms."""

import torch
from torch import nn


class Connection(nn.Module):
    def __init__(self, c, actions):
        super().__init__()
        m = c["model"]
        d = m["width"]
        k = m["step_width"]
        self.step = nn.Embedding(m["budget"] + 1, k)
        self.source = nn.Embedding(actions + 1, m["edge_rank"])
        self.target = nn.Embedding(actions, m["edge_rank"])
        self.scorer = nn.Sequential(
            nn.Linear(d + k + actions + 2, m["router_hidden"]),
            nn.Tanh(),
            nn.Linear(m["router_hidden"], actions),
        )
        self.edge_bias = nn.Parameter(torch.zeros(actions + 1, actions))
        self.critic = nn.Sequential(nn.Linear(d + k, 32), nn.Tanh(), nn.Linear(32, 1))
        self.budget = m["budget"]

    def forward(
        self, h, previous, depth, legal, registry_version=0, *, detach_state=True, edge_zero=False
    ):
        features = h.detach() if detach_state else h
        step = self.step.weight[depth].expand(len(h), -1)
        mask = legal.to(h.dtype).expand(len(h), -1)
        rest = h.new_full((len(h), 1), (self.budget - depth) / self.budget)
        version = h.new_full((len(h), 1), registry_version / 4)
        q = self.scorer(torch.cat((features, step, mask, rest, version), -1))
        if not edge_zero:
            q = (
                q
                + self.source(previous) @ self.target.weight.T * self.source.embedding_dim**-0.5
                + self.edge_bias[previous]
            )
        value = self.critic(torch.cat((h.detach(), step.detach()), -1)).squeeze(-1)
        return q, value
