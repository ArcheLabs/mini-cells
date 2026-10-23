"""Six legal next nodes, all self edges, no topology mutation or target features."""

import torch
from torch import nn
from torch.nn import functional as F


class GraphConnection(nn.Module):
    def __init__(self, width):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(width + 7 + 8, 6)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, h, t, z, previous):
        pred = F.one_hot(previous, 7).to(h)
        step = F.one_hot(torch.full_like(previous, t), 8).to(h)
        delta = self.linear(torch.cat([h.detach(), pred, step], -1))
        return z.detach() + delta, delta


def random_action(tokens, step, seed):
    # Input token and absolute position only; no labels, hidden state or preceding choices.
    pos = (
        torch.arange(tokens.numel(), device=tokens.device).reshape(tokens.shape) % tokens.shape[-1]
    )
    v = (tokens.long() + 1) * 1103515245 + (pos + 1) * 12345 + (seed + step * 104729) * 2654435761
    v = (v ^ (v >> 16)) * 2246822519
    return ((v ^ (v >> 13)) & 0x7FFFFFFFFFFFFFFF).remainder(6).reshape(-1)
