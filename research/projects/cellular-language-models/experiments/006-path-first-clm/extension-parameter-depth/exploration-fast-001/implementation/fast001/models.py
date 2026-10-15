"""Reuse hard PDCLM; remove learnable STOP rows rather than adding idle padding."""

import torch
from torch import nn
from torch.nn import functional as F
from exp006.model import initialize
from exp006.baseline import MoELM
from pd001.models import PDCLM
from pd001.router import Router
from .config import model_config


class FixedRouter(Router):
    def __init__(self, c):
        super().__init__(c)
        self.actions = c["cell_count"] + 1
        self.scorer[-1] = nn.Linear(c["router_hidden"], self.actions)
        self.source = nn.Embedding(self.actions + 1, c["edge_rank"])
        self.target = nn.Embedding(self.actions, c["edge_rank"])

    def forward(self, h, previous, t, edge_mode="normal", permutation=None):
        # Parent executor's START is one beyond its disabled STOP sentinel.
        previous = torch.where(previous == self.actions + 1, self.actions, previous)
        scores, value = super().forward(h, previous, t, edge_mode, permutation)
        # Compatibility slot: constant -inf, no parameters, never an executed action.
        return F.pad(scores, (0, 1), value=float("-inf")), value


class FixedCLM(PDCLM):
    def __init__(self, c, arm, seed):
        super().__init__(c, arm, seed)
        self.controller = FixedRouter(c)
        initialize(self, seed)


def make_model(c, arm, seed=None):
    seed = c["seed"] if seed is None else seed
    m = model_config(c, arm)
    return MoELM(m, seed) if arm == "M0-TINY" else FixedCLM(m, arm, seed)
