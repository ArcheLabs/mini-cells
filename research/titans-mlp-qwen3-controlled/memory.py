"""Phase 1 controlled nonlinear Titans memory."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call, grad


@dataclass(frozen=True)
class State:
    weights: dict
    momentum: dict
    updates: int = 0


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(512, 2048, bias=False)
        self.project = nn.Linear(2048, 512, bias=False)
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        return x + self.norm(self.project(nn.functional.gelu(self.expand(x))))

    def create(self):
        weights = {name: p.clone() for name, p in self.named_parameters()}
        return State(weights, {name: torch.zeros_like(p) for name, p in weights.items()})

    def read(self, state, query):
        return functional_call(self, state.weights, (query,))

    def event(self, state, key, value, query, *, valid=True, eta=.5, theta=.02, alpha=.0001):
        readout = self.read(state, query)
        if not valid:
            return readout, state

        def loss(weights):
            return (functional_call(self, weights, (key,)) - value).square().sum()

        gradients = grad(loss)(state.weights)
        norm = torch.stack([g.norm() for g in gradients.values()]).norm()
        scale = norm.clamp_min(1.).reciprocal()
        momentum = {name: eta * state.momentum[name] - theta * scale * gradient
                    for name, gradient in gradients.items()}
        weights = {name: (1 - alpha) * p + momentum[name] for name, p in state.weights.items()}
        return readout, State(weights, momentum, state.updates + 1)
