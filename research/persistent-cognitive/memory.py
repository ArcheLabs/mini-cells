"""Independent per-KV-head memory with tokenwise causal transitions."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Shard:
    fast: tuple
    momentum: tuple
    address: torch.Tensor
    step: int


class NeuralMemory(nn.Module):
    def __init__(self, width=256, expansion=4, ridge=.01):
        super().__init__()
        if width < 1 or expansion < 1 or ridge <= 0:
            raise ValueError('Invalid memory geometry')
        self.ridge = ridge
        self.initial = nn.ParameterList([
            nn.Parameter(torch.empty(width * expansion, width)),
            nn.Parameter(torch.empty(width, width * expansion)),
        ])
        for weight in self.initial:
            nn.init.xavier_uniform_(weight)
        self.controls = nn.Linear(width, 4)

    def create(self):
        fast = tuple(weight.clone() for weight in self.initial)
        return Shard(fast, tuple(torch.zeros_like(w) for w in fast),
                     fast[0].new_zeros((fast[0].shape[1],) * 2), 0)

    @staticmethod
    def predict(x, fast):
        return x + F.layer_norm(F.linear(F.gelu(F.linear(x, fast[0])), fast[1]),
                                (x.shape[-1],))

    def read(self, query, state):
        eye = torch.eye(state.address.shape[0], device=query.device, dtype=query.dtype)
        projected = torch.linalg.solve(state.address + self.ridge * eye,
                                       (query @ state.address).mT).mT
        return self.predict(projected, state.fast)

    def transition(self, key, value, state, reference, *, meta=False):
        u = F.normalize(key, dim=-1)
        with torch.enable_grad():
            weights = reference if meta else tuple(w.detach().requires_grad_(True) for w in reference)
            loss = (self.predict(u, weights) - value).square().mean()
            grads = torch.autograd.grad(loss, weights, create_graph=meta)
        eta, rate, forget, rho = self.controls(u).sigmoid().unbind()
        momentum = tuple(eta * m - .05 * rate * g for m, g in zip(state.momentum, grads))
        fast = tuple((1 - .01 * forget) * w + m for w, m in zip(state.fast, momentum))
        address = rho * state.address + (1 - rho) * torch.outer(u, u)
        if not meta:
            fast = tuple(w.detach() for w in fast)
            momentum = tuple(m.detach() for m in momentum)
            address = address.detach()
        return Shard(fast, momentum, address, state.step + 1)

    def scan(self, keys, values, queries, state, *, chunk=1, meta=False):
        if (chunk < 1 or keys.ndim != 2 or not len(keys) or keys.shape != values.shape
                or queries.ndim != 3 or queries.shape[0] != len(keys)
                or queries.shape[-1] != keys.shape[-1]):
            raise ValueError('Expected keys/values [T,D] and queries [T,Q,D]')
        reads = []
        for start in range(0, len(keys), chunk):
            reference = state.fast
            for i in range(start, min(start + chunk, len(keys))):
                state = self.transition(keys[i], values[i], state, reference, meta=meta)
                reads.append(self.read(queries[i], state))
        return torch.stack(reads), state
