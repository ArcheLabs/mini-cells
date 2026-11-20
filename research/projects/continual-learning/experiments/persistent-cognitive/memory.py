"""Independent per-KV-head memory with tokenwise causal transitions."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Shard:
    fast: tuple
    momentum: tuple
    address: torch.Tensor
    step: int


@dataclass(frozen=True)
class UpdateConfig:
    gamma: float = .05
    delta: float = .01
    controls: tuple = (.5, .1, .01, .99)


class NeuralMemory(nn.Module):
    def __init__(self, width=256, expansion=4, ridge=.01, update=UpdateConfig()):
        super().__init__()
        if width < 1 or expansion < 1 or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError('Invalid memory geometry')
        if (not 0 < update.gamma <= 1 or not 0 <= update.delta <= 1
                or len(update.controls) != 4 or not all(0 < x < 1 for x in update.controls)):
            raise ValueError('Invalid update controls')
        self.ridge = ridge
        self.update = update
        self.initial = nn.ParameterList([
            nn.Parameter(torch.empty(width * expansion, width)),
            nn.Parameter(torch.empty(width, width * expansion)),
        ])
        for weight in self.initial:
            nn.init.xavier_uniform_(weight)
        self.controls = nn.Linear(width, 4)
        nn.init.zeros_(self.controls.weight)
        with torch.no_grad():
            self.controls.bias.copy_(torch.tensor(update.controls).logit())

    def create(self):
        fast = tuple(weight.clone() for weight in self.initial)
        return Shard(fast, tuple(torch.zeros_like(w) for w in fast),
                     fast[0].new_zeros((fast[0].shape[1],) * 2), 0)

    @staticmethod
    def predict(x, fast):
        return x + F.layer_norm(F.linear(F.gelu(F.linear(x, fast[0])), fast[1]),
                                (x.shape[-1],))

    def project(self, query, address):
        dtype = torch.float64 if address.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=query.device.type, enabled=False):
            matrix, query = address.to(dtype), query.to(dtype)
            if not torch.isfinite(matrix).all() or not torch.allclose(matrix, matrix.mT):
                raise ValueError('Address matrix must be finite and symmetric')
            eye = torch.eye(matrix.shape[0], device=query.device, dtype=dtype)
            projected = torch.linalg.solve(matrix + self.ridge * eye, (query @ matrix).mT).mT
            if not torch.isfinite(projected).all():
                raise FloatingPointError('Nonfinite address projection')
        return projected

    def read(self, query, state):
        return self.predict(self.project(query, state.address).to(state.fast[0].dtype), state.fast)

    def transition(self, key, value, state, reference, *, meta=False):
        if torch.is_inference_mode_enabled():
            if meta:
                raise ValueError('Meta-training requires autograd')
            with torch.inference_mode(False), torch.no_grad():
                return self.transition(key.clone(), value.clone(), state, reference)
        u = F.normalize(key, dim=-1)
        with torch.enable_grad():
            weights = reference if meta else tuple(w.detach().clone().requires_grad_(True) for w in reference)
            loss = (self.predict(u, weights) - value).square().mean()
            grads = torch.autograd.grad(loss, weights, create_graph=meta)
        eta, rate, forget, rho = self.controls(u).sigmoid().unbind()
        momentum = tuple(eta * m - self.update.gamma * rate * g for m, g in zip(state.momentum, grads))
        fast = tuple((1 - self.update.delta * forget) * w + m for w, m in zip(state.fast, momentum))
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
