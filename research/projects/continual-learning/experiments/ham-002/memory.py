"""Rectangular Delta state and independent parallel branches."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    heads: int = 2
    key_dim: int = 32
    value_dim: int = 32
    beta: float = 1.
    alpha: float = 1.
    normalize_keys: bool = True


class DeltaMemory:
    def __init__(self, config):
        self.reset(config)

    def reset(self, config):
        if min(config.heads, config.key_dim, config.value_dim) < 1 or not (
            0 <= config.beta <= 1 and 0 <= config.alpha <= 1
        ):
            raise ValueError("Invalid Delta configuration")
        self.config = config
        self.matrix = np.zeros((config.heads, config.value_dim, config.key_dim))

    def key(self, x):
        if self.config.normalize_keys:
            return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        return x

    def read(self, q):
        return np.einsum("hvk,...hk->...hv", self.matrix, self.key(q))

    def write(self, k, v):
        k = self.key(k)
        self.matrix *= self.config.alpha
        self.matrix += self.config.beta * (v - np.einsum("hvk,hk->hv", self.matrix, k))[..., None] * k[:, None, :]

    def step(self, q, k, v):
        result = self.read(q)
        self.write(k, v)
        return result

    def fresh(self):
        return type(self)(self.config)

    def export_state(self):
        return self.matrix.copy()

    def import_state(self, matrix):
        if matrix.shape != self.matrix.shape or matrix.dtype != self.matrix.dtype or not np.isfinite(matrix).all():
            raise ValueError("Incompatible Delta state")
        self.matrix = matrix.copy()

    def get_stats(self):
        return dict(state_bytes=self.matrix.nbytes, parameters=0)


class Parallel:
    def __init__(self, delta, titans, coefficient, mode="residual"):
        self.delta, self.titans = delta, titans
        self.coefficient, self.mode = coefficient, mode

    def branches(self, q):
        return self.delta.read(q), self.titans.read(q)

    def read(self, q):
        d, t = self.branches(q)
        if self.mode == "titans":
            return t
        return d if self.coefficient == 0 else d + self.coefficient * t

    def write(self, k, v):
        self.delta.write(k, v)
        self.titans.write(k, v)

    step = DeltaMemory.step

    def fresh(self):
        return type(self)(self.delta.fresh(), self.titans.fresh(), self.coefficient, self.mode)

    def export_state(self):
        return self.delta.export_state(), self.titans.export_state()

    def import_state(self, state):
        self.delta.import_state(state[0])
        self.titans.import_state(state[1])

    def reset(self, _config=None):
        self.delta, self.titans = self.delta.fresh(), self.titans.fresh()

    @property
    def config(self):
        return self.delta.config

    def get_stats(self):
        d, t = self.delta.get_stats(), self.titans.get_stats()
        return {k: d[k] + t[k] for k in d}
