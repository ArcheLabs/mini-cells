"""Projected synthetic KV adapter; upstream owns all memory updates."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from benchmark import Backend

REVISION = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"


def load_upstream(path):
    path = Path(path).resolve()
    rev = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise ValueError(f"Expected Titans {REVISION}, got {rev}")
    sys.path.insert(0, str(path))
    import titans_pytorch.neural_memory as upstream
    if path not in Path(upstream.__file__).resolve().parents:
        raise RuntimeError("Titans imported from a different checkout")
    return upstream


class Titans(Backend):
    def __init__(self, config, upstream, seed):
        self.config, self.upstream, self.seed = config, upstream, seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.memory = upstream.NeuralMemory(
                dim=config.heads * config.dimension, heads=config.heads,
                dim_head=config.dimension, chunk_size=1, batch_size=1,
                bypass_projections=True, heterogeneous_qkv=True,
                detach_inner_grads=False, mem_model_norm_add_residual=False,
                default_model_kwargs={"depth": 2, "expansion_factor": 1.0},
                max_grad_norm=1.0, default_step_transform_max_lr=.0025,
            ).eval()
        self.state = None

    def fresh(self):
        return type(self)(self.config, self.upstream, self.seed)

    def reset(self, config):
        self.__init__(config, self.upstream, self.seed)

    def address(self, x):
        return self.tensor(x) * self.config.dimension ** -.25

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32).reshape(1, -1, self.config.heads * self.config.dimension)

    @torch.no_grad()
    def write(self, key, value):
        k = self.address(key)
        self.state = self.memory.store_only((k, k, self.tensor(value)), state=self.state)

    @torch.no_grad()
    def read(self, query):
        # Common benchmark contract: empty episodic state returns zero.
        if self.state is None:
            return np.zeros_like(query)
        read, _ = self.memory.read_only((self.address(query), None, None), self.state)
        return read.numpy().reshape(query.shape)

    @staticmethod
    def _copy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone()
        if isinstance(obj, tuple):
            fields = tuple(Titans._copy(v) for v in obj)
            return type(obj)(*fields) if hasattr(obj, "_fields") else fields
        return obj

    def export_state(self):
        return self._copy(self.state)

    def import_state(self, state):
        if state is not None and not isinstance(state, self.upstream.NeuralMemState):
            raise ValueError("Invalid Titans state")
        self.state = self._copy(state)

    def get_stats(self):
        def size(obj):
            if isinstance(obj, torch.Tensor):
                return obj.numel() * obj.element_size()
            if isinstance(obj, tuple):
                return sum(map(size, obj))
            return 8 if isinstance(obj, int) else 0
        return dict(state_bytes=size(self.state), parameters=sum(
            p.numel() for p in self.memory.parameters()))
