"""Reuse PDCLM's continuous Cell/Context/head inventory, not its J0-only router/executor."""

from torch.nn import functional as F
from pd001.models import PDCLM
from exp006.model import initialize
from .q_connection import Connection
from .actions import Registry


class Model(PDCLM):
    def __init__(self, c, arm, seed=None):
        m = c["model"]
        cfg = {
            **m,
            "cell_count": m["accepted_cells"] + m["reserve_slots"],
            "budget": m["budget"],
            "adaptive": True,
            "step_capacity": m["budget"] + 1,
            "router_cost_units": 0.0,
            "cell_cost_units": 1 / m["budget"],
            "context_cost_units": 1 / m["budget"],
        }
        super().__init__(cfg, arm, c["seed"] if seed is None else seed)
        self.experiment_config = c
        self.registry = Registry(m["accepted_cells"], m["reserve_slots"])
        self.controller = Connection(c, self.registry.actions)
        initialize(self, c["seed"] if seed is None else seed)
        self.version = 0

    def head(self, h):
        return F.linear(self.final(h), self.embedding.weight)

    def execute(self, h, action, memory, batches, positions):
        if action == self.registry.stop:
            return h
        normalized = self.norm(h)
        update = (
            self.context(normalized, memory, batches, positions)
            if action == self.registry.ctx
            else self.cells[action](normalized)
        )
        return h + self.config["alpha"] * update

    def forward(
        self,
        tokens,
        *,
        policy="sampled",
        generator=None,
        replay=None,
        past=None,
        probe_coordinates=(),
        uniforms=None,
        edge_zero=False,
        intervention=None,
    ):
        from .rollout import rollout

        return rollout(
            self,
            tokens,
            policy=policy,
            generator=generator,
            replay=replay,
            past=past,
            probe_coordinates=probe_coordinates,
            uniforms=uniforms,
            edge_zero=edge_zero,
            intervention=intervention,
        )

    def prefill(self, tokens, **kwargs):
        return self(tokens, **kwargs)

    def decode_step(self, token, memory, **kwargs):
        return self(token.reshape(-1, 1), past=memory, **kwargs)
