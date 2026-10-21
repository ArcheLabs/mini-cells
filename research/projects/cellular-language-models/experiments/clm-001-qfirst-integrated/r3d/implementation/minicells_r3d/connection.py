"""Target-free 15-pair correction. Original D weights and mixing remain frozen."""

import itertools
import torch
from torch import nn
from torch.nn import functional as F
from minicells_r3c_moe_bridge.model import BridgeModel


class Connection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stages = c["routing"]["stages"]
        self.register_buffer(
            "pairs", torch.tensor(list(itertools.combinations(range(c["routing"]["cells"]), 2)))
        )
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(c["model"]["width"] + self.stages, len(self.pairs))
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, h, t, z):
        stage = F.one_hot(torch.full((len(h),), t, device=h.device), self.stages).to(h)
        delta = self.linear(torch.cat([h.detach(), stage], -1))
        base = z.detach()[:, self.pairs].sum(-1)
        return base + delta, delta

    def base_action(self, z):
        indices = torch.argsort(z, descending=True, stable=True, dim=-1)[:, :2].sort(-1).values
        return (indices[:, None, :] == self.pairs[None, :, :]).all(-1).long().argmax(-1)

    def choose(self, h, t, z):
        scores, delta = self(h, t, z)
        selected = scores.argmax(-1)
        # Exact zero initialization uses D's stable Top2, including floating-point sum ties.
        selected = torch.where((delta == 0).all(-1), self.base_action(z), selected)
        return selected, scores


class PathModel(BridgeModel):
    def __init__(self, c, seed):
        super().__init__(c["engine"], "D", seed)
        self.connection = Connection(self.c)
        for p in self.parameters():
            p.requires_grad_(False)

    def indices_weights(self, z, action):
        pair = self.connection.pairs[action]
        order = torch.argsort(z.gather(1, pair), descending=True, stable=True, dim=-1)
        indices = pair.gather(1, order)
        return indices, (z.gather(1, indices) / self.c["routing"]["temperature"]).softmax(
            -1
        ).detach()

    def decision(self, h, t, *, original=False, epsilon=0, rng=None, z=None):
        with torch.no_grad():
            z = self.gate(h, t) if z is None else z
            baseline = self.connection.base_action(z)
            action = baseline if original else self.connection.choose(self.norm(h), t, z)[0]
            if epsilon:
                mask = torch.rand(len(h), generator=rng) < epsilon
                explore = torch.randint(len(self.connection.pairs), (len(h),), generator=rng)
                action = torch.where(mask.to(h.device), explore.to(h.device), action)
            indices, weights = self.indices_weights(z, action)
        return action, indices, weights, baseline

    def forward(
        self, tokens, *, route_rng=None, intervention=None, epsilon=0, original=None, **kwargs
    ):
        original = getattr(self, "force_original", False) if original is None else original

        def route(t, h, z, indices, weights):
            _, ids, w, _ = self.decision(
                h, t, original=original, epsilon=epsilon, rng=route_rng, z=z
            )
            if intervention:
                return intervention(t, h, z, ids, w)
            return ids, w

        return super().forward(tokens, route_rng=route_rng, intervention=route, **kwargs)

    def advance(self, h, t, action=None):
        # Same selected-row dispatch as BridgeModel; used only on sampled detached states.
        if action is None:
            _, indices, weights, _ = self.decision(h, t)
        else:
            with torch.no_grad():
                z = self.gate(h, t)
            indices, weights = self.indices_weights(z, action)
        delta = torch.zeros_like(h)
        normalized = self.norm(h)
        for i, cell in enumerate(self.cells):
            rows, slots = torch.where(indices == i)
            if rows.numel():
                delta = delta.index_add(
                    0, rows, weights[rows, slots, None] * cell(normalized[rows])
                )
        return h + self.c["model"]["alpha"] * delta

    def set_trainable(self, arm):
        self.force_original = arm in ("F0", "F")
        for p in self.parameters():
            p.requires_grad_(False)
        if arm in ("F", "P", "J"):
            for p in self.cells.parameters():
                p.requires_grad_(True)
        if arm in ("L", "M", "J"):
            for p in self.connection.parameters():
                p.requires_grad_(True)
