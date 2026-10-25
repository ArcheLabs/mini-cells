"""Known primitive hard paths; inputs expose only value and requested affine coefficients."""

import torch
from torch import nn
from types import SimpleNamespace
from minicells_r3e.graph import GraphConnection


class AffineCell(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))

    def forward(self, h):
        return torch.cat([(h[:, :1] * self.a + self.b), h[:, 1:]], -1)


class KnownPathModel(nn.Module):
    def __init__(self, fixture):
        super().__init__()
        self.cells = nn.ModuleList([AffineCell(a, b) for a, b in fixture["primitive_affines"]])
        self.connection = GraphConnection(3)
        low, high = fixture["classification_values"]
        self.register_buffer("classes", torch.arange(low, high + 1, dtype=torch.float32))
        self.steps = fixture["path_steps"]
        self.graph = True
        self.meter = None
        self.work = {k: 0 for k in ("input_rows", "cell_rows", "policy_rows", "head_rows")}

    def scores(self, h, t, previous):
        self.work["policy_rows"] += len(h)
        if self.meter is not None:
            self.meter.add("policy_rows", len(h))
        base = torch.zeros((len(h), 6), dtype=h.dtype, device=h.device)
        return self.connection(h, t, base, previous)[0], None, base

    def select(self, h, t, previous, epsilon=0, rng=None):
        with torch.no_grad():
            q, _, base = self.scores(h, t, previous)
            action = q.argmax(-1)
            if epsilon:
                u = torch.rand(q.shape, generator=rng).to(q.device).clamp(1e-7, 1 - 1e-7)
                sampled = (q - (-u.log()).log()).argmax(-1)
                mask = torch.rand(len(h), generator=rng).to(h.device) < epsilon
                random = torch.randint(6, (len(h),), generator=rng).to(h.device)
                action = torch.where(mask, random, sampled)
        return action, base

    def transition(self, h, t, previous, action):
        result = torch.zeros_like(h)
        for index, cell in enumerate(self.cells):
            selected = torch.where(action == index)[0]
            if selected.numel():
                result = result.index_add(0, selected, cell(h[selected]))
        self.work["cell_rows"] += len(h)
        if self.meter is not None:
            self.meter.add("cell_rows", len(h))
        return result, action

    def head(self, h):
        self.work["head_rows"] += len(h)
        if self.meter is not None:
            self.meter.add("head_rows", len(h))
        return -((h[:, :1] - self.classes[None]) ** 2)

    def forward(self, features, *, keep_trace=False, epsilon=0, route_rng=None):
        if features.ndim != 2 or features.shape[-1] != 3:
            raise ValueError("JCT_ROUTER_INPUT_IS_ONLY_VALUE_AND_AFFINE_DESCRIPTOR")
        h = features
        previous = torch.full((len(h),), 6, dtype=torch.long, device=h.device)
        self.work["input_rows"] += len(h)
        if self.meter is not None:
            self.meter.add("input_rows", len(h))
        actions = []
        trace = []
        for t in range(self.steps):
            action, _ = self.select(h, t, previous, epsilon=epsilon, rng=route_rng)
            if keep_trace:
                trace.append(dict(state=h.detach(), previous=previous.detach(), stage=t))
            actions.append(action)
            h, previous = self.transition(h, t, previous, action)
        return SimpleNamespace(
            logits=self.head(h), actions=torch.stack(actions, -1), trace=trace, value=h[:, 0]
        )
