"""A reuses R3d pairs and exact advance; B changes only serial execution and correction features."""

from types import SimpleNamespace
import torch
from minicells_r3d.connection import PathModel
from .graph import GraphConnection, random_action


class Model(PathModel):
    def __init__(self, c, seed, arm):
        super().__init__(c, seed)
        self.arm_name = arm
        self.graph = arm.startswith("B-")
        self.steps = c["training"]["graph_steps"] if self.graph else 4
        self.alpha_graph = c["training"]["graph_alpha"]
        if self.graph:
            self.connection = GraphConnection(self.c["model"]["width"])
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.cells.parameters():
            p.requires_grad_(True)
        self.train_connection = False
        self.work = {
            k: 0
            for k in (
                "input_rows",
                "context_rows",
                "cell_rows",
                "gate_rows",
                "head_rows",
                "policy_rows",
            )
        }

    def scores(self, h, t, previous):
        z = self.gate(h, t // 2 if self.graph else t).detach()
        if self.graph:
            q, d = self.connection(self.norm(h), t, z, previous)
        else:
            q, d = self.connection(self.norm(h), t, z)
        self.work["gate_rows"] += len(h)
        self.work["policy_rows"] += len(h)
        return q, d, z

    def select(self, h, t, previous, *, epsilon=0, rng=None, forced=None, random_ids=None):
        with torch.no_grad():
            q, delta, z = self.scores(h, t, previous)
            if self.graph:
                action = z.argmax(-1) if self.arm_name == "B-F" else q.argmax(-1)
                if self.arm_name == "B-R":
                    action = random_ids
            else:
                base = self.connection.base_action(z)
                action = (
                    base
                    if self.arm_name == "A-F"
                    else torch.where((delta == 0).all(-1), base, q.argmax(-1))
                )
            if epsilon and self.arm_name != "B-R":
                # Train-time categorical policy plus controlled uniform exposure; evaluation is hard Argmax.
                uniform = torch.rand(q.shape, generator=rng).to(q.device).clamp_(1e-7, 1 - 1e-7)
                sampled = (q - (-uniform.log()).log()).argmax(-1)
                mask = torch.rand(len(h), generator=rng) < epsilon
                pick = torch.randint(6 if self.graph else 15, (len(h),), generator=rng)
                action = torch.where(mask.to(h.device), pick.to(h.device), sampled)
            if forced is not None:
                action = forced
        return action, z

    def transition(self, h, t, previous, action):
        if not self.graph:
            result = super().advance(h, t, action)
            self.work["gate_rows"] += len(h)
            self.work["cell_rows"] += 2 * len(h)
            return result, previous
        normal = self.norm(h)
        delta = torch.zeros_like(h)
        for i, cell in enumerate(self.cells):
            rows = torch.where(action == i)[0]
            if rows.numel():
                delta = delta.index_add(0, rows, cell(normal[rows]))
        self.work["cell_rows"] += len(h)
        return h + self.alpha_graph * delta, action

    def head(self, h):
        self.work["head_rows"] += h.numel() // h.shape[-1]
        return super().head(h)

    def initial(self, tokens):
        b, n = tokens.shape
        d = self.c["model"]["width"]
        pos = torch.arange(n, device=tokens.device)
        z = self.embedding(tokens) + self.position(pos)[None]
        memory = self.memory(z)
        h = self.init(z).reshape(-1, d)
        h = h + self.c["model"]["alpha"] * self.context(
            self.norm(h),
            memory,
            torch.arange(b, device=tokens.device).repeat_interleave(n),
            pos.repeat(b),
        )
        self.work["input_rows"] += b * n
        self.work["context_rows"] += b * n
        return h, memory

    def forward(self, tokens, *, epsilon=0, route_rng=None, keep_trace=False, **kwargs):
        b, n = tokens.shape
        h, memory = self.initial(tokens)
        previous = torch.full((len(h),), 6, device=h.device, dtype=torch.long)
        trace = []
        actions = []
        counts = []
        for t in range(self.steps):
            action, z = self.select(
                h,
                t,
                previous,
                epsilon=epsilon,
                rng=route_rng,
                random_ids=random_action(tokens, t, self.seed) if self.arm_name == "B-R" else None,
            )
            if keep_trace:
                trace.append(
                    {"state": h.detach(), "previous": previous, "action": action, "stage": t}
                )
            actions.append(action)
            indices = action[:, None] if self.graph else self.connection.pairs[action]
            counts.append(torch.bincount(indices.reshape(-1), minlength=6))
            h, previous = self.transition(h, t, previous, action)
        return SimpleNamespace(
            logits=self.head(h.reshape(b, n, -1)),
            trace=trace,
            actions=torch.stack(actions, -1),
            operator_rows=torch.stack(counts),
            memory=memory,
        )


def build(c, parent, seed, arm, device):
    m = Model(c, seed, arm)
    current = m.state_dict()
    keys = {k for k in current if not k.startswith("connection.")}
    if set(parent["model"]) != keys:
        raise RuntimeError("PARENT_MODEL_KEYS_DIFFER")
    current.update(parent["model"])
    m.load_state_dict(current, strict=True)
    return m.to(device)
