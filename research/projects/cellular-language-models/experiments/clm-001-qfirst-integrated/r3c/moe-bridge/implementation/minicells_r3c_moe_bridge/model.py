"""Exact selected-row sparse dispatch through shared historical computation primitives."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from exp006.cells import FeedForward, ContextRead
from exp006.memory import InputMemory
from exp006.model import initialize
from .gate import Gate, controlled_indices
from .common import validate, ARMS


@dataclass
class Stage:
    inputs: torch.Tensor
    logits: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    delta: torch.Tensor


@dataclass
class Output:
    logits: torch.Tensor
    stages: list
    balance: torch.Tensor
    memory: object
    states: torch.Tensor
    operator_rows: torch.Tensor


class BridgeModel(nn.Module):
    def __init__(self, c, arm, seed):
        super().__init__()
        validate(c)
        if arm not in ARMS:
            raise ValueError("unknown arm")
        self.c = c
        self.arm = arm
        self.seed = seed
        m = c["model"]
        d = m["width"]
        # Constructors cannot consume an experiment's existing global CPU RNG.
        with torch.random.fork_rng(devices=[]):
            self.embedding = nn.Embedding(m["vocab_size"], d)
            self.position = nn.Embedding(m["max_sequence_length"], d)
            self.init = nn.Linear(d, d, bias=False)
            self.memory = InputMemory(d)
            self.norm = nn.LayerNorm(d)
            self.final = nn.LayerNorm(d)
            self.cells = nn.ModuleList(
                FeedForward(d, m["cell_hidden"]) for _ in range(c["routing"]["cells"])
            )
            self.context = ContextRead(d, m["heads"])
            self.gate = Gate(c)
            initialize(self, seed)

    def head(self, h):
        return F.linear(self.final(h), self.embedding.weight)

    def forward(self, tokens, *, route_rng=None, past=None, replay=None, intervention=None):
        b, n = tokens.shape
        m = self.c["model"]
        r = self.c["routing"]
        d = m["width"]
        offset = 0 if past is None else past.key.shape[1]
        if offset + n > m["max_sequence_length"]:
            raise ValueError("position capacity exceeded")
        pos = torch.arange(offset, offset + n, device=tokens.device)
        z = self.embedding(tokens) + self.position(pos)[None]
        memory = self.memory(z, past)
        h = self.init(z).reshape(-1, d)
        batch = torch.arange(b, device=h.device).repeat_interleave(n)
        positions = pos.repeat(b)
        h = h + m["alpha"] * self.context(self.norm(h), memory, batch, positions)
        traces = []
        balances = []
        counts = []
        for t in range(r["stages"]):
            before = h
            if self.arm in ("A", "B"):
                with torch.no_grad():
                    logits = self.gate(h, t)
                indices = controlled_indices(
                    {**self.c, "stage_index": t}, self.arm, positions, route_rng
                )
                weights = torch.full((b * n, 2), 0.5, device=h.device, dtype=h.dtype)
            else:
                logits = self.gate(h, t)
                indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :2]
                weights = (logits.gather(1, indices) / r["temperature"]).softmax(-1)
                if self.arm == "D":
                    weights = weights.detach()
            # Interventions and suffix replay have no access to targets.
            if replay is not None:
                indices = replay[t]["indices"].to(h.device)
                weights = replay[t]["weights"].to(h)
            if intervention:
                indices, weights = intervention(t, h, logits, indices, weights)
            if indices.shape != (b * n, 2) or weights.shape != (b * n, 2):
                raise ValueError("invalid Top-2 shape")
            if replay is not None or intervention:
                if not bool(((indices >= 0) & (indices < r["cells"])).all()) or bool(
                    (indices[:, 0] == indices[:, 1]).any()
                ):
                    raise ValueError("invalid distinct Top-2 dispatch")
                if (
                    not torch.isfinite(weights).all()
                    or not torch.allclose(
                        weights.sum(-1), torch.ones(b * n, device=h.device, dtype=h.dtype)
                    )
                    or bool((weights < 0).any())
                ):
                    raise ValueError("invalid mixture weights")
            probability = (logits / r["temperature"]).softmax(-1)
            exposure = torch.bincount(indices.reshape(-1), minlength=r["cells"]).to(logits) / (
                2 * b * n
            )
            balances.append(r["cells"] * (exposure.detach() * probability.mean(0)).sum())
            normalized = self.norm(h)
            delta = torch.zeros_like(h)
            for cell_id, cell in enumerate(self.cells):
                rows, slots = torch.where(indices == cell_id)
                if rows.numel():
                    value = cell(normalized[rows])
                    delta = delta.index_add(0, rows, weights[rows, slots, None].to(value) * value)
            counts.append(torch.bincount(indices.reshape(-1), minlength=r["cells"]))
            h = h + m["alpha"] * delta
            traces.append(Stage(before, logits, indices, weights, delta))
        return Output(
            self.head(h.reshape(b, n, d)),
            traces,
            torch.stack(balances).mean(),
            memory,
            h.reshape(b, n, d),
            torch.stack(counts),
        )
