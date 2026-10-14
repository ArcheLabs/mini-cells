"""No fixed backbone: only selected live positions execute real residual actions."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from exp006.model import initialize
from exp006.memory import InputMemory
from exp006.cells import FeedForward, ContextRead
from exp006.baseline import MoELM
from .router import Router


@dataclass
class Trajectory:
    logits: torch.Tensor
    actions: torch.Tensor
    decision_mask: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor
    lengths: torch.Tensor
    forced_stop: torch.Tensor
    stop_prob: torch.Tensor
    memory: object
    states: torch.Tensor
    state_records: list
    cost_units: torch.Tensor
    balance: object = None
    context_zero: bool = False
    edge_mode: str = "normal"


class PDCLM(nn.Module):
    def __init__(self, config, arm, seed=0):
        super().__init__()
        self.config, self.arm = dict(config), arm
        c, d = self.config, config["width"]
        if c["memory_mode"] != "static_input_readonly":
            raise ValueError("per-token reward requires path-independent input memory")
        if d % c["heads"] or c["budget"] > c["step_capacity"] or c["budget"] < 1:
            raise ValueError("invalid width or path budget")
        self.embedding = nn.Embedding(c["vocab_size"], d)
        self.position = nn.Embedding(c["max_sequence_length"], d)
        self.init = nn.Linear(d, d, bias=False)
        self.memory = InputMemory(d)
        self.norm, self.final = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cells = nn.ModuleList(FeedForward(d, c["cell_hidden"]) for _ in range(c["cell_count"]))
        self.context = ContextRead(d, c["heads"])
        self.controller = Router(c, state_only=c.get("state_only", False))
        initialize(self, seed)

    def forward(
        self,
        tokens,
        *,
        policy="greedy",
        replay=None,
        generator=None,
        past=None,
        edge_mode="normal",
        edge_permutation=None,
        context_zero=False,
        stop_after=None,
        action_overrides=None,
        capture=False,
        minimum_actions=0,
        token_overrides=None,
        replay_prefix=None,
        prefix_length=0,
    ):
        if not 0 <= minimum_actions <= self.config["budget"]:
            raise ValueError("minimum actions exceeds trained budget")
        if policy not in ("greedy", "sampled"):
            raise ValueError("invalid policy")
        c = self.config
        b, n = tokens.shape
        offset = 0 if past is None else past.key.shape[1]
        if offset + n > c["max_sequence_length"]:
            raise ValueError("position budget exceeded")
        positions = torch.arange(offset, offset + n, device=tokens.device)
        z = self.embedding(tokens) + self.position(positions)[None]
        memory = self.memory(z, past)
        h = self.init(z).reshape(-1, c["width"])
        count, ctx, stop = b * n, c["cell_count"], c["cell_count"] + 1
        previous = torch.full((count,), stop + 1, dtype=torch.long, device=h.device)
        last_cell = torch.full_like(previous, -1)
        alive = torch.ones(count, dtype=torch.bool, device=h.device)
        lengths = torch.zeros_like(previous)
        units = h.new_zeros(count)
        batches = torch.arange(b, device=h.device).repeat_interleave(n)
        pos = positions.repeat(b)
        actions, masks, lps, ents, vals, stops, records = [], [], [], [], [], [], []
        if replay is not None and replay.shape != (b, n, c["budget"]):
            raise ValueError("replay shape mismatch")
        for t in range(c["budget"]):
            if stop_after is not None and t >= stop_after:
                alive = torch.zeros_like(alive)
            rows = torch.where(alive)[0]
            act = torch.full_like(previous, -1)
            mask = torch.zeros_like(alive)
            policy_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
            lp, ent, val, sp = [h.new_zeros(count, dtype=policy_dtype) for _ in range(4)]
            if rows.numel():
                state = h.index_select(0, rows)
                if capture and state.requires_grad:
                    state.retain_grad()
                scores, value = self.controller(state, previous[rows], t, edge_mode, edge_permutation)
                legal = torch.ones_like(scores, dtype=torch.bool)
                if not c["adaptive"] or t < minimum_actions:
                    legal[:, stop] = False
                if c.get("self_only", False):
                    settled = last_cell[rows] >= 0
                    legal[settled, :ctx] = False
                    legal[torch.where(settled)[0], last_cell[rows][settled]] = True
                scores = scores.masked_fill(~legal, float("-inf"))
                logq = scores.log_softmax(-1)
                q = logq.exp()
                chosen = (
                    replay[:, :, t].reshape(-1)[rows]
                    if replay is not None
                    else torch.multinomial(q, 1, generator=generator).squeeze(-1)
                    if policy == "sampled"
                    else scores.argmax(-1)
                )
                if replay_prefix is not None and t < prefix_length:
                    chosen = replay_prefix[:, :, t].reshape(-1)[rows]
                if token_overrides and t in token_overrides:
                    selected, replacement = token_overrides[t]
                    chosen = torch.where(selected.reshape(-1)[rows], replacement.reshape(-1)[rows], chosen)
                if action_overrides and t in action_overrides:
                    chosen = torch.full_like(chosen, action_overrides[t])
                if (
                    torch.any(chosen < 0)
                    or torch.any(chosen > stop)
                    or not legal.gather(1, chosen[:, None]).all()
                ):
                    raise ValueError("illegal path action")
                if stop_after is not None and t >= stop_after:
                    alive = alive.index_fill(0, rows, False)
                    chosen = torch.full_like(chosen, -1)  # diagnostic forced halt, no fake decision
                else:
                    act = act.index_copy(0, rows, chosen)
                    mask = mask.index_fill(0, rows, True)
                    lp = lp.index_copy(0, rows, logq.gather(1, chosen[:, None]).squeeze(-1))
                    ent = ent.index_copy(0, rows, -(q * logq.masked_fill(~legal, 0)).sum(-1))
                    val = val.index_copy(0, rows, value.to(val.dtype))
                    sp = sp.index_copy(0, rows, q[:, stop])
                    units = units.index_add(
                        0,
                        rows,
                        torch.full((rows.numel(),), c["router_cost_units"], device=h.device, dtype=h.dtype),
                    )
                    stopped = rows[chosen == stop]
                    alive = alive.index_fill(0, stopped, False)
                    for a in range(ctx + 1):
                        selected = rows[chosen == a]
                        if not selected.numel():
                            continue
                        normalized = self.norm(h[selected])
                        update = (
                            (
                                torch.zeros_like(normalized)
                                if context_zero
                                else self.context(normalized, memory, batches[selected], pos[selected])
                            )
                            if a == ctx
                            else self.cells[a](normalized)
                        )
                        new = h[selected] + c["alpha"] * update.to(h.dtype)
                        if capture:
                            records.append(
                                {
                                    "step": t,
                                    "action": a,
                                    "rows": selected.detach(),
                                    "before": h[selected],
                                    "delta": c["alpha"] * update,
                                    "router_state": state,
                                }
                            )
                        h = h.index_copy(0, selected, new)
                        lengths = lengths.index_add(0, selected, torch.ones_like(selected))
                        if a < ctx:
                            last_cell = last_cell.index_fill(0, selected, a)
                        # Cost reward is explicitly an operation-budget proxy, not hardware FLOPs.
                        cost = c["context_cost_units"] if a == ctx else c["cell_cost_units"]
                        units = units.index_add(
                            0, selected, torch.full((selected.numel(),), cost, device=h.device, dtype=h.dtype)
                        )
                    previous = previous.index_copy(0, rows, chosen)
            if replay is not None:
                replay_t = replay[:, :, t].reshape(-1)
                if torch.any(replay_t[~mask] != -1):
                    raise ValueError("stopped replay positions must be padded with -1")
            actions.append(act)
            masks.append(mask)
            lps.append(lp)
            ents.append(ent)
            vals.append(val)
            stops.append(sp)
        logits = F.linear(self.final(h).reshape(b, n, -1), self.embedding.weight)

        def stack(values):
            return torch.stack(values, -1).reshape(b, n, -1)

        return Trajectory(
            logits,
            stack(actions),
            stack(masks),
            stack(lps),
            stack(ents),
            stack(vals),
            lengths.reshape(b, n),
            alive.reshape(b, n),
            stack(stops),
            memory,
            h.reshape(b, n, -1),
            records,
            units.reshape(b, n),
            context_zero=context_zero,
            edge_mode=edge_mode,
        )

    def prefill(self, tokens, **kwargs):
        return self(tokens, **kwargs)

    def decode_step(self, token, memory, **kwargs):
        return self(token.reshape(-1, 1), past=memory, **kwargs)


def make_model(c, arm, seed=0):
    return MoELM(c, seed) if arm == "M0" else PDCLM(c, arm, seed)
