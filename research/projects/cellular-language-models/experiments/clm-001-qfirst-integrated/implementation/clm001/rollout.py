"""Selected rows alone execute. STOP emits no operator; budget end is a different outcome."""

from dataclasses import dataclass
import torch
from exp006.memory import Memory
from .sampling import distribution, inverse_cdf
from .common import tensor_hash, canonical_sha256


@dataclass
class Probe:
    hidden: torch.Tensor
    memory: object
    batch: int
    position: int
    previous: int
    depth: int
    chosen: int
    registry_sha: str
    token_index: tuple

    def fingerprint(self):
        return canonical_sha256(
            {
                "hidden": tensor_hash(self.hidden),
                "memory_key": tensor_hash(self.memory.key),
                "memory_value": tensor_hash(self.memory.value),
                "batch": self.batch,
                "position": self.position,
                "previous": self.previous,
                "depth": self.depth,
                "registry": self.registry_sha,
            }
        )


@dataclass
class Trajectory:
    logits: torch.Tensor
    actions: torch.Tensor
    decision_mask: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor
    lengths: torch.Tensor
    actual_stop: torch.Tensor
    forced_end: torch.Tensor
    cost_units: torch.Tensor
    memory: object
    states: torch.Tensor
    probes: list
    balance: object = None


def rollout(
    model,
    tokens,
    *,
    policy,
    generator,
    replay,
    past,
    probe_coordinates,
    uniforms,
    edge_zero,
    intervention,
):
    if intervention not in (
        None,
        "context_zero",
        "state_permuted",
        "previous_permuted",
        "cell_swapped",
    ):
        raise ValueError("unknown frozen diagnostic intervention")
    c = model.experiment_config
    m = c["model"]
    registry = model.registry
    b, n = tokens.shape
    budget = m["budget"]
    offset = 0 if past is None else past.key.shape[1]
    if offset + n > m["max_sequence_length"]:
        raise ValueError("causal position capacity exceeded")
    if replay is not None and replay.shape != (b, n, budget):
        raise ValueError("replay budget shape differs")
    pos = torch.arange(offset, offset + n, device=tokens.device)
    z = model.embedding(tokens) + model.position(pos)[None]
    memory = model.memory(z, past)
    if intervention == "context_zero":
        memory = Memory(memory.key * 0, memory.value * 0)
    hidden = model.init(z).reshape(-1, m["width"])
    count = b * n
    previous = torch.full((count,), registry.start, dtype=torch.long, device=tokens.device)
    alive = torch.ones(count, dtype=torch.bool, device=tokens.device)
    stopped = ~alive
    lengths = torch.zeros_like(previous)
    batch_ids = torch.arange(b, device=tokens.device).repeat_interleave(n)
    positions = pos.repeat(b)
    traces = []
    lps = []
    ents = []
    values = []
    masks = []
    probes = []
    chosen_probes = set(tuple(v) for v in probe_coordinates)
    for depth in range(budget):
        rows = torch.where(alive)[0]
        action = torch.full_like(previous, -1)
        lp = hidden.new_zeros(count)
        ent = lp.clone()
        value = lp.clone()
        mask = torch.zeros_like(alive)
        if len(rows):
            legal = registry.legal(depth, budget, m["minimum_actions"], tokens.device)
            state = hidden[rows]
            route_previous = previous[rows]
            if intervention == "state_permuted":
                state = state.roll(1, dims=-1)
            if intervention == "previous_permuted":
                route_previous = (route_previous + 1) % (registry.actions + 1)
            q, critic = model.controller(
                state,
                route_previous,
                depth,
                legal,
                registry.version,
                detach_state=model.arm == "QF-FULL",
                edge_zero=edge_zero,
            )
            prob = distribution(
                q, legal, c["policy"]["epsilon"], c["policy"]["temperature"], policy
            )
            u = (
                uniforms[:, :, depth].reshape(-1)[rows]
                if uniforms is not None
                else torch.rand(len(rows), device=tokens.device, generator=generator)
            )
            chosen = inverse_cdf(prob, u)
            if registry.plan and depth < len(registry.plan):
                chosen = torch.full_like(chosen, registry.plan[depth])
                prob = torch.nn.functional.one_hot(chosen, registry.actions).to(prob)
            if replay is not None:
                chosen = replay[:, :, depth].reshape(-1)[rows]
            if (
                torch.any(chosen < 0)
                or torch.any(chosen >= registry.actions)
                or not legal[chosen].all()
            ):
                raise ValueError("illegal accepted action/STOP")
            action = action.index_copy(0, rows, chosen)
            mask = mask.index_fill(0, rows, True)
            lp = lp.index_copy(
                0, rows, prob.gather(1, chosen[:, None]).squeeze(-1).clamp_min(1e-30).log()
            )
            ent = ent.index_copy(0, rows, -(prob * prob.clamp_min(1e-30).log()).sum(-1))
            value = value.index_copy(0, rows, critic)
            # Probe selection is target-free and independent; only at most two token-states are cloned.
            for token, step in chosen_probes:
                if step == depth and bool(alive[token]):
                    index = (rows == token).nonzero().item()
                    detached = Memory(memory.key.detach(), memory.value.detach())
                    probes.append(
                        Probe(
                            hidden[token].detach().clone(),
                            detached,
                            int(batch_ids[token]),
                            int(positions[token]),
                            int(previous[token]),
                            depth,
                            int(chosen[index]),
                            registry.fingerprint,
                            (token // n, token % n),
                        )
                    )
            stop_rows = rows[chosen == registry.stop]
            stopped = stopped.index_fill(0, stop_rows, True)
            alive = alive.index_fill(0, stop_rows, False)
            for a in range(registry.slots + 1):
                selected = rows[chosen == a]
                if not len(selected):
                    continue
                update = model.execute(
                    hidden[selected],
                    (a + 1) % registry.cells
                    if intervention == "cell_swapped" and a < registry.cells
                    else a,
                    memory,
                    batch_ids[selected],
                    positions[selected],
                )
                hidden = hidden.index_copy(0, selected, update)
                lengths = lengths.index_add(0, selected, torch.ones_like(selected))
            previous = previous.index_copy(0, rows, chosen)
        if replay is not None and torch.any(replay[:, :, depth].reshape(-1)[~mask] != -1):
            raise ValueError("no replay decisions after STOP")
        traces.append(action)
        lps.append(lp)
        ents.append(ent)
        values.append(value)
        masks.append(mask)

    def stack(rows):
        return torch.stack(rows, -1).reshape(b, n, budget)

    return Trajectory(
        model.head(hidden.reshape(b, n, -1)),
        stack(traces),
        stack(masks),
        stack(lps),
        stack(ents),
        stack(values),
        lengths.reshape(b, n),
        stopped.reshape(b, n),
        alive.reshape(b, n),
        lengths.reshape(b, n) / budget,
        memory,
        hidden.reshape(b, n, -1),
        probes,
    )
