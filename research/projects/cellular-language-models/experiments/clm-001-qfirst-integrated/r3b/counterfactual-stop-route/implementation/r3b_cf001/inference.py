"""Inference-only policy adapter. Original Model/Registry/Connection/operators stay unchanged."""

from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from clm001_r3.sampling import distribution
from clm001_r3.common import tensor_hash
from clm001_r3.rollout import Probe
from minicells_runtime.store import canonical, ContinuationError


def greedy(q, legal):
    mask = legal.expand_as(q)
    if not mask.any(-1).all():
        raise ContinuationError("CF_NO_LEGAL_ACTION")
    score = q.masked_fill(~mask, float("inf"))
    choice = score.argmin(-1)
    return torch.where(
        mask[:, -1] & (score[:, -1] == score.min(-1).values),
        torch.full_like(choice, q.shape[-1] - 1),
        choice,
    )


def action_selection(model, q, legal, depth, previous, ctx_count, policy):
    base = greedy(q, legal)
    mask = legal.expand_as(q).clone()
    if policy in ("CTX_ONCE", "CTX_ONCE_AND_NO_REPEAT"):
        mask[:, model.registry.ctx] &= ctx_count == 0
    if policy in ("NO_ADJACENT_SAME_CELL", "CTX_ONCE_AND_NO_REPEAT"):
        repeat = (previous >= 0) & (previous < model.registry.slots)
        rows = torch.where(repeat)[0]
        mask[rows, previous[rows]] = False
    chosen = greedy(q, mask)
    if policy.startswith("FORCE_STOP_D") and depth == int(policy.rsplit("D", 1)[1]):
        if not bool(legal[model.registry.stop]):
            raise ContinuationError("CF_STOP_ILLEGAL")
        chosen = torch.full_like(chosen, model.registry.stop)
    plan = model.registry.plan
    if plan and depth < len(plan):
        forced = int(plan[depth])
        if not bool(legal[forced]):
            raise ContinuationError("CF_ILLEGAL_FROZEN_PLAN")
        conflict = not mask[:, forced].all() or (
            policy.startswith("FORCE_STOP_D")
            and depth == int(policy.rsplit("D", 1)[1])
            and forced != model.registry.stop
        )
        if conflict:
            raise ContinuationError("CF_FROZEN_PLAN_CONFLICT")
        chosen = torch.full_like(chosen, forced)
        base = chosen.clone()
    return chosen, chosen != base


def initialize(model, tokens):
    b, n = tokens.shape
    positions = torch.arange(n, device=tokens.device)
    z = model.embedding(tokens) + model.position(positions)[None]
    memory = model.memory(z, None)
    h = model.init(z).reshape(-1, model.config["width"])
    return (
        h,
        memory,
        torch.arange(b, device=tokens.device).repeat_interleave(n),
        positions.repeat(b),
    )


@torch.no_grad()
def rollout(model, tokens, policy="NOOP_CONTROL"):
    h, memory, batches, positions = initialize(model, tokens)
    b, n = tokens.shape
    r = model.registry
    previous = torch.full((b * n,), r.start, device=tokens.device, dtype=torch.long)
    alive = torch.ones(b * n, device=tokens.device, dtype=torch.bool)
    stopped = torch.zeros_like(alive)
    lengths = torch.zeros_like(previous)
    ctx = torch.zeros_like(previous)
    touched = torch.zeros_like(alive)
    first = torch.full_like(previous, -1)
    replacements = torch.full_like(previous, -1)
    blocked = torch.full_like(previous, -1)
    actions = []
    for depth in range(model.config["budget"]):
        rows = torch.where(alive)[0]
        column = torch.full_like(previous, -1)
        if len(rows):
            legal = r.legal(
                depth, model.config["budget"], model.config["minimum_actions"], tokens.device
            )
            q, _ = model.controller(
                h[rows], previous[rows], depth, legal, r.version, detach_state=True
            )
            chosen, changed = action_selection(
                model, q, legal, depth, previous[rows], ctx[rows], policy
            )
            newly = rows[changed & ~touched[rows]]
            first[newly] = depth
            blocked[newly] = greedy(q, legal)[changed & ~touched[rows]]
            replacements[newly] = chosen[changed & ~touched[rows]]
            touched[rows] |= changed
            column[rows] = chosen
            stoprows = rows[chosen == r.stop]
            stopped[stoprows] = True
            alive[stoprows] = False
            for action in range(r.slots + 1):
                selected = rows[chosen == action]
                if not len(selected):
                    continue
                update = model.execute(
                    h[selected], action, memory, batches[selected], positions[selected]
                )
                h = h.index_copy(0, selected, update)
                lengths[selected] += 1
                if action == r.ctx:
                    ctx[selected] += 1
            previous[rows] = chosen
        actions.append(column)
    return SimpleNamespace(
        logits=model.head(h.reshape(b, n, -1)),
        actions=torch.stack(actions, -1).reshape(b, n, -1),
        states=h.reshape(b, n, -1),
        lengths=lengths.reshape(b, n),
        actual_stop=stopped.reshape(b, n),
        forced_end=alive.reshape(b, n),
        cost_units=lengths.reshape(b, n) / model.config["budget"],
        touched=touched.reshape(b, n),
        first_trigger=first.reshape(b, n),
        replacement=replacements.reshape(b, n),
        blocked=blocked.reshape(b, n),
        memory=memory,
    )


def statistics(out, targets, registry):
    nll = F.cross_entropy(
        out.logits.reshape(-1, out.logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    paths = out.actions.reshape(-1, out.actions.shape[-1]).cpu().tolist()
    ctx = []
    loops = []
    two = []
    triples = []
    transitions = {}
    total_edges = 0
    for path in paths:
        p = [a for a in path if a >= 0 and a != registry.stop]
        ctx.append(p.count(registry.ctx))
        loops.append(sum(a == b and a < registry.slots for a, b in zip(p, p[1:])))
        two.append(sum(a == c and a != b for a, b, c in zip(p, p[1:], p[2:])))
        triples.append(sum((a, b, c) == (0, registry.ctx, 0) for a, b, c in zip(p, p[1:], p[2:])))
        for a, b in zip(p, p[1:]):
            key = f"{a}->{b}"
            transitions[key] = transitions.get(key, 0) + 1
            total_edges += 1
    return {
        "nll": nll.cpu().reshape(-1).tolist(),
        "cost": out.cost_units.cpu().reshape(-1).tolist(),
        "lengths": out.lengths.cpu().reshape(-1).tolist(),
        "actual_stop": out.actual_stop.cpu().reshape(-1).tolist(),
        "forced_end": out.forced_end.cpu().reshape(-1).tolist(),
        "ctx_calls": ctx,
        "self_loops": loops,
        "two_step_loops": two,
        "cell0_ctx_cell0": triples,
        "transition_matrix": transitions,
        "transition_count": total_edges,
        "actions": paths,
        "action_sha256": tensor_hash(out.actions),
        "state_sha256": tensor_hash(out.states),
        "touched": getattr(out, "touched", torch.zeros_like(out.actual_stop))
        .cpu()
        .reshape(-1)
        .tolist(),
        "first_trigger": getattr(out, "first_trigger", torch.full_like(out.lengths, -1))
        .cpu()
        .reshape(-1)
        .tolist(),
        "blocked_action": getattr(out, "blocked", torch.full_like(out.lengths, -1))
        .cpu()
        .reshape(-1)
        .tolist(),
        "replacement": getattr(out, "replacement", torch.full_like(out.lengths, -1))
        .cpu()
        .reshape(-1)
        .tolist(),
    }


@torch.no_grad()
def census(model, tokens, targets, story_id, window_id):
    """Observe the unchanged native rollout; batch CPU transfers once per depth."""
    records = []
    alive = torch.ones(tokens.numel(), dtype=torch.bool, device=tokens.device)
    prefixes = [[] for _ in range(tokens.numel())]
    ctx = [0] * tokens.numel()
    repeats = [0] * tokens.numel()
    memory_hash = None
    target_cpu = targets.detach().cpu().reshape(-1)
    target_hashes = [tensor_hash(v.reshape(())) for v in target_cpu]
    registry_sha = model.registry.fingerprint
    # Memory is input-only; calculating it again does not depend on path decisions.
    _, memory, _, positions = initialize(model, tokens)
    memory_hash = (tensor_hash(memory.key), tensor_hash(memory.value))

    def observe(module, args, output):
        hidden, previous, depth, legal, _ = args
        q, _ = output
        rows = torch.where(alive)[0]
        chosen = distribution(
            q,
            legal,
            model.experiment_config["policy"]["epsilon"],
            model.experiment_config["policy"]["temperature"],
            "greedy",
        ).argmax(-1)
        if model.registry.plan and depth < len(model.registry.plan):
            chosen = torch.full_like(chosen, model.registry.plan[depth])
        hs = hidden.detach().cpu()
        qs = q.detach().cpu()
        pr = previous.cpu().tolist()
        acts = chosen.cpu().tolist()
        ids = rows.cpu().tolist()
        lm = legal.cpu().tolist()
        for local, token in enumerate(ids):
            a = acts[local]
            fingerprint = canonical(
                {
                    "hidden": tensor_hash(hs[local]),
                    "memory_key": memory_hash[0],
                    "memory_value": memory_hash[1],
                    "batch": token // tokens.shape[1],
                    "position": token % tokens.shape[1],
                    "previous": pr[local],
                    "depth": depth,
                    "registry": registry_sha,
                }
            )
            cells = [i for i in range(model.registry.slots) if lm[i]]
            best = min(cells, key=lambda i: float(qs[local, i])) if cells else None
            scores = [float(v) if np.isfinite(float(v)) else None for v in qs[local]]
            records.append(
                {
                    "story_id": story_id,
                    "window_id": window_id,
                    "token_position": token,
                    "depth_zero_based": depth,
                    "state_fingerprint": fingerprint,
                    "previous_action": pr[local],
                    "ctx_seen": bool(ctx[token]),
                    "n_ctx_so_far": ctx[token],
                    "consecutive_same_cell_count": repeats[token],
                    "legal_actions": [i for i, v in enumerate(lm) if v],
                    "q_scores": scores,
                    "chosen_action": a,
                    "q_stop": scores[model.registry.stop],
                    "q_ctx": scores[model.registry.ctx],
                    "best_cell": best,
                    "q_best_cell": scores[best] if best is not None else None,
                    "q_chosen": scores[a],
                    "stop_legal": lm[model.registry.stop],
                    "actual_stop": a == model.registry.stop,
                    "action_prefix": list(prefixes[token]),
                    "action_prefix_fingerprint": canonical(prefixes[token]),
                    "target_identity": target_hashes[token],
                    "ctx_action": model.registry.ctx,
                }
            )
            repeats[token] = (
                repeats[token] + 1
                if prefixes[token] and a == prefixes[token][-1] and a < model.registry.slots
                else int(a < model.registry.slots)
            )
            prefixes[token].append(a)
            ctx[token] += int(a == model.registry.ctx)
        alive[rows[chosen == model.registry.stop]] = False

    handle = model.controller.register_forward_hook(observe)
    try:
        generator = torch.Generator(device=tokens.device).manual_seed(
            model.experiment_config["evaluation"]["seed"]
        )
        native = model(tokens, policy="greedy", generator=generator)
    finally:
        handle.remove()
    terminal = native.forced_end.reshape(-1).cpu().tolist()
    for row in records:
        row["forced_end_path"] = terminal[row["token_position"]]
    return native, records


@torch.no_grad()
def reconstruct(model, tokens, paths, record):
    h, memory, batches, positions = initialize(model, tokens)
    actions = torch.tensor(paths, device=tokens.device, dtype=torch.long)
    previous = torch.full(
        (tokens.numel(),), model.registry.start, device=tokens.device, dtype=torch.long
    )
    for depth in range(record["depth_zero_based"]):
        column = actions[:, depth]
        for a in range(model.registry.slots + 1):
            rows = torch.where(column == a)[0]
            if len(rows):
                h = h.index_copy(
                    0, rows, model.execute(h[rows], a, memory, batches[rows], positions[rows])
                )
        rows = torch.where(column >= 0)[0]
        previous[rows] = column[rows]
    token = record["token_position"]
    state = Probe(
        h[token],
        memory,
        int(batches[token]),
        int(positions[token]),
        int(previous[token]),
        record["depth_zero_based"],
        record["chosen_action"],
        model.registry.fingerprint,
        (token // tokens.shape[1], token % tokens.shape[1]),
    )
    if (
        state.fingerprint() != record["state_fingerprint"]
        or canonical(record["action_prefix"]) != record["action_prefix_fingerprint"]
    ):
        raise ContinuationError("CF_STATE_REPLAY_MISMATCH")
    if actions[token, : state.depth].tolist() != record["action_prefix"]:
        raise ContinuationError("CF_PREFIX_REPLAY_MISMATCH")
    return state
