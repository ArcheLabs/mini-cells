"""All-action terminal Q, rerouted suffix; direct expected centered cost, no pairwise BCE."""

import copy
import torch
from torch.nn import functional as F
from minicells_r3c_moe_bridge.data import train_batches
from .evaluation import reference, trust, within


def add_work(total, work):
    for k, v in work.items():
        total[k] = total.get(k, 0) + v


def difference(after, before):
    return {k: v - before.get(k, 0) for k, v in after.items()}


@torch.no_grad()
def collect(tr, data):
    m = tr.model
    # Peek the next Train batch without consuming the main stream or looking at evaluation labels.
    rng = copy.deepcopy(tr.data_rng)
    batches = train_batches(data, tr.c["engine"], rng, next(m.parameters()).device)
    route_rng = torch.Generator()
    route_rng.set_state(tr.route_rng.get_state())
    visited = [
        (m(x, keep_trace=True, epsilon=tr.epsilon(), route_rng=route_rng), y) for x, y in batches
    ]
    order = torch.randperm(m.steps, generator=tr.cf_rng).tolist()
    records = []
    for i in range(tr.c["training"]["states"]):
        t = order[i % len(order)]
        b = int(torch.randint(len(visited), (1,), generator=tr.cf_rng))
        out, y = visited[b]
        j = int(torch.randint(y.numel(), (1,), generator=tr.cf_rng))
        state = out.trace[t]["state"][j : j + 1]
        prev = out.trace[t]["previous"][j : j + 1]
        actions = torch.arange(6 if m.graph else 15, device=state.device)
        h = state.expand(len(actions), -1)
        p = prev.expand(len(actions))
        h, p = m.transition(h, t, p, actions)
        for suffix in range(t + 1, m.steps):
            a, _ = m.select(h, suffix, p)
            h, p = m.transition(h, suffix, p, a)
        q = F.cross_entropy(m.head(h), y.reshape(-1)[j].expand(len(actions)), reduction="none")
        if not torch.isfinite(q).all():
            raise FloatingPointError("NONFINITE_COUNTERFACTUAL_Q")
        records.append(
            {
                "h": state.detach(),
                "previous": prev,
                "stage": t,
                "values": q.detach(),
                "target": y.reshape(-1)[j : j + 1],
            }
        )
    return records


def optimize(tr, data):
    m = tr.model
    c = tr.c
    start = copy.deepcopy(m.work)
    records = collect(tr, data)
    candidate = copy.deepcopy(m)
    for p in candidate.parameters():
        p.requires_grad_(False)
    for p in candidate.connection.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        candidate.connection.parameters(),
        lr=c["training"]["connection_lr"],
        weight_decay=c["training"]["connection_weight_decay"],
    )
    losses = []
    for _ in range(c["training"]["policy_steps"]):
        opt.zero_grad(set_to_none=True)
        terms = []
        for r in records:
            q, _, _ = candidate.scores(r["h"], r["stage"], r["previous"])
            costs = r["values"] - r["values"].mean()
            terms.append((q.softmax(-1) * costs).sum())
        loss = torch.stack(terms).mean()
        loss.backward()
        grads = [p.grad for p in candidate.connection.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("INVALID_POLICY_GRADIENT")
        torch.nn.utils.clip_grad_norm_(
            candidate.connection.parameters(), c["training"]["connection_clip"]
        )
        opt.step()
        losses.append(float(loss.detach()))
    work = difference(m.work, start)
    add_work(work, difference(candidate.work, m.work))
    refs = reference(m, c, data)
    old_params = {n: p.detach().clone() for n, p in m.connection.named_parameters()}
    new_params = {n: p.detach().clone() for n, p in candidate.connection.named_parameters()}
    attempts = []
    chosen = None
    for scale in c["proposal"]["backtrack"]:
        with torch.no_grad():
            for n, p in candidate.connection.named_parameters():
                p.copy_(old_params[n] + scale * (new_params[n] - old_params[n]))
        old_work = copy.deepcopy(m.work)
        new_work = copy.deepcopy(candidate.work)
        metrics = trust(m, candidate, refs)
        add_work(work, difference(m.work, old_work))
        add_work(work, difference(candidate.work, new_work))
        attempts.append({"scale": scale, **metrics, "passed": within(metrics, c)})
        if within(metrics, c):
            chosen = scale
            break
    for p in candidate.parameters():
        p.requires_grad_(False)
    for p in candidate.cells.parameters():
        p.requires_grad_(True)
    # Probe forwards are additional work, not normal main-stream updates.
    m.work = start
    return candidate if chosen is not None else None, {
        "losses": losses,
        "attempts": attempts,
        "chosen_scale": chosen,
        "counterfactual_states": len(records),
        "actions_per_state": 6 if m.graph else 15,
        "counterfactual_suffix": "CURRENT_POLICY_REROUTED",
        "work": work,
        "policy_optimizer_updates": len(losses),
        "proposal_optimizer_final_state": opt.state_dict(),
    }
