"""Biased subset credit, with detached h/g/candidate operators and no inference mixture."""

import torch
from .training_errors import CreditFailure


def candidates(model, event, generator):
    s = event.state
    legal = model.registry.legal(
        s.depth, model.config["budget"], model.config["minimum_actions"], s.hidden.device
    )
    ids = [i for i in range(model.registry.slots) if bool(legal[i])]
    if s.chosen not in ids:
        raise CreditFailure("CTX/STOP or unaccepted Cell in competition")
    pool = [i for i in ids if i != s.chosen]
    number = model.experiment_config["competition"]["candidate_cells"] - 1
    if len(pool) < number:
        raise CreditFailure("insufficient distinct legal Cells")
    order = torch.randperm(len(pool), generator=generator, device=s.hidden.device).tolist()[:number]
    return [s.chosen] + [pool[i] for i in order], [1.0] + [number / len(pool)] * number


def subset_loss(logits, local_costs, temperature=1.0):
    if not torch.isfinite(logits).all() or not torch.isfinite(local_costs).all():
        raise CreditFailure("nonfinite competition logits/credit")
    return ((logits / temperature).softmax(-1) * local_costs.detach()).sum(-1).mean()


def loss(model, events, generator):
    terms = []
    records = []
    calls = 0
    for event in events:
        s = event.state
        g = event.gradient
        if g is None or not torch.isfinite(g).all():
            raise CreditFailure("missing/nonfinite actual post-state gradient")
        ids, inclusion = candidates(model, event, generator)
        with torch.no_grad():
            # Selected delta is the actual operator result. Only K-1 additional Cells execute.
            selected = (event.post - s.hidden).detach()
            deltas = [selected]
            for action in ids[1:]:
                deltas.append(
                    (
                        model.execute(
                            s.hidden[None],
                            action,
                            s.memory,
                            torch.tensor([s.batch], device=s.hidden.device),
                            torch.tensor([s.position], device=s.hidden.device),
                        )[0]
                        - s.hidden
                    ).detach()
                )
                calls += 1
            delta = torch.stack(deltas)
            costs = ((delta - selected) * g.detach()).sum(-1)
        legal = model.registry.legal(
            s.depth, model.config["budget"], model.config["minimum_actions"], s.hidden.device
        )
        q, _ = model.controller(
            s.hidden[None].detach(),
            torch.tensor([s.previous], device=s.hidden.device),
            s.depth,
            legal,
            model.registry.version,
            detach_state=True,
        )
        term = subset_loss(-q[0, ids], costs, model.experiment_config["competition"]["temperature"])
        terms.append(term)
        records.append(
            {
                "depth": s.depth,
                "position": s.position,
                "previous": s.previous,
                "ctx_seen": event.ctx_seen,
                "chosen": s.chosen,
                "group_feature": event.group_feature,
                "candidates": ids,
                "candidate_inclusion": inclusion,
                "event_conditional_inclusion": event.inclusion,
                "local_costs": costs.cpu().tolist(),
                "gradient_norm": float(g.norm()),
                "state_sha": s.fingerprint(),
                "cell_entropy": float(
                    (-(-q[0, ids]).softmax(-1) * (-q[0, ids]).log_softmax(-1)).sum().detach()
                ),
            }
        )
    zero = next(model.parameters()).new_zeros(())
    return (torch.stack(terms).mean() if terms else zero), records, calls


def vector(grads, parameters):
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, parameters)
        ]
    )
