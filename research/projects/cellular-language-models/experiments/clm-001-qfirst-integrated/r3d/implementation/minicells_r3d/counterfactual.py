"""Train-only sampled-candidate preferences; suffixes are always rerouted."""

import itertools
import torch
from torch.nn import functional as F


@torch.no_grad()
def collect(model, visited, rng, count, mode):
    """Visited contains only Train states/labels; select states/actions before scoring labels."""
    if mode not in ("long", "short"):
        raise ValueError("explicit credit horizon required")
    # Stratification covers all four stages with an independent private RNG.
    order = torch.randperm(model.c["routing"]["stages"], generator=rng).tolist()
    records = []
    work = {
        "counterfactual_cell_rows": 0,
        "counterfactual_gate_rows": 0,
        "counterfactual_head_rows": 0,
    }
    for i in range(count):
        t = order[i % len(order)]
        out, y = visited[int(torch.randint(len(visited), (1,), generator=rng))]
        row = int(torch.randint(y.numel(), (1,), generator=rng))
        h = out.stages[t].inputs[row : row + 1].detach()
        z = model.gate(h, t)
        current = int(model.connection.choose(model.norm(h), t, z)[0])
        base = int(model.connection.base_action(z))
        explore = int(torch.randint(len(model.connection.pairs), (1,), generator=rng))
        candidates = list(dict.fromkeys((current, base, explore)))
        values = []
        for action in candidates:
            state = model.advance(h, t, torch.tensor([action], device=h.device))
            steps = 1
            if mode == "long":
                for suffix in range(t + 1, model.c["routing"]["stages"]):
                    state = model.advance(state, suffix)
                    steps += 1
            # No label enters connection or candidate selection.
            values.append(F.cross_entropy(model.head(state), y.reshape(-1)[row : row + 1]).detach())
            work["counterfactual_cell_rows"] += 2 * steps
            work["counterfactual_gate_rows"] += steps
            work["counterfactual_head_rows"] += 1
        records.append(
            {
                "state": model.norm(h).detach(),
                "stage": t,
                "base_logits": z.detach(),
                "candidates": candidates,
                "values": torch.stack(values),
                "horizon": mode,
                "suffix": "REROUTED_CURRENT_POLICY",
            }
        )
        work["counterfactual_gate_rows"] += 1
    return records, work


def preference_loss(model, records, temperature):
    losses = []
    correct = []
    gaps = []
    for r in records:
        q, _ = model.connection(r["state"], r["stage"], r["base_logits"])
        for a, b in itertools.combinations(range(len(r["candidates"])), 2):
            delta = q[0, r["candidates"][a]] - q[0, r["candidates"][b]]
            difference = r["values"][b] - r["values"][a]
            target = torch.sigmoid(difference / temperature).detach()
            losses.append(F.binary_cross_entropy_with_logits(delta, target))
            correct.append(((delta.detach() > 0) == (difference > 0)).float())
            gaps.append(difference.abs())
    if not losses:
        return None, {"scored_pairs": 0, "ranking_accuracy": None, "mean_candidate_gap": None}
    return torch.stack(losses).mean(), {
        "scored_pairs": len(losses),
        "ranking_accuracy": float(torch.stack(correct).mean()),
        "mean_candidate_gap": float(torch.stack(gaps).mean()),
        "estimand": "SAMPLED_CANDIDATE_PREFERENCE_NOT_EXACT_ALL_ACTION_Q",
    }
