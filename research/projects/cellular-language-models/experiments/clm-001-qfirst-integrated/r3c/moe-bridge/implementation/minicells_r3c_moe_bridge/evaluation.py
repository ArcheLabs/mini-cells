"""Target-free inference interventions and separate, explicitly labelled suffix estimands."""

import torch
from torch.nn import functional as F
from clm001_r3b.instrumentation import isolated
from clm001_r3b.analysis_runner import fingerprint


def replay_plan(out):
    return [
        {"indices": s.indices.detach().clone(), "weights": s.weights.detach().clone()}
        for s in out.stages
    ]


@torch.no_grad()
def evaluate(model, windows, seed, *, intervention=None):
    rows = {}
    rng = torch.Generator().manual_seed(seed)
    with isolated(model):
        model.eval()
        for story, x, y in windows:
            if story in rows:
                raise ValueError("Story windows must be aggregated explicitly, not overwritten")
            out = model(x, route_rng=rng, intervention=intervention)
            ce = F.cross_entropy(
                out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1), reduction="none"
            )
            rows[story] = {
                "mean_nll": float(ce.mean()),
                "tokens": y.numel(),
                "cell_calls_per_token": 8,
                "context_calls_per_token": 1,
            }
    return rows


def permutation_intervention(permutation):
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("Cell permutation must be bijective")

    def intervene(stage, h, logits, indices, weights):
        mapping = torch.tensor(permutation, device=indices.device)
        return mapping[indices], weights

    return intervene


@torch.no_grad()
def cell_values(model, x, y, stage, seed, *, suffix):
    if suffix not in ("FIXED_SUFFIX", "REPLANNED_SUFFIX"):
        raise ValueError("suffix estimand required")
    with isolated(model):
        model.eval()
        rng = torch.Generator().manual_seed(seed)
        original = model(x, route_rng=rng)
        plan = replay_plan(original)
        values = []
        for candidate in range(model.c["routing"]["cells"]):

            def intervene(t, h, logits, indices, weights):
                if t < stage or t > stage and suffix == "FIXED_SUFFIX":
                    return plan[t]["indices"], plan[t]["weights"]
                if t != stage:
                    return indices, weights
                partner = plan[t]["indices"][:, 1]
                partner = torch.where(partner == candidate, plan[t]["indices"][:, 0], partner)
                forced = torch.stack((torch.full_like(partner, candidate), partner), -1)
                weight = (
                    torch.full_like(weights, 0.5)
                    if model.arm in ("A", "B")
                    else (logits.gather(1, forced) / model.c["routing"]["temperature"]).softmax(-1)
                )
                return forced, weight

            out = model(x, route_rng=torch.Generator().manual_seed(seed), intervention=intervene)
            values.append(
                F.cross_entropy(
                    out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1), reduction="none"
                )
            )
        return {
            "values": torch.stack(values, -1).cpu().numpy(),
            "suffix": suffix,
            "stage": stage,
            "budget": "4x2_cells_plus_one_context",
            "pair_rule": "replace slot0; retain slot1 unless duplicate, then use baseline slot0; recompute selected weights",
            "selection_uses_target": False,
            "target_used_for_read_only_scoring": True,
            "model_sha": fingerprint(model.state_dict()),
        }
