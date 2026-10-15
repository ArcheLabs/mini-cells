"""Independent streams and inverse-CDF CRN; candidates never share consumed generators."""

import torch


def distribution(cost, legal, epsilon, temperature, kind="sampled"):
    if not legal.any():
        raise ValueError("no legal actions at a nonterminal decision")
    if kind == "constant":
        choice = torch.where(legal)[0][0].expand(cost.shape[:-1])
        return torch.nn.functional.one_hot(choice, cost.shape[-1]).to(cost.dtype)
    if kind == "uniform":
        return legal.to(cost.dtype).expand_as(cost) / legal.sum()
    if kind == "greedy":
        choice = cost.masked_fill(~legal, float("inf")).argmin(-1)
        return torch.nn.functional.one_hot(choice, cost.shape[-1]).to(cost.dtype)
    if kind not in ("sampled", "soft"):
        raise ValueError("unknown suffix policy")
    mix = epsilon if kind == "sampled" else 0.0
    return (1 - mix) * (-cost / temperature).masked_fill(~legal, float("-inf")).softmax(
        -1
    ) + mix * legal.to(cost.dtype) / legal.sum()


def inverse_cdf(probabilities, uniform):
    cdf = probabilities.cumsum(-1)
    chosen = (uniform[..., None] >= cdf).sum(-1).clamp_max(probabilities.shape[-1] - 1)
    if not torch.all(probabilities.gather(-1, chosen[..., None]) > 0):
        raise ValueError("invalid inverse-CDF support")
    return chosen


def streams(seed, device="cpu"):
    return {
        name: torch.Generator(device=device).manual_seed(seed + offset)
        for name, offset in (
            ("live", 1),
            ("proposal", 2),
            ("suffix", 3),
            ("shadow", 4),
            ("evaluation", 5),
        )
    }


def proposals(legal, chosen, stop, extra, generator):
    ids = torch.where(legal)[0].tolist()
    if chosen not in ids:
        raise ValueError("actual action was not legal")
    forced = [chosen] + ([stop] if stop in ids and stop != chosen else [])
    pool = [a for a in ids if a not in forced]
    number = min(extra, len(pool))
    order = (
        torch.randperm(len(pool), generator=generator, device=legal.device).cpu().tolist()[:number]
    )
    return [(a, 1.0) for a in forced] + [(pool[i], number / len(pool)) for i in order]
