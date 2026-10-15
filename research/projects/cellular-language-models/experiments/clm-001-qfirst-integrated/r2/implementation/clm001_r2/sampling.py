"""Shared tie contract; R1 RNG and candidate inclusion helpers reused without mutation."""

import torch
from clm001.sampling import distribution as raw_distribution, inverse_cdf, streams, proposals

__all__ = ["distribution", "inverse_cdf", "streams", "proposals"]


def distribution(cost, legal, epsilon, temperature, kind="sampled"):
    if kind == "greedy" and legal[-1]:
        masked = cost.masked_fill(~legal, float("inf"))
        choice = masked.argmin(-1)
        choice = torch.where(
            masked[..., -1] == masked.min(-1).values,
            torch.full_like(choice, cost.shape[-1] - 1),
            choice,
        )
        return torch.nn.functional.one_hot(choice, cost.shape[-1]).to(cost)
    return raw_distribution(cost, legal, epsilon, temperature, kind)
