"""Frozen R2 behavior streams plus independent competition sampling."""

import torch
from clm001_r2.sampling import distribution, inverse_cdf, proposals, streams as old_streams

__all__ = ["distribution", "inverse_cdf", "proposals", "streams"]


def streams(seed, device="cpu"):
    result = old_streams(seed, device)
    result["competition"] = torch.Generator(device=device).manual_seed(seed + 19001)
    return result
