"""Fixed dedicated streams; no labels or output-dependent search expansion."""

import torch
from minicells_r3e.common import canonical


def directions(connection, architecture, seed, rms):
    original = connection.state_dict()
    parameters = dict(connection.named_parameters())
    result = []
    for family, count in [("RANDOM", 8), ("STRUCTURED", 4)]:
        for index in range(count):
            key = int(canonical(dict(seed=seed, family=family, index=index))[:15], 16)
            rng = torch.Generator(device="cpu").manual_seed(key)
            direction = {name: torch.zeros_like(value) for name, value in original.items()}
            for name, param in parameters.items():
                value = torch.randn(param.shape, generator=rng, dtype=param.dtype)
                if family == "STRUCTURED":
                    value.zero_()
                    if name == "linear.weight":
                        row = torch.randn(param.shape[0], generator=rng, dtype=param.dtype)
                        row -= row.mean()
                        if architecture == "A":
                            value[:, param.shape[1] - 4 + index] = row
                        else:
                            value[:, param.shape[1] - 8 + 2 * index] = row
                            value[:, param.shape[1] - 15 + (index % 7)] = row
                scale = value.double().square().sum().sqrt()
                if scale > 0:
                    value = value * (rms / scale.to(value))
                direction[name] = value.to(param.device)
            result.append(dict(family=family, index=index, seed=key, direction=direction))
    return result
