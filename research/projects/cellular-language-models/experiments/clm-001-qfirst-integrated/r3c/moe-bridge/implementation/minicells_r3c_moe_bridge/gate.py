"""Shared state/stage scorer; no previous-action, STOP, CTX or target input."""

import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self, c):
        super().__init__()
        m = c["model"]
        r = c["routing"]
        self.stage = nn.Embedding(r["stages"], m["step_width"])
        self.net = nn.Sequential(
            nn.Linear(m["width"] + m["step_width"], m["router_hidden"]),
            nn.GELU(),
            nn.Linear(m["router_hidden"], r["cells"]),
        )

    def forward(self, h, stage):
        e = self.stage.weight[stage].expand(h.shape[0], -1)
        return self.net(torch.cat((h, e), -1))


def controlled_indices(c, arm, positions, rng):
    n = c["routing"]["cells"]
    count = positions.numel()
    device = positions.device
    if arm == "A":
        first = (positions + 2 * c["stage_index"]) % n
        return torch.stack((first, (first + c["routing"]["fixed_pair_offset"]) % n), -1)
    if arm != "B" or rng is None or rng.device.type != "cpu":
        raise ValueError("B requires its independent CPU route generator")
    cells = torch.randperm(n, generator=rng)
    order = torch.randperm(count, generator=rng) % n
    offset = int(torch.randint(1, n, (1,), generator=rng))
    return torch.stack((cells[order], cells[(order + offset) % n]), -1).to(device)
