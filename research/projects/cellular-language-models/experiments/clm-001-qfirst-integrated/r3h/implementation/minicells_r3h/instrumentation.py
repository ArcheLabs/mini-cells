"""GPU-resident counters during each update; one CPU transfer at update completion."""

import torch


class Exposure:
    def __init__(self, m):
        self.m = m
        d = next(m.parameters()).device
        self.cells = torch.zeros((m.steps, 6), dtype=torch.long, device=d)
        self.actions = torch.zeros((m.steps, 6 if m.graph else 15), dtype=torch.long, device=d)
        self.edges = torch.zeros((6, 6), dtype=torch.long, device=d)
        self.matches = torch.zeros((m.steps, 2), dtype=torch.long, device=d)
        self.probability = torch.zeros((m.steps, 2), dtype=torch.double, device=d)

    def observe(self, t, previous, action, hard, p):
        self.actions[t] += torch.bincount(action, minlength=self.actions.shape[-1])
        ids = action if self.m.graph else self.m.connection.pairs[action].flatten()
        self.cells[t] += torch.bincount(ids, minlength=6)
        self.matches[t, 0] += (action == hard).sum()
        self.matches[t, 1] += len(action)
        self.probability[t, 0] += p.gather(-1, hard[:, None]).sum()
        self.probability[t, 1] += -(p * p.clamp_min(1e-300).log()).sum() / torch.log(
            torch.tensor(p.shape[-1], device=p.device, dtype=p.dtype)
        )
        if self.m.graph and t:
            self.edges += torch.bincount(previous * 6 + action, minlength=36).reshape(6, 6)

    def result(self):
        return dict(
            stage_cells=self.cells.cpu().tolist(),
            stage_actions=self.actions.cpu().tolist(),
            transitions=self.edges.cpu().tolist(),
            matches=self.matches.cpu().tolist(),
            probability_sums=self.probability.cpu().tolist(),
        )
