"""Run the Phase 1 capacity matrix and report retrieval and state changes."""
import json

import torch
from torch.nn.functional import normalize

from memory import Memory, State


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    memory = Memory()
    rows = []
    for count in (4, 8, 16, 32):
        keys, values = normalize(torch.randn(2, count, 512), dim=-1)
        start = state = memory.create()
        for step in range(1, 65):
            i = (step - 1) % count
            _, state = memory.event(state, keys[i], values[i], keys[i])
            state = State({k: v.detach() for k, v in state.weights.items()},
                          {k: v.detach() for k, v in state.momentum.items()}, state.updates)
            if step not in (1, 4, 16, 64):
                continue
            scores = normalize(memory.read(state, keys)) @ values.T
            target = scores.diag()
            negatives = scores.masked_fill(torch.eye(count, dtype=torch.bool), -torch.inf)
            ranks = 1 + (negatives >= target[:, None]).sum(dim=1)
            rows.append(dict(associations=count, updates=step, recall_at_1=float((ranks == 1).float().mean()),
                             mrr=float(ranks.float().reciprocal().mean()), cosine=float(target.mean()),
                             margin=float((target - negatives.max(dim=1).values).mean()),
                             fast_delta=float(sum((v - start.weights[k]).square().sum()
                                                  for k, v in state.weights.items()).sqrt().detach()),
                             momentum_norm=float(sum(v.square().sum() for v in state.momentum.values()).sqrt())))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
