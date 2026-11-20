"""Run the Phase 1 capacity matrix and report retrieval and state changes."""
import argparse
import json

import torch
from torch.nn.functional import normalize

from memory import Memory, State


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equal-writes', action='store_true')
    equal = parser.parse_args().equal_writes
    arms = [('baseline', .5, .0001)]
    if not equal:
        arms += [('no_momentum', 0., .0001), ('no_forgetting', .5, 0.)]

    def checkpoints(count):
        return tuple(count * repeats for repeats in (1, 2, 4)) if equal else (1, 4, 16, 64)

    torch.set_num_threads(1)
    rows = []
    for arm, eta, alpha in arms:
        for seed in (42, 43, 44, 45, 46):
            torch.manual_seed(seed)
            memory = Memory()
            for count in (4, 8, 16, 32):
                keys, values = normalize(torch.randn(2, count, 512), dim=-1)
                negatives_for_query = normalize(torch.randn_like(keys), dim=-1)
                start = state = memory.create()
                for step in range(1, max(checkpoints(count)) + 1):
                    i = (step - 1) % count
                    _, state = memory.event(state, keys[i], values[i], keys[i], eta=eta, alpha=alpha)
                    state = State({k: v.detach() for k, v in state.weights.items()},
                                  {k: v.detach() for k, v in state.momentum.items()}, state.updates)
                    if step not in checkpoints(count):
                        continue
                    scores = normalize(memory.read(state, keys)) @ values.T
                    target = scores.diag()
                    negatives = scores.masked_fill(torch.eye(count, dtype=torch.bool), -torch.inf)
                    ranks = 1 + (negatives >= target[:, None]).sum(dim=1)
                    hard = normalize(.8 * keys + .6 * negatives_for_query)
                    threshold = .5
                    fpr = ((normalize(memory.read(state, hard)) @ values.T).max(dim=1).values >= threshold).float().mean()
                    rows.append(dict(seed=seed, arm=arm, hard_negative_fpr=float(fpr), associations=count, updates=step, recall_at_1=float((ranks == 1).float().mean()),
                                     mrr=float(ranks.float().reciprocal().mean()), cosine=float(target.mean()),
                                     margin=float((target - negatives.max(dim=1).values).mean()),
                                     fast_delta=float(sum((v - start.weights[k]).square().sum()
                                                          for k, v in state.weights.items()).sqrt().detach()),
                                     momentum_norm=float(sum(v.square().sum() for v in state.momentum.values()).sqrt())))
    import statistics
    summary = []
    for arm, _, _ in arms:
        for count in (4, 8, 16, 32):
            for updates in checkpoints(count):
                group = [row for row in rows if (row['arm'], row['associations'], row['updates']) == (arm,count,updates)]
                metrics = {}
                for name in ('recall_at_1', 'mrr', 'hard_negative_fpr'):
                    values = [r[name] for r in group]
                    metrics[name] = dict(mean=statistics.mean(values), sd=statistics.stdev(values))
                summary.append(dict(arm=arm, associations=count, updates=updates,
                                    writes_per_association=updates // count if equal else None, **metrics))
    print(json.dumps(dict(rows=rows, summary=summary), indent=2))


if __name__ == '__main__':
    main()
