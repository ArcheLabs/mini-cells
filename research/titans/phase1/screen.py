"""Paired, unified Gaussian capacity and native M3 update screening."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

P4 = Path(__file__).resolve().parents[2] / (
    'projects/cellular-language-models/experiments/p4-peer-titans-collaboration')
sys.path.insert(0, str(P4 / 'm1.1-capacity'))
from smoke import generate, initialize, read
from memory import memory_forward, transition
from chunk_reference import chunk_reference

CONFIG = dict(loads=[8, 16, 32, 64], seeds=[41036, 41037, 41038],
              episodes=20, exposures=128, geometry='gaussian', threshold=.8,
              initialization_seed=41035, bootstrap=10000)


def run(n, seed, method, width):
    memory = initialize(CONFIG['initialization_seed'])
    keys, candidates, labels, _, hard, _ = generate(n, seed, 'gaussian', memory.dim)
    values = candidates[labels] * math.sqrt(memory.dim)
    cell = memory.create(mode='online_adapt')
    initial = cell.detached()
    rng = torch.Generator().manual_seed(seed + 17)
    start = time.perf_counter()
    for _ in range(CONFIG['exposures']):
        for ids in torch.randperm(n, generator=rng).split(width):
            if method == 'chunk_start_reference4':
                cell = chunk_reference(memory, cell, keys[ids][None], values[ids][None],
                                       memory_forward, transition)
            else:
                cell, _ = memory.learn(keys[ids][None], values[ids][None], cell, 'online_adapt')
    seconds = time.perf_counter() - start
    scores = F.normalize(read(memory, cell, keys), dim=-1) @ candidates.T
    correct = scores.gather(1, labels[:, None]).flatten()
    other = scores.scatter(1, labels[:, None], -torch.inf).amax(1)
    ranks = (scores.argsort(descending=True, stable=True) == labels[:, None]).long().argmax(1) + 1
    hard_scores = (F.normalize(read(memory, cell, hard), dim=-1) @ candidates.T).amax(1)
    norm = lambda group: sum(float(x.detach().square().sum()) for x in group.values()) ** .5
    return dict(method=method, width=width, recall=float((ranks == 1).float().mean()),
                mrr=float(ranks.float().reciprocal().mean()), margin=float((correct-other).mean()),
                hard_fpr=float((hard_scores >= CONFIG['threshold']).float().mean()),
                fast_norm=norm(cell.fast), momentum_norm=norm(cell.momentum),
                delta=norm({k: v-initial.fast[k] for k, v in cell.fast.items()}),
                presentations=n*CONFIG['exposures'], gradients=cell.update_step,
                transitions=cell.update_step, controls=cell.update_step,
                outer_optimizer_updates=0, update_seconds=seconds)


def main(output):
    torch.set_num_threads(1)
    output.mkdir()
    (output / 'config.json').write_text(json.dumps(CONFIG, indent=2))
    with (output / 'episodes.jsonl').open('w', buffering=1) as stream:
        for n in CONFIG['loads']:
            for seed in CONFIG['seeds']:
                for episode in range(CONFIG['episodes']):
                    data_seed = seed * 100 + episode
                    for method, width in [('batch', n), ('chunk4', 4), ('chunk8', 8), ('sequential', 1), ('chunk_start_reference4', 4)]:
                        try:
                            row = run(n, data_seed, method, width)
                        except Exception as error:
                            row = dict(method=method, status='FAIL', error=repr(error))
                        stream.write(json.dumps(dict(n=n, seed=seed, episode=episode, **row),
                                                allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
