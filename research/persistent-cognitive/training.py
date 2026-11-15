"""One clean-slate outer update using the experimental retrofit."""
import math
import time

import torch
from torch.nn.functional import cross_entropy

from cell import Cell


def train_episode(model, retrofit, optimizer, ids, fingerprints, *, chunk=1):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze Qwen before training memory')
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError('Episode needs at least two tokens')
    optimizer.zero_grad(set_to_none=True)
    cell = Cell.create(retrofit.memories, fingerprints)
    diagnostics = {}
    started = time.perf_counter()
    outputs, updated = retrofit(model, cell, input_ids=ids[:, :-1], use_cache=False,
                          chunk=chunk, meta=True, diagnostics=diagnostics)
    loss = cross_entropy(outputs.logits.float().flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(retrofit.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    nll = float(loss.detach())
    return dict(loss=nll, perplexity=math.exp(min(nll, 700)),
                lr=optimizer.param_groups[0]['lr'], gradient_norm=float(gradient),
                memory_norm={k: sum(float(w.detach().square().sum()) for w in shard.fast)**.5
                             for k, shard in updated.shards.items()},
                momentum_norm={k: sum(float(w.detach().square().sum()) for w in shard.momentum)**.5
                               for k, shard in updated.shards.items()},
                address_trace={k: float(shard.address.detach().trace()) for k, shard in updated.shards.items()},
                forgetting={k: value['forgetting'] for k, value in diagnostics.items()},
                fusion_gate={k: value['gate'] for k, value in diagnostics.items()},
                tokens_per_second=(ids.numel()-1)/max(time.perf_counter()-started, 1e-9),
                kv_bytes=0)


@torch.no_grad()
def evaluate(model, retrofit, episodes, fingerprints, *, chunk=1):
    total = count = 0
    for ids in episodes:
        cell = Cell.create(retrofit.memories, fingerprints)
        out, _ = retrofit(model, cell, input_ids=ids[:, :-1], use_cache=False, chunk=chunk)
        total += float(cross_entropy(out.logits.float().flatten(0, 1),
                                     ids[:, 1:].flatten(), reduction='sum'))
        count += ids.numel()-1
    if not count:
        raise ValueError('Empty evaluation')
    return dict(loss=total/count, targets=count)


def fit(model, retrofit, optimizer, episodes, validation, fingerprints, checkpoint, *,
        dataset_sha256, steps, chunk=1, resume=False):
    """Deterministic indexable episodes; checkpoint at every committed update."""
    from pathlib import Path

    checkpoint = Path(checkpoint)
    identity = dict(fingerprints=fingerprints, dataset_sha256=dataset_sha256, chunk=chunk)
    history = []
    if resume:
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if state['identity'] != identity or state['step'] != len(state['history']):
            raise ValueError('Resume identity or cursor differs')
        retrofit.load_state_dict(state['memory'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        history = state['history']
    if not len(history) <= steps <= len(episodes):
        raise ValueError('Invalid training budget')
    for i in range(len(history), steps):
        history.append(train_episode(model, retrofit, optimizer, episodes[i], fingerprints, chunk=chunk))
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(identity=identity, memory=dict(retrofit.state_dict()),
                        optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                        step=i+1, history=history), temp)
        temp.replace(checkpoint)
    return dict(history=history, validation=evaluate(model, retrofit, validation, fingerprints, chunk=chunk))
