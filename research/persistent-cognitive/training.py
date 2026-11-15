"""One clean-slate outer update using the experimental retrofit."""
import torch
from torch.nn.functional import cross_entropy

from cell import Cell


def train_episode(model, retrofit, optimizer, ids, fingerprints, *, chunk=1):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze Qwen before training memory')
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError('Episode needs at least two tokens')
    optimizer.zero_grad(set_to_none=True)
    cell = Cell.create(retrofit.memories, fingerprints)
    outputs, _ = retrofit(model, cell, input_ids=ids[:, :-1], use_cache=False,
                          chunk=chunk, meta=True)
    loss = cross_entropy(outputs.logits.float().flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(retrofit.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())
