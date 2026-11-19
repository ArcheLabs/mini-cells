"""Serial reference for chunk-start gradients followed by tokenwise recurrence."""
import torch
from torch.nn import functional as F


def chunk_reference(memory, cell, keys, values, memory_forward, transition):
    """Use frozen chunk-start weights for gradients, but carry every transition."""
    from dataclasses import replace

    keys = F.normalize(keys, dim=-1)
    start = {name: tensor.detach().requires_grad_(True) for name, tensor in cell.fast.items()}
    fast, momentum = cell.fast, cell.momentum
    for key, value in zip(keys.split(1, 1), values.split(1, 1)):
        with torch.enable_grad():
            loss = (memory_forward(key, start) - value).square().sum()
            gradients = torch.autograd.grad(loss, tuple(start.values()))
        with torch.no_grad():
            eta, theta, alpha = memory.controls(key[:, 0]).sigmoid().unbind(-1)
            fast, momentum = transition(fast, momentum, gradients, eta,
                                        theta * memory.max_lr, alpha * memory.max_forget)
    count = keys.shape[1]
    return replace(cell, fast=fast, momentum=momentum, update_step=cell.update_step + count,
                   metadata={**cell.metadata, 'revision': cell.metadata['revision'] + count})
