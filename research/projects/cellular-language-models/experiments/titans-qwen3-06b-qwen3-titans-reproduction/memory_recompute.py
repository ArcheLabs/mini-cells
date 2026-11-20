"""Recompute a complete causal memory call without truncating its gradients."""
import torch
from torch.utils.checkpoint import checkpoint


def recompute_memory(function, *inputs):
    # Reentrant checkpoint avoids saved-tensor hooks inside torch.func.grad.
    # Layer 0 receives frozen projections, so explicitly anchor backward.
    anchor = inputs[0].new_zeros((), requires_grad=True)
    return checkpoint(lambda _, *args: function(*args), anchor, *inputs,
                      use_reentrant=True)
