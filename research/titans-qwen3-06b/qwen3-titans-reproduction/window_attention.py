"""Bound the SDPA workspace without changing the upstream window layout."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


class WindowAttention(nn.Module):
    def __init__(self, windows_per_call=2):
        super().__init__()
        self.windows_per_call = windows_per_call
        self.backend = None

    def forward(self, q, k, v):
        # Upstream folds independent causal windows into this batch dimension
        # and supplies repeated KV heads. No masks or prefix tokens on this path.
        sample = tuple(t[:self.windows_per_call] for t in (q, k, v))
        backend = SDPBackend.MATH
        if q.is_cuda:
            params = torch.backends.cuda.SDPAParams(*sample, None, 0., True, False)
            if torch.backends.cuda.can_use_flash_attention(params):
                backend = SDPBackend.FLASH_ATTENTION
            elif torch.backends.cuda.can_use_efficient_attention(params):
                backend = SDPBackend.EFFICIENT_ATTENTION
        self.backend = backend.name
        if q.is_cuda and torch.is_grad_enabled():
            print(dict(sdpa_backend=self.backend, folded_q=list(q.shape),
                       windows_per_call=self.windows_per_call), flush=True)

        def attend(query, key, value):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(query, key, value, is_causal=True)

        outputs = []
        for start in range(0, q.shape[0], self.windows_per_call):
            tensors = tuple(t[start:start+self.windows_per_call] for t in (q, k, v))
            if backend == SDPBackend.MATH and torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
                # A split alone still retains every window's softmax for backward.
                out = checkpoint(attend, *tensors, use_reentrant=False)
            else:
                out = attend(*tensors)
            outputs.append(out)
        return torch.cat(outputs, dim=0), None
