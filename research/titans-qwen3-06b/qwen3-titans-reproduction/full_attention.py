"""Exact, cache-free Qwen full attention for the reproduction workload."""
import warnings

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

REPORTED = set()


def exact_attention(q, k, v, padding=None, *, scale=None, block=128, force_blocks=False):
    # Probe with concrete tensors, including the actual layout and GQA mode.
    source_shapes = [list(t.shape) for t in (q, k, v)]
    if q.shape[-2] != k.shape[-2]:
        raise ValueError('Expected a cache-free causal sequence')
    groups = q.shape[1] // k.shape[1]
    k, v = (t.repeat_interleave(groups, dim=1) if groups != 1 else t for t in (k, v))
    backend = None
    results = {}
    key = (q.device, q.dtype, tuple(q.shape), q.stride(), k.stride(), v.stride(), padding is not None)
    if q.is_cuda and not force_blocks:
        params = torch.backends.cuda.SDPAParams(q, k, v, None, 0., True, False)
        for name, probe in (
            ('FLASH_ATTENTION', torch.backends.cuda.can_use_flash_attention),
            ('EFFICIENT_ATTENTION', torch.backends.cuda.can_use_efficient_attention),
            ('CUDNN_ATTENTION', torch.backends.cuda.can_use_cudnn_attention),
        ):
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always')
                supported = probe(params, debug=key not in REPORTED)
            results[name] = dict(available=supported, reasons=[str(w.message) for w in messages] or
                                 ([] if supported else ['See PyTorch native SDPA debug messages in run.log']))
            if supported and backend is None and padding is None:
                backend = getattr(SDPBackend, name)
        if key not in REPORTED:
            REPORTED.add(key)
            print(dict(full_attention_backend=backend.name if backend else 'QUERY_BLOCK_MATH',
                       probes=results, qkv_shapes=source_shapes, qkv_strides=[t.stride() for t in (q,k,v)],
                       dtype=str(q.dtype), mask=None if padding is None else
                       dict(shape=list(padding.shape), dtype=str(padding.dtype), stride=padding.stride(),
                            layout=str(padding.layout))), flush=True)

    if backend is not None:
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

    def chunk(query, keys, values, start):
        query_positions = torch.arange(start, start + query.shape[-2], device=q.device)
        key_positions = torch.arange(keys.shape[-2], device=q.device)
        mask = key_positions <= query_positions[:, None]
        if padding is not None:
            mask = mask[None, None] & padding[:, None, None, :]
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=scale)

    outputs = []
    for start in range(0, q.shape[-2], block):
        query = q[..., start:start+block, :]
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
            out = checkpoint(chunk, query, k, v, start, use_reentrant=False)
        else:
            out = chunk(query, k, v, start)
        outputs.append(out)
    return torch.cat(outputs, dim=-2)


def qwen_attention(module, q, k, v, attention_mask, dropout=0., scaling=None, **kwargs):
    if dropout or (attention_mask is not None and attention_mask.ndim != 2):
        raise ValueError('Expected zero dropout and no mask or a compact padding mask')
    return exact_attention(q, k, v, attention_mask, scale=scaling).transpose(1, 2).contiguous(), None


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q,k,v = [torch.randn(1,16384,h,128,device='cuda',dtype=dtype).transpose(1,2).requires_grad_()
             for h in (16,8,8)]
    result = dict(status='RUNNING', gpu=torch.cuda.get_device_name(), dtype=str(dtype))
    for phase, operation in [('forward', lambda: exact_attention(q,k,v)),
                             ('backward', lambda: output.float().square().mean().backward())]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            value = operation()
            torch.cuda.synchronize()
        finally:
            result[phase] = dict(peak_bytes=torch.cuda.max_memory_allocated(),
                                 seconds=time.monotonic()-start)
            args.output.write_text(json.dumps(result, indent=2))
        if phase == 'forward':
            output = value
    if not all(torch.isfinite(t).all() for t in (output,q.grad,k.grad,v.grad)):
        raise FloatingPointError('Nonfinite attention output or gradient')
    result['status'] = 'PASS'
    args.output.write_text(json.dumps(result, indent=2))
