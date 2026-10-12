"""Observe executed ATen GEMM shapes; no nominal Cell FLOP formulas.

SDPA math exposes its actual bmm calls; fused SDPA is reported as uncovered
and invalidates efficiency gates. This is GEMM arithmetic, not hardware FLOPs.
"""
from collections import Counter
from torch.utils._python_dispatch import TorchDispatchMode


class ExecutedMatmuls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.flops = 0
        self.calls = Counter()
        self.uncovered = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name in {'aten.mm.default', 'aten.addmm.default', 'aten.bmm.default'}:
            a, b = (args[1], args[2]) if name == 'aten.addmm.default' else args[:2]
            self.flops += 2 * a.numel() * b.shape[-1]
            self.calls[name] += 1
        elif 'scaled_dot_product' in name or 'flash_attention' in name:
            self.uncovered[name] += 1
        return func(*args, **(kwargs or {}))

    def metrics(self, tokens):
        return {'executed_gemm_flops_per_token': self.flops / tokens,
                'observed_gemm_calls': dict(self.calls),
                'uncovered_attention_calls': dict(self.uncovered),
                'flops_scope': 'observed mm/addmm/bmm shapes including math-SDPA attention; 2 FLOPs/MAC; '
                               'uncovered fused SDPA invalidates gates; excludes dispatch, normalization, '
                               'activation, softmax and backward; not total hardware FLOPs'}
