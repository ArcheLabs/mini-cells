import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention
from window_attention import WindowAttention


class WindowAttentionTest(unittest.TestCase):
    def test_output_and_gradients_match_folded_reference(self):
        gen = torch.Generator().manual_seed(42)
        q, k, v = [torch.randn(5, h, 12, 8, dtype=torch.float64, generator=gen,
                              requires_grad=True) for h in (4, 2, 2)]
        expanded = (q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1))
        weight = torch.randn(q.shape, dtype=q.dtype, generator=gen)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expected = scaled_dot_product_attention(*expanded, is_causal=True)
        actual, _ = WindowAttention(2)(*expanded)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        ref = torch.autograd.grad((expected*weight).sum(), (q, k, v), retain_graph=True)
        got = torch.autograd.grad((actual*weight).sum(), (q, k, v))
        for a, b in zip(got, ref):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
