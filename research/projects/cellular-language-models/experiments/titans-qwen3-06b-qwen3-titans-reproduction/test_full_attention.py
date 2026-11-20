import unittest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from full_attention import exact_attention


class FullAttentionTest(unittest.TestCase):
    def test_causal_padding_gqa_and_gradients(self):
        generator = torch.Generator().manual_seed(42)
        for padded in (False, True):
            tensors = [torch.randn(2,h,19,8,dtype=torch.float64,generator=generator,
                                   requires_grad=True) for h in (4,2,2)]
            q,k,v = tensors
            pad = (torch.arange(19)[None] < torch.tensor([[15],[18]])) if padded else None
            mask = torch.ones(19,19,dtype=torch.bool).tril()
            if padded:
                mask = mask[None,None] & pad[:,None,None,:]
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),
                    v.repeat_interleave(2,1),attn_mask=mask,scale=.25)
            actual = exact_attention(*tensors,pad,scale=.25,block=7,force_blocks=True)
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-12)
            weight = torch.randn(q.shape,dtype=q.dtype,generator=generator)
            grads = lambda out: torch.autograd.grad((out*weight).sum(),tensors)
            for a,b in zip(grads(actual),grads(expected)):
                torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
