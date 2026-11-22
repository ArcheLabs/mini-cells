"""Diagnostic-only companion to FLA 0.2.2 fused recurrent gated delta rule.

The layout and reductions follow fla/ops/gated_delta_rule/fused_recurrent.py
(Copyright 2023–2025 Songlin Yang, Yu Zhang, MIT). The diagnostic additionally
captures the pre-update prediction and residual. See the experiment license notice. It supports the fixed-length
MoM path, where Q/K were normalized upstream and beta is a scalar per head.
"""
import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.jit(do_not_specialize=["T"])
def residual_kernel(Q, Kptr, Vptr, G, Beta, H0, Y, H1, P, E, T, scale,
                    heads: tl.constexpr, keys: tl.constexpr, values: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, HAS_STATE: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // heads, bh % heads
    kid = tl.arange(0, BK)
    vid = tile * BV + tl.arange(0, BV)
    state_pos = bh * keys * values + kid[:, None] * values + vid[None, :]
    state_mask = (kid[:, None] < keys) & (vid[None, :] < values)
    state = tl.zeros((BK, BV), tl.float32)
    if HAS_STATE:
        state = tl.load(H0 + state_pos, state_mask, other=0).to(tl.float32)
    for t in range(T):
        token = (batch * T + t) * heads + head
        q = tl.load(Q + token * keys + kid, kid < keys, other=0).to(tl.float32) * scale
        k = tl.load(Kptr + token * keys + kid, kid < keys, other=0).to(tl.float32)
        vpos = token * values + vid
        v = tl.load(Vptr + vpos, vid < values, other=0).to(tl.float32)
        state = state * exp(tl.load(G + token).to(tl.float32))
        prediction = tl.sum(state * k[:, None], 0)
        error = v - prediction
        delta = error * tl.load(Beta + token).to(tl.float32)
        state = state + k[:, None] * delta[None, :]
        y = tl.sum(state * q[:, None], 0)
        tl.store(P + vpos, prediction, vid < values)
        tl.store(E + vpos, error, vid < values)
        tl.store(Y + vpos, y, vid < values)
    tl.store(H1 + state_pos, state, state_mask)


def diagnostic(q, k, v, g, beta, initial_state=None):
    batch, length, heads, keys = k.shape
    values = v.shape[-1]
    if beta.ndim != 3 or g.shape != beta.shape:
        raise ValueError("Diagnostic requires the MoM scalar headwise beta and decay")
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    initial_state = initial_state.contiguous() if initial_state is not None else None
    output = torch.empty_like(v, dtype=q.dtype)
    prediction = torch.empty_like(v, dtype=torch.float32)
    residual = torch.empty_like(prediction)
    final = torch.empty((batch, heads, keys, values), dtype=torch.float32, device=q.device)
    bv = min(triton.next_power_of_2(values), 8)
    residual_kernel[(triton.cdiv(values, bv), batch * heads)](
        q, k, v, g, beta, initial_state if initial_state is not None else final,
        output, final, prediction, residual, length, keys ** -0.5, heads, keys, values,
        triton.next_power_of_2(keys), bv, initial_state is not None, num_stages=3, num_warps=1)
    return output, final, prediction, residual
