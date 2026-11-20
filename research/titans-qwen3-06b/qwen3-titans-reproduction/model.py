"""Qwen attention adapter for the pinned Titan-Llama implementation."""
from contextvars import ContextVar

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from titans_pytorch.mac_transformer import SegmentedAttention
from titans_pytorch.neural_memory import NeuralMemory, MemoryMLP

from window_attention import WindowAttention

LAYERS = (0, 1, 2, 3)
SEGMENT = 512
TTT_BATCH = 1024
SESSION = ContextVar('reproduction_forward')
DIAGNOSTICS = ContextVar('reproduction_diagnostics', default=None)


class MixLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x.to(self.weight.dtype)).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, native, inv_freq, layer):
        super().__init__()
        self.native, self.layer = native, layer
        self.segmented = SegmentedAttention(
            dim=1024, segment_len=SEGMENT, num_persist_mem_tokens=0,
            num_longterm_mem_tokens=0, dim_head=128, heads=16, num_kv_heads=8,
            sliding=False, accept_value_residual=layer != 0,
            attend_kwargs={'flash': True}, use_flex_attn=False, pre_normed=True,
            rope_freqs=inv_freq,
        ).to(native.q_proj.weight)
        # The trainer must not register the reference module's unused Q/K/V
        # projections as trainable. Use the original Qwen output projection.
        self.segmented.attend = WindowAttention()
        self.segmented.requires_grad_(False)
        del self.segmented.to_q, self.segmented.to_kv
        self.segmented.to_out = native.o_proj
        mix = self.segmented.to_learned_v_mix
        if mix is not None:
            mix[0] = MixLinear(1024, 8, device=native.q_proj.weight.device)
            mix.float().requires_grad_(True)
            nn.init.zeros_(mix[0].weight)
            nn.init.constant_(mix[0].bias, -10.)
        self.memory = NeuralMemory(
            dim=1024, chunk_size=SEGMENT, batch_size=TTT_BATCH, dim_head=128,
            heads=8, retrieve_heads=16, model=MemoryMLP(
                dim=128, depth=3, expansion_factor=1., activation='gelu'),
            qkv_receives_diff_views=True, accept_weight_residual=False,
            max_grad_norm=1., default_step_transform_max_lr=1.,
            detach_inner_grads=False, bypass_projections=True,
            mem_model_norm_add_residual=False, heterogeneous_qkv=True,
            history_context_dim=1024, history_mix_init=.05,
        ).to(device=native.q_proj.weight.device, dtype=torch.float32)
        self.memory.initialize_kv_memory(native.k_proj.weight, native.v_proj.weight, identity=True)

        self.memory.to_adaptive_step.register_forward_hook(self.capture_step)

    def capture_step(self, module, inputs, result):
        rows = DIAGNOSTICS.get()
        if rows is not None:
            step = self.memory.adaptive_step_transform(result.detach()).flatten()
            percentiles = step.quantile(step.new_tensor([.5, .95])).tolist()
            rows.append(dict(layer=self.layer, adaptive_step_mean=float(step.mean()),
                             adaptive_step_p50=percentiles[0], adaptive_step_p95=percentiles[1]))

    def projections(self, x, position_embeddings):
        shape = (*x.shape[:2], -1, 128)
        q = self.native.q_norm(self.native.q_proj(x).view(shape)).transpose(1, 2)
        k = self.native.k_norm(self.native.k_proj(x).view(shape)).transpose(1, 2)
        v = self.native.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if (q.shape[1], k.shape[1], v.shape[1]) != (16, 8, 8):
            raise ValueError('Qwen GQA layout mismatch')
        return q, k, v

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        session = SESSION.get()
        if session['mode'] == 'native':
            return self.native(hidden_states, position_embeddings, attention_mask, **kwargs)
        q, k, v = self.projections(hidden_states, position_embeddings)
        def trace(stage, **fields):
            if DIAGNOSTICS.get() is not None and q.is_cuda:
                row = dict(layer=self.layer, stage=stage, **fields,
                           allocated_bytes=torch.cuda.memory_allocated(q.device),
                           running_peak_bytes=torch.cuda.max_memory_allocated(q.device))
                DIAGNOSTICS.get().append(row)
                print(row, flush=True)

        trace('before_memory', q=list(q.shape), k=list(k.shape), v=list(v.shape),
              windows_per_sequence=(q.shape[-2] + SEGMENT - 1) // SEGMENT)
        if session['mode'] == 'titans':
            with torch.autocast(hidden_states.device.type, enabled=False):
                read, blend, state = self.memory(q.float(), k.float(), v.float(),
                    context=hidden_states.float(), store_mask=session['mask'])
            rows = DIAGNOSTICS.get()
            if rows is not None and state.seq_index >= SEGMENT:
                fast, momentum = state.states
                initial = self.memory.memory_model_parameters
                differences = [(w.detach().reshape(-1, *p.shape) - p.detach()).square()
                               for w, p in zip(fast, initial)]
                mix = self.segmented.to_learned_v_mix
                rows.append(dict(layer=self.layer, sequence_index=int(state.seq_index),
                    fast_weight_delta_rms=float((sum(v.sum() for v in differences) /
                                                 sum(v.numel() for v in differences)).sqrt()),
                    momentum_norm=float(sum(m.detach().square().sum() for m in momentum).sqrt()),
                    memory_read_rms=float(read.detach().square().mean().sqrt()),
                    history_blend_mean=float(blend.detach().mean()),
                    value_residual_mix=float(mix(hidden_states.detach()).detach().float().mean()) if mix else None))
            read, blend = read.to(q.dtype), blend.to(q.dtype)
        else:
            read = blend = None
        trace("before_attention")
        out, info = self.segmented(hidden_states, value_residual=session['value'],
            memory_values=read, memory_mix=blend, return_cache=False,
            precomputed_qkv=(q, k, v), qkv_already_rotated=True)
        trace('after_attention', backend=self.segmented.attend.backend,
              windows_per_call=self.segmented.attend.windows_per_call)
        session['value'] = info.value_residual
        return out, None


def install(model):
    cfg = model.config
    actual = (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
              cfg.num_key_value_heads, cfg.head_dim, cfg.max_position_embeddings)
    if actual != (28, 1024, 16, 8, 128, 40960):
        raise ValueError(f'Unexpected Qwen configuration: {actual}')
    model.requires_grad_(False)
    for i in LAYERS:
        layer = model.model.layers[i]
        layer.self_attn = Attention(layer.self_attn, model.model.rotary_emb.inv_freq, i)
    return model


def hidden_states(model, ids, mask, mode):
    if mode not in ('native', 'segmented', 'titans'):
        raise ValueError(mode)
    mask = mask.bool()
    if ids.shape[1] > 32768:
        raise ValueError('Experiment context limit is 32768 tokens')
    if mask.shape != ids.shape or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError('Expected right padded sequences')
    handle = SESSION.set(dict(mode=mode, mask=mask, value=None))
    try:
        return model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    finally:
        SESSION.reset(handle)


def checkpoint_native_layers(model):
    """Do not put Titans torch.func.grad inside saved-tensor checkpoint hooks."""
    from torch.utils.checkpoint import checkpoint

    def wrap(forward):
        def recompute(*args, **kwargs):
            if not torch.is_grad_enabled():
                return forward(*args, **kwargs)
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return recompute

    for layer in model.model.layers[4:]:
        layer.forward = wrap(layer.forward)
