"""Experimental native Qwen attention integration with atomic Cell publication."""
from contextlib import ExitStack
from dataclasses import dataclass, replace

import torch
from torch import nn

from memory import NeuralMemory


@dataclass(frozen=True)
class RuntimeConfig:
    local_window: int = 512


class Retrofit(nn.Module):
    def __init__(self, model, config=RuntimeConfig()):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        if type(model) is not Qwen3_5ForCausalLM:
            raise TypeError('Expected Qwen3_5ForCausalLM')
        self.local_window = config.local_window
        c = model.config
        self.layers = tuple(i for i, kind in enumerate(c.layer_types) if kind == 'full_attention')
        self.q_heads, self.kv_heads, self.width = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        if (not self.layers or self.local_window < 1 or self.kv_heads < 1
                or self.q_heads < 1 or self.q_heads % self.kv_heads):
            raise ValueError('Invalid attention geometry')
        self.group = self.q_heads // self.kv_heads
        self.memories = nn.ModuleDict({f'{i}:{h}': NeuralMemory(self.width)
                                      for i in self.layers for h in range(self.kv_heads)})
        self.mix = nn.ModuleDict({str(i): nn.Linear(3, self.width) for i in self.layers})
        for gate in self.mix.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.1972246)

    def forward(self, model, cell, *, chunk=1, meta=False, memory_enabled=True,
                memory_strength=1., **inputs):
        """Run a text call with explicit absolute positions on cached continuation."""
        if memory_enabled and (cell is None or cell.shards.keys() != self.memories.keys()):
            raise ValueError('Cell mapping differs')
        if not 0 <= memory_strength <= 1:
            raise ValueError('Memory strength must be in [0, 1]')
        if model.config._attn_implementation != 'eager':
            raise ValueError('Expected pinned eager Qwen backend')
        cache = inputs.get('past_key_values')
        ids = inputs.get('input_ids')
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1] or inputs.get('attention_mask') is not None:
            raise ValueError('Expected unpadded single-stream input_ids')
        if cache is not None and cache.has_previous_state:
            if inputs.get('position_ids') is None or ids.shape[1] != 1:
                raise ValueError('Cached continuation requires one token and absolute position_ids')
        staged = {}
        with ExitStack() as hooks:
            for index in self.layers:
                attention = model.model.layers[index].self_attn
                captured = {}

                def mask(module, args, kwargs):
                    x = kwargs['hidden_states']
                    local_cache = kwargs.get('past_key_values')
                    past = 0 if local_cache is None else local_cache.get_seq_length(module.layer_idx)
                    length = x.shape[1]
                    q = torch.arange(past, past + length, device=x.device)[:, None]
                    k = torch.arange(past + length, device=x.device)[None]
                    allowed = (k <= q) & (k > q - self.local_window)
                    kwargs['attention_mask'] = x.new_zeros((1, 1, length, past + length)).masked_fill(
                        ~allowed, torch.finfo(x.dtype).min)
                    return args, kwargs

                hooks.callback(attention.register_forward_pre_hook(mask, with_kwargs=True).remove)
                if not memory_enabled or memory_strength == 0:
                    continue
                for name, module in [('q', attention.q_norm), ('k', attention.k_norm),
                                     ('v', attention.v_proj), ('gate', attention.q_proj)]:
                    def capture(module, args, value, name=name, captured=captured):
                        if name == 'gate':
                            value = value.reshape(-1, self.q_heads, 2*self.width)[..., self.width:].sigmoid()
                        elif name == 'v':
                            value = value.reshape(-1, self.kv_heads, self.width)
                        else:
                            value = value.squeeze(0)
                        captured[name] = value.float()

                    hooks.callback(module.register_forward_hook(capture).remove)

                def fuse(module, args, captured=captured, index=index):
                    reads = []
                    for h in range(self.kv_heads):
                        name = f'{index}:{h}'
                        r, staged[name] = self.memories[name].scan(
                            captured['k'][:, h], captured['v'][:, h],
                            captured['q'][:, h*self.group:(h+1)*self.group], cell.shards[name],
                            chunk=chunk, meta=meta)
                        reads.append(r)
                    local = args[0].reshape(1, -1, self.q_heads, self.width)
                    memory = torch.cat(reads, dim=1)[None] * captured['gate'][None]
                    features = torch.stack([t.float().square().mean(-1).add(1e-8).sqrt()
                                            for t in (captured['q'][None], local, memory)], dim=-1)
                    weight = self.mix[str(index)](features).sigmoid() * memory_strength
                    mixed = torch.lerp(local.float(), memory.float(), weight)
                    return (mixed.flatten(-2).to(args[0].dtype),)

                hooks.callback(attention.o_proj.register_forward_pre_hook(fuse).remove)
            outputs = model(**inputs)
        cache = getattr(outputs, 'past_key_values', None)
        if cache is not None:
            for index in self.layers:
                for name in ('key_cache', 'value_cache'):
                    tensors = getattr(cache, name)
                    if tensors[index].shape[-2] > self.local_window:
                        tensors[index] = tensors[index][..., -self.local_window:, :].clone()
        if not memory_enabled or memory_strength == 0:
            return outputs, cell
        if staged.keys() != self.memories.keys() or not all(
                t.isfinite().all() for s in staged.values() for t in (*s.fast, *s.momentum, s.address)):
            raise RuntimeError('Incomplete or nonfinite Cell transition')
        return outputs, replace(cell, shards=staged, revision=cell.revision + 1)
