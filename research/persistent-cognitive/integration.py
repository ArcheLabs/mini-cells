"""Experimental native Qwen attention integration with atomic Cell publication."""
from contextlib import ExitStack
from dataclasses import replace

import torch
from torch import nn

from memory import NeuralMemory


class Retrofit(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = tuple(i for i, kind in enumerate(model.config.layer_types)
                            if kind == 'full_attention')
        if not self.layers or (model.config.num_attention_heads, model.config.num_key_value_heads,
                               model.config.head_dim) != (8, 2, 256):
            raise ValueError('Unsupported Qwen head mapping')
        self.memories = nn.ModuleDict({f'{i}:{h}': NeuralMemory() for i in self.layers for h in range(2)})
        self.mix = nn.ParameterDict({str(i): nn.Parameter(torch.full((8, 256), -2.1972246))
                                     for i in self.layers})

    def forward(self, model, cell, *, chunk=1, meta=False, **inputs):
        """Run native attention against caller-owned Session state."""
        if cell.shards.keys() != self.memories.keys():
            raise ValueError('Cell mapping differs')
        if model.config._attn_implementation != 'eager':
            raise ValueError('Expected pinned eager Qwen backend')
        staged = {}
        with ExitStack() as hooks:
            for index in self.layers:
                attention = model.model.layers[index].self_attn
                captured = {}

                def capture(module, args, kwargs, captured=captured):
                    x = kwargs['hidden_states']
                    if x.ndim != 3 or x.shape[0] != 1:
                        raise ValueError('One independent Session per call')
                    t = x.shape[1]
                    q, gate = module.q_proj(x).reshape(t, 8, 512).chunk(2, -1)
                    captured.update(q=module.q_norm(q).float(),
                                    k=module.k_norm(module.k_proj(x).reshape(t, 2, 256)).float(),
                                    v=module.v_proj(x).reshape(t, 2, 256).float(),
                                    gate=gate.sigmoid())

                def fuse(module, args, captured=captured, index=index):
                    reads = []
                    for h in range(2):
                        name = f'{index}:{h}'
                        r, staged[name] = self.memories[name].scan(
                            captured['k'][:, h], captured['v'][:, h],
                            captured['q'][:, h*4:h*4+4], cell.shards[name],
                            chunk=chunk, meta=meta)
                        reads.append(r)
                    local = args[0].reshape(1, -1, 8, 256)
                    memory = torch.cat(reads, dim=1)[None] * captured['gate'][None]
                    mixed = torch.lerp(local.float(), memory.float(), self.mix[str(index)].sigmoid())
                    return (mixed.flatten(-2).to(args[0].dtype),)

                hooks.callback(attention.register_forward_pre_hook(capture, with_kwargs=True).remove)
                hooks.callback(attention.o_proj.register_forward_pre_hook(fuse).remove)
            outputs = model(**inputs)
        if staged.keys() != self.memories.keys() or not all(
                t.isfinite().all() for s in staged.values() for t in (*s.fast, *s.momentum, s.address)):
            raise RuntimeError('Incomplete or nonfinite Cell transition')
        return outputs, replace(cell, shards=staged, revision=cell.revision + 1)
