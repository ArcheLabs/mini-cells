"""Segment-local Qwen attention with per-layer Titans memory."""
from contextlib import ExitStack
import math

import torch

from segment6c import SegmentState
from titans_attention import HeadMemory


class HybridSession:
    def __init__(self, adapter, memories, *, mode='online_adapt', length=512, detach_backbone_cache=False):
        self.adapter = adapter
        self.model = adapter.model
        self.memories = memories
        self.mode = mode
        self.detach_backbone_cache = detach_backbone_cache
        self.cache = adapter.cache_type(adapter.config)
        self.cells = {i: [m.create(mode=mode) for m in memory.shards]
                      for i, memory in memories.items()}
        self.pending = {i: [] for i in memories}
        self.clock = SegmentState(self.cache, memories, self._commit, length)
        self.metrics = {}

    def _commit(self):
        new = {}
        for index, memory in self.memories.items():
            keys, values = (torch.cat(parts, dim=2) for parts in zip(*self.pending[index]))
            new[index] = [shard.learn(keys[:, h], values[:, h], self.cells[index][h],
                                      self.mode)[0] for h, shard in enumerate(memory.shards)]
        if not all(t.isfinite().all() for group in new.values() for cell in group
                   for t in (*cell.fast.values(), *cell.momentum.values())):
            raise FloatingPointError('Nonfinite segment memory')
        self.cells = new
        for entries in self.pending.values():
            entries.clear()

    def flush_segment(self):
        self.clock.flush_segment()

    def forward(self, ids, *, read=True):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1]:
            raise ValueError('Expected one nonempty token stream')
        outputs = []
        # Native DeltaNet consumes carried state on single-token calls.
        while ids.shape[1]:
            width = min(ids.shape[1], self.clock.length) if self.clock.position == 0 else 1
            tokens, ids = ids[:, :width], ids[:, width:]
            position = torch.arange(self.clock.position, self.clock.position + width,
                                    device=tokens.device)[None]
            with ExitStack() as stack:
                for index, memory in self.memories.items():
                    attention = self.model.model.layers[index].self_attn
                    captured = {}

                    def capture(module, args, kwargs, captured=captured):
                        hidden = kwargs['hidden_states']
                        shape = (*hidden.shape[:-1], -1, module.head_dim)
                        q, gate = module.q_proj(hidden).view(*shape[:-1], 2 * shape[-1]).chunk(2, -1)
                        k = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
                        q, k = apply_rotary_pos_emb(module.q_norm(q).transpose(1, 2), k,
                                                    *kwargs['position_embeddings'])
                        captured.update(q=q, k=k.float().detach(),
                                        v=module.v_proj(hidden).view(shape).transpose(1, 2).float().detach(),
                                        gate=gate.flatten(-2).sigmoid())

                    def fuse(module, args, captured=captured, memory=memory, index=index):
                        local = args[0]
                        q = captured['q']
                        reads = []
                        for shard, cell, group in zip(memory.shards, self.cells[index],
                                                       q.chunk(len(memory.shards), dim=1)):
                            value = shard.recall(group.transpose(1, 2).flatten(1, 2).float(), cell,
                                'meta_train' if self.mode == 'meta_train' else 'read_only')
                            reads.append(value.reshape(q.shape[0], q.shape[2], -1, q.shape[-1]))
                        recalled = torch.cat(reads, dim=2).flatten(-2).to(local.dtype) * captured['gate']
                        mix = memory.gate.sigmoid().to(local.dtype)
                        output = torch.lerp(local, recalled, mix)
                        self.metrics[index] = torch.stack([mix.float().mean(), *(
                            t.float().square().mean().sqrt() for t in (local, recalled, output))]).detach()
                        return (output,)

                    stack.callback(attention.register_forward_pre_hook(capture, with_kwargs=True).remove)
                    if read and all(c.update_step for c in self.cells[index]):
                        stack.callback(attention.o_proj.register_forward_pre_hook(fuse).remove)
                    self.pending[index].append(captured)
                hidden = self.model.model(input_ids=tokens, past_key_values=self.cache,
                                          position_ids=position, use_cache=True).last_hidden_state
            for entries in self.pending.values():
                capture = entries[-1]
                entries[-1] = (capture['k'], capture['v'])
            if self.detach_backbone_cache:
                for name in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states'):
                    tensors = getattr(self.cache, name)
                    for index, tensor in enumerate(tensors):
                        if tensor is not None:
                            # Conv state is updated in place by the native decoder.
                            tensors[index] = tensor.detach().clone() if name == 'conv_states' else tensor.detach()
            outputs.append(hidden)
            self.clock.advance(width)
        return torch.cat(outputs, dim=1)


def build_memories(model, memory_type):
    memories = {}
    for index, kind in enumerate(model.config.layer_types):
        if kind == 'full_attention':
            memory = HeadMemory(model.model.layers[index].self_attn, memory_type)
            with torch.no_grad():
                memory.gate.fill_(math.log(.1 / .9))
            memories[index] = memory.to(next(model.parameters()).device)
    return memories
