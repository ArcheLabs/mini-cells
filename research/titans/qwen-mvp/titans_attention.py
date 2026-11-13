"""Single-layer online Titans retrofit for the pinned eager Qwen path."""
from contextlib import contextmanager

import torch
from torch import nn

from local_attention import chunk_local


class HeadMemory(nn.Module):
    def __init__(self, attention, memory_type):
        super().__init__()
        width = attention.head_dim
        count = attention.k_proj.out_features // width
        self.shards = nn.ModuleList(memory_type(width, width, 64) for _ in range(count))
        self.gate = nn.Parameter(torch.zeros(attention.o_proj.in_features))

    @contextmanager
    def attach(self, model, index, cells, *, write=True, mode='online_adapt'):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        if len(cells) != len(self.shards):
            raise ValueError('Expected one state per KV head')
        attention = model.model.layers[index].self_attn
        pending = []
        original = tuple(cells)

        def capture(module, args, kwargs):
            hidden = kwargs['hidden_states']
            shape = (*hidden.shape[:-1], -1, attention.head_dim)
            q, output_gate = attention.q_proj(hidden).view(*shape[:-1], 2 * shape[-1]).chunk(2, -1)
            q = attention.q_norm(q).transpose(1, 2)
            k = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
            v = attention.v_proj(hidden).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
            if pending:
                raise RuntimeError('Attach covers exactly one model call')
            pending.append((q, k, v, output_gate.flatten(-2)))

        def fuse(module, args):
            q, _, _, output_gate = pending[0]
            groups = q.chunk(len(self.shards), dim=1)
            reads = []
            for memory, cell, queries in zip(self.shards, original, groups):
                queries = queries.transpose(1, 2).flatten(1, 2)
                reads.append(memory.recall(queries.float(), cell, 'meta_train' if mode == 'meta_train' else 'read_only').reshape(
                    q.shape[0], q.shape[2], -1, q.shape[-1]))
            residual = torch.cat(reads, dim=2).flatten(-2).to(args[0].dtype)
            return (args[0] + (residual * self.gate.sigmoid() * output_gate.sigmoid()).to(args[0].dtype),)

        with chunk_local(model, index):
            capture_handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
            fuse_handle = (attention.o_proj.register_forward_pre_hook(fuse)
                           if all(cell.update_step for cell in original) else None)
            try:
                yield
                if write:
                    _, keys, values, _ = pending[0]
                    updates = [memory.learn(keys[:, i].float(), values[:, i].float(),
                                            cell, mode)[0]
                               for i, (memory, cell) in enumerate(zip(self.shards, original))]
                    if not all(t.isfinite().all() for cell in updates
                               for t in (*cell.fast.values(), *cell.momentum.values())):
                        raise FloatingPointError('Nonfinite Titans state')
                    cells[:] = updates
            finally:
                capture_handle.remove()
                if fuse_handle is not None:
                    fuse_handle.remove()
