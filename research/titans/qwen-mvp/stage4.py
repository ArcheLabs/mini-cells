"""Verify all-layer KV removal on the pinned Qwen reference."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import torch

from stage2 import HeadMemory, NeuralMemory, Qwen35Adapter


@torch.no_grad()
def main(output):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model
    model.set_attn_implementation('eager')
    indices = [i for i, kind in enumerate(model.config.layer_types) if kind == 'full_attention']
    retrofits = {i: HeadMemory(model.model.layers[i].self_attn, NeuralMemory).cuda() for i in indices}
    cells = {i: [m.create(mode='online_adapt') for m in retrofit.shards]
             for i, retrofit in retrofits.items()}
    ids = torch.tensor([[101, 307, 509, 701]], device='cuda')
    baseline, native = adapter.encode(ids)
    state = None
    for step, tokens in enumerate((ids, ids[:, :1], ids[:, 1:2]), 1):
        with ExitStack() as stack:
            for i, retrofit in retrofits.items():
                stack.enter_context(retrofit.attach(model, i, cells[i]))
            hidden, state = adapter.encode(tokens, state, absolute_positions=True)
        if step == 1:
            torch.testing.assert_close(hidden, baseline, atol=0, rtol=0)
        assert hidden.isfinite().all()
        assert all(t is None for t in (*state.cache.key_cache, *state.cache.value_cache))
        assert all(cell.update_step == step for group in cells.values() for cell in group)
    native_bytes = sum(t.numel() * t.element_size()
                       for t in (*native.cache.key_cache, *native.cache.value_cache) if t is not None)
    output.write_text(json.dumps(dict(status='STAGE4_ENGINEERING_PASS', identity=adapter.identity,
        layers=indices, native_prefill_kv_bytes=native_bytes, retrofit_kv_bytes=0,
        processed=state.processed, updates_per_shard=3), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
