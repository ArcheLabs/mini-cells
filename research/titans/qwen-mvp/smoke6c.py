"""Check real Qwen segment boundaries and three-segment outer gradients."""
import json
from pathlib import Path
import sys

import torch
from torch.nn.functional import cross_entropy

from hybrid6c import HybridSession, build_memories
from stage2 import NeuralMemory, Qwen35Adapter


def main(output):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    adapter.model.requires_grad_(False).set_attn_implementation('eager')
    memories = build_memories(adapter.model, NeuralMemory)
    session = HybridSession(adapter, memories, mode='meta_train', detach_backbone_cache=True)
    ids = torch.randint(100, 8000, (1, 1537), device='cuda')
    with torch.no_grad():
        native, _ = adapter.encode(ids[:, :511])
    actual = session.forward(ids[:, :511])
    torch.testing.assert_close(actual, native, atol=1e-5, rtol=1e-5)
    if session.cache.get_seq_length() != 511:
        raise RuntimeError('Local KV was discarded inside the segment')
    session.forward(ids[:, 511:512])
    if session.cache.get_seq_length() != 0:
        raise RuntimeError('Local KV survived commit')
    for index, cell in session.cells.items():
        if any(shard.update_step != 1 for shard in cell):
            raise RuntimeError(f'Wrong commit count at layer {index}')
    loss_sum = 0.
    for start in range(512, 1536, 16):
        hidden = session.forward(ids[:, start:start + 16])
        logits = adapter.logits(hidden).flatten(0, 1)
        term = cross_entropy(logits, ids[:, start + 1:start + 17].flatten(),
                             reduction='sum')
        # Keep the shared Titans update graph live until the last tile.
        (term / 1024).backward(retain_graph=start + 16 < 1536)
        loss_sum += float(term.detach())
        del hidden, logits, term
        if (start + 16) % 128 == 0:
            print(f'Completed backward through token {start + 16}', flush=True)
    for index, memory in memories.items():
        for param in [memory.gate, *(m.controls.weight for m in memory.shards)]:
            if param.grad is None or not param.grad.isfinite().all() or not param.grad.any():
                raise RuntimeError(f'Invalid outer gradient at layer {index}')
    output.write_text(json.dumps(dict(status='STAGE6C_TRUNCATED_DIAGNOSTIC_PASS',
        identity=adapter.identity, layers=list(memories), segment_length=512,
        supervised_targets=1024, backbone_cache_gradients="detached_between_calls", loss_tile_tokens=16, optimizer_updates=0, nll=loss_sum / 1024,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
