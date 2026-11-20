"""Bounded pretrained Memory-on and independent snapshot smoke."""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import save
from transformers import AutoTokenizer

from smoke_ab import Qwen35Adapter, Retrofit, RuntimeConfig
from minicells_runtime.qwen35 import MODEL_ID, REVISION
from cell import Cell, CellSlot, restore, snapshot, state_hash
from session import Session, _tensors, advance, restore_session, snapshot_session


@torch.no_grad()
def setup():
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model.requires_grad_(False)
    model.set_attn_implementation('eager')
    retrofit = Retrofit(model, RuntimeConfig(local_window=4)).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    weights = {k: v.detach().cpu().contiguous() for k, v in retrofit.state_dict().items()}
    sha = lambda data: hashlib.sha256(data).hexdigest()
    identity = dict(backbone=sha(json.dumps(adapter.identity, sort_keys=True).encode()),
                    tokenizer=sha(tokenizer.backend_tokenizer.to_str().encode()),
                    architecture='persistent-cognitive-v2/window=4',
                    meta=sha(save(weights)), layer_map=sha(repr(tuple(retrofit.memories)).encode()))
    return adapter, retrofit, identity


@torch.no_grad()
def main(output):
    started = time.monotonic()
    adapter, retrofit, identity = setup()
    model = adapter.model

    def fresh():
        return adapter.cache_type(adapter.config)

    original = Cell.create(retrofit.memories, identity)
    slot = CellSlot(original)
    session = Session.create(original, fresh())
    ids = torch.tensor([[101, 307, 509, 701, 907, 1103]], device='cuda')
    off, _ = advance(session, slot, model, retrofit, ids[:, :4], identity, memory_enabled=False)
    on, session = advance(session, slot, model, retrofit, ids[:, :4], identity)
    difference = float((off.logits - on.logits).abs().max())
    if not difference > 0:
        raise RuntimeError('Memory-on fusion did not change logits')
    for token in ids[:, 4:5].split(1, dim=1):
        _, session = advance(session, slot, model, retrofit, token, identity)
    current = slot.current
    if session.position != 5 or any(s.step != 5 for s in current.shards.values()):
        raise RuntimeError('Wrong absolute position or shard counters')
    if any(session.runtime_cache.get_seq_length(i) != 4 for i in retrofit.layers):
        raise RuntimeError('Expected bounded KV after eviction')
    if any(s.step != 0 for s in original.shards.values()):
        raise RuntimeError('Source Cell was mutated')
    print('Memory-on and eviction checked; verifying snapshots', flush=True)
    snapshot(current, output.with_name('cell'))
    snapshot_session(session, output.with_name('session'))
    restored = restore(output.with_name('cell'), retrofit.memories, identity)
    if state_hash(restored) != state_hash(current):
        raise RuntimeError('Cell hash differs')
    fresh_session = Session.create(restored, fresh())
    if _tensors(fresh_session.runtime_cache):
        raise RuntimeError('Cell restore brought Session state')
    resumed = restore_session(output.with_name('session'), restored, fresh(),
                              _tensors(session.runtime_cache))
    a, continued = advance(session, slot, model, retrofit, ids[:, -1:], identity)
    restore_slot = CellSlot(restored)
    b, resumed = advance(resumed, restore_slot, model, retrofit, ids[:, -1:], identity)
    torch.testing.assert_close(a.logits, b.logits, atol=1e-5, rtol=1e-5)
    if (continued.position != resumed.position or
            state_hash(slot.current) != state_hash(restore_slot.current)):
        raise RuntimeError('Continued Cell state differs')
    output.write_text(json.dumps(dict(status='PCA_MEMORY_RUNTIME_SMOKE_PASS',
        identity=adapter.identity, fingerprints=identity, layers=retrofit.layers,
        window=4, processed_tokens=resumed.position, memory_on_logit_max_abs=difference,
        restored_logit_max_abs=float((a.logits-b.logits).abs().max()),
        cell_only_empty_session=True, full_session_restore=True, training_updates=0,
        cross_window_recall='UNVERIFIED', phase_ab='PARTIAL',
        seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated()),
        indent=2, allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
