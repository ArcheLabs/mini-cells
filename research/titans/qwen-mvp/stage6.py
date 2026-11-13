"""Evaluate the four single-layer fixture arms on paired unseen text."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn.functional import cross_entropy

from stage2 import HeadMemory, NeuralMemory, Qwen35Adapter
from local_attention import chunk_local
from minicells_runtime.qwen35 import MODEL_ID, REVISION

PASSAGES = (
    'The archive stores maps in a locked cabinet. The cabinet is opened every Friday.',
    'During winter the lake freezes slowly. A thin layer of ice first forms near the shore.',
)


@torch.no_grad()
def main(output, checkpoint, receipt):
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    proof = json.loads(receipt.read_text())
    with checkpoint.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if proof.get('status') != 'E3_GRADIENT_SMOKE_PASS' or proof.get('checkpoint_sha256') != sha:
        raise ValueError('Stage 3 checkpoint does not match its successful smoke report')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model
    model.set_attn_implementation('eager')
    layer = model.config.layer_types.index('full_attention')
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if saved['identity'] != adapter.identity or saved['layer'] != layer or saved['updates'] != 2:
        raise ValueError('Incompatible Stage 3 checkpoint')
    retrofit = HeadMemory(model.model.layers[layer].self_attn, NeuralMemory).cuda()
    initial = {k: v.clone() for k, v in retrofit.state_dict().items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    rows = []
    for passage in PASSAGES:
        ids = torch.tensor([tokenizer.encode(passage, add_special_tokens=False)], device='cuda')
        for arm in ('E0', 'E1', 'E2', 'E3'):
            retrofit.load_state_dict(saved['adapter'] if arm == 'E3' else initial)
            cells = [m.create(mode='online_adapt') for m in retrofit.shards]
            state = None
            losses = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for end in range(4, ids.shape[1]):
                context = (nullcontext() if arm == 'E0' else chunk_local(model, layer)
                           if arm == 'E1' else retrofit.attach(model, layer, cells))
                with context:
                    hidden, state = adapter.encode(ids[:, :end] if state is None else
                                                   ids[:, end - 1:end], state)
                losses.append(float(cross_entropy(adapter.logits(hidden[:, -1]).float(), ids[:, end])))
            torch.cuda.synchronize()
            kv_bytes = sum(t.numel() * t.element_size() for t in
                           (*state.cache.key_cache, *state.cache.value_cache) if t is not None)
            rows.append(dict(arm=arm, token_ids=ids.tolist(), losses=losses,
                             nll=sum(losses)/len(losses), seconds=time.perf_counter()-start,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(), kv_bytes=kv_bytes))
    output.write_text(json.dumps(dict(status='STAGE6_FIXTURE_COMPLETE', mvp='INCONCLUSIVE',
        identity=adapter.identity, checkpoint_sha256=sha, layer=layer,
        gpu=torch.cuda.get_device_name(), rows=rows), indent=2, allow_nan=False))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
