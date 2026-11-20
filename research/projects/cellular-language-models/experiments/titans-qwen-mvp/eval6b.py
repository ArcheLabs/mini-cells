"""Evaluate the four single-layer fixture arms on paired unseen text."""
from contextlib import ExitStack
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
    if proof.get('status') != 'STAGE6B_TRAIN_COMPLETE' or proof.get('checkpoint_sha256') != sha:
        raise ValueError('Stage 3 checkpoint does not match its successful smoke report')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model
    model.set_attn_implementation('eager')
    layers = [i for i, kind in enumerate(model.config.layer_types) if kind == 'full_attention']
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if saved['identity'] != adapter.identity or saved['layers'] != layers or saved['updates'] != proof['updates']:
        raise ValueError('Incompatible six-layer checkpoint')
    torch.manual_seed(proof['seed'])
    retrofit = torch.nn.ModuleDict({
        str(i): HeadMemory(model.model.layers[i].self_attn, NeuralMemory) for i in layers
    }).cuda()
    initial = {k: v.clone() for k, v in retrofit.state_dict().items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    rows = []
    for passage in PASSAGES:
        ids = torch.tensor([tokenizer.encode(passage, add_special_tokens=False)], device='cuda')
        for arm in ('E0', 'E1', 'E2', 'E3', 'E3_GATE_ZERO'):
            retrofit.load_state_dict(saved['adapter'] if arm.startswith('E3') else initial)
            cells = {i: [m.create(mode='online_adapt') for m in retrofit[str(i)].shards] for i in layers}
            state = None
            losses = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for end in range(4, ids.shape[1]):
                with ExitStack() as stack:
                    for i in layers:
                        if arm == 'E1':
                            stack.enter_context(chunk_local(model, i))
                        elif arm in ('E2', 'E3', 'E3_GATE_ZERO'):
                            stack.enter_context(retrofit[str(i)].attach(
                                model, i, cells[i], read_cells=() if arm == 'E3_GATE_ZERO' else None))
                    hidden, state = adapter.encode(ids[:, :end] if state is None else
                                                   ids[:, end - 1:end], state,
                                                   absolute_positions=True)
                losses.append(float(cross_entropy(adapter.logits(hidden[:, -1]).float(), ids[:, end])))
            torch.cuda.synchronize()
            kv_bytes = sum(t.numel() * t.element_size() for t in
                           (*state.cache.key_cache, *state.cache.value_cache) if t is not None)
            if arm != 'E0' and kv_bytes:
                raise RuntimeError(f'{arm} retained attention KV')
            cell_bytes = sum(t.numel() * t.element_size() for group in cells.values()
                             for cell in group for t in (*cell.fast.values(), *cell.momentum.values()))
            rows.append(dict(arm=arm, token_ids=ids.tolist(), losses=losses,
                             nll=sum(losses)/len(losses), seconds=time.perf_counter()-start,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(), kv_bytes=kv_bytes, cell_bytes=cell_bytes if arm in ("E2", "E3", "E3_GATE_ZERO") else 0))
    output.write_text(json.dumps(dict(status='STAGE6B_FIXTURE_COMPLETE', mvp='INCONCLUSIVE',
        identity=adapter.identity, checkpoint_sha256=sha, layers=layers,
        gpu=torch.cuda.get_device_name(), rows=rows), indent=2, allow_nan=False))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
