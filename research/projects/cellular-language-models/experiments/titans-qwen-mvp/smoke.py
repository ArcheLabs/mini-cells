"""Compare native and one-layer chunk-local Qwen with real pinned weights."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from minicells_runtime.qwen35 import Qwen35Adapter
from local_attention import chunk_local
from contextlib import nullcontext


def language_fixture(adapter, layer):
    """Teacher-forced comparison on identical fixed text and chunk schedule."""
    from transformers import AutoTokenizer
    from minicells_runtime.qwen35 import MODEL_ID, REVISION

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    passages = (
        'The observatory opens at sunset. Visitors gather outside the observatory '
        'before sunset to watch the stars.',
        'A river flows through the valley. After heavy rain, the river rises '
        'and spreads across the valley floor.',
    )
    rows = []
    for passage in passages:
        ids = torch.tensor([tokenizer.encode(passage, add_special_tokens=False)], device='cuda')
        row = {'token_ids': ids.tolist(), 'prefix_tokens': 4, 'continuation_call_tokens': 1}
        for arm, context in [('E0', nullcontext()), ('E1', chunk_local(adapter.model, layer))]:
            with context:
                hidden, state = adapter.encode(ids[:, :4])
                losses = []
                hits = []
                for position in range(4, ids.shape[1]):
                    logits = adapter.logits(hidden[:, -1]).float()
                    target = ids[:, position]
                    losses.append(float(torch.nn.functional.cross_entropy(logits, target)))
                    hits.append(int(logits.argmax(-1).eq(target).item()))
                    if position + 1 < ids.shape[1]:
                        hidden, state = adapter.encode(ids[:, position:position + 1], state)
                row[arm] = {'nll': sum(losses) / len(losses), 'target_count': len(losses),
                            'top1_correct': sum(hits), 'token_losses': losses}
        rows.append(row)
    return rows


@torch.no_grad()
def main(output):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    adapter = Qwen35Adapter.load()
    model = adapter.model
    model.set_attn_implementation('eager')
    layer = model.config.layer_types.index('full_attention')
    ids = torch.tensor([[101, 307, 509, 701]], device='cuda')
    native, native_state = adapter.encode(ids)
    _, native_next = adapter.encode(ids[:, :1], native_state)
    with chunk_local(model, layer):
        local, local_state = adapter.encode(ids)
        torch.testing.assert_close(local, native, rtol=0, atol=0)
        continued, state = adapter.encode(ids[:, :1], local_state)
    if state.cache.key_cache[layer] is not None or state.cache.value_cache[layer] is not None:
        raise RuntimeError('Selected layer retained KV')
    for index, kind in enumerate(model.config.layer_types):
        if kind == 'full_attention' and index != layer:
            if state.cache.key_cache[index].shape[-2] != 5:
                raise RuntimeError('Unselected cache did not advance')
    if not continued.isfinite().all() or native_next.cache.key_cache[layer].shape[-2] != 5:
        raise RuntimeError('Invalid continuation')
    output.write_text(json.dumps(dict(status='E1_ENGINEERING_PASS', identity=adapter.identity,
                                      layer=layer, layer_types=model.config.layer_types,
                                      first_chunk_max_abs=float((native-local).abs().max()),
                                      processed=state.processed,
                                      language_fixture=language_fixture(adapter, layer)), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
