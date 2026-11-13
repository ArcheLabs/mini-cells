"""Compare native and one-layer chunk-local Qwen with real pinned weights."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from minicells_runtime.qwen35 import Qwen35Adapter
from local_attention import chunk_local


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
                                      processed=state.processed), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
