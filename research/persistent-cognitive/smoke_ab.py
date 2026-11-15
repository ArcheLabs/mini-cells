"""Run the pretrained Phase A/B engineering check."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from minicells_runtime.qwen35 import Qwen35Adapter
from integration import Retrofit, RuntimeConfig


@torch.no_grad()
def main(output):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model
    model.set_attn_implementation('eager')
    retrofit = Retrofit(model, RuntimeConfig(local_window=4)).cuda()
    ids = torch.tensor([[101, 307, 509, 701, 907, 1103]], device='cuda')
    native = model(input_ids=ids[:, :4], use_cache=True)
    actual, _ = retrofit(model, None, input_ids=ids[:, :4], use_cache=True, memory_enabled=False)
    torch.testing.assert_close(actual.logits, native.logits, atol=1e-5, rtol=1e-5)
    for index, kind in enumerate(model.config.layer_types):
        if kind == 'linear_attention':
            for name in ('conv_states', 'recurrent_states'):
                torch.testing.assert_close(getattr(actual.past_key_values, name)[index],
                                           getattr(native.past_key_values, name)[index])
    for index in (4, 5):
        actual, _ = retrofit(model, None, input_ids=ids[:, index:index+1],
                             past_key_values=actual.past_key_values,
                             position_ids=torch.tensor([[index]], device='cuda'),
                             use_cache=True, memory_enabled=False)
        for layer in retrofit.layers:
            assert actual.past_key_values.get_seq_length(layer) == 4
        assert actual.logits.isfinite().all()
    output.write_text(json.dumps(dict(status='PCA_LOCAL_ATTENTION_SMOKE_PASS',
        identity=adapter.identity, layers=retrofit.layers, q_heads=retrofit.q_heads,
        kv_heads=retrofit.kv_heads, head_dim=retrofit.width, window=4,
        torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
        memory_retrieval='UNVERIFIED', training_updates=0), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
