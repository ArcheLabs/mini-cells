"""T4 causal and state checks for the online retrofit."""
import json
from pathlib import Path
import sys

from smoke import Qwen35Adapter, torch
from titans_attention import HeadMemory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/projects/cellular-language-models/experiments'
                      / 'epclm-004-unified-memory-10m/m3'))
from memory import NeuralMemory


@torch.no_grad()
def main(output):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load()
    model = adapter.model
    model.set_attn_implementation('eager')
    index = model.config.layer_types.index('full_attention')
    retrofit = HeadMemory(model.model.layers[index].self_attn, NeuralMemory).cuda()
    cells = [m.create(mode='online_adapt') for m in retrofit.shards]
    initial = tuple(cells)
    ids = torch.tensor([[101, 307, 509, 701]], device='cuda')
    base, _ = adapter.encode(ids)
    with retrofit.attach(model, index, cells):
        first, state = adapter.encode(ids)
    torch.testing.assert_close(first, base, atol=0, rtol=0)
    for before, after in zip(initial, cells):
        assert after.update_step == 1
        assert any(not torch.equal(before.fast[k], after.fast[k]) for k in before.fast)
        assert any(t.count_nonzero() for t in after.momentum.values())
    before = tuple(cells)
    token = ids[:, :1]
    with retrofit.attach(model, index, cells, write=False):
        read, _ = adapter.encode(token, state)
    assert all(a is b for a, b in zip(before, cells))
    with retrofit.attach(model, index, cells):
        continued, state = adapter.encode(token, state)
    torch.testing.assert_close(continued, read, atol=0, rtol=0)
    assert state.cache.key_cache[index] is None
    assert state.cache.value_cache[index] is None
    assert all(cell.update_step == 2 for cell in cells)
    output.write_text(json.dumps(dict(status='E2_ENGINEERING_PASS', identity=adapter.identity,
        layer=index, kv_heads=len(cells), gpu=torch.cuda.get_device_name(),
        updates=[cell.update_step for cell in cells], read_before_write_max_abs=0), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
