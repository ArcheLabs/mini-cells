"""Pretrained Qwen and Titans engineering smoke for a saved T4 run."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from minicells_runtime import MultiCellRuntime
from minicells_runtime.qwen35 import Qwen35Adapter

sys.path.insert(0, str(ROOT / 'research/projects/cellular-language-models/experiments'
                      / 'epclm-006-cross-session-cell-portability'))
from cell_artifact import export_cell, import_cell, state_hash, tensor_digest
from memory import NeuralMemory


def setup():
    torch.manual_seed(42)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Select a Tesla T4 accelerator')
    backbone = Qwen35Adapter.load()
    dim = backbone.config.hidden_size
    memory = NeuralMemory(dim).cuda().eval()
    projection = torch.nn.Linear(memory.dim, dim, bias=False).cuda()
    gate = torch.nn.Linear(dim, 1).cuda()
    identity = {**backbone.identity, 'adapter_sha256': tensor_digest({
        f'{prefix}.{key}': value for prefix, module in
        [('memory', memory), ('projection', projection), ('gate', gate)]
        for key, value in module.state_dict().items()
    })}
    return MultiCellRuntime(backbone, memory, projection, gate), identity


def main(output):
    runtime, identity = setup()
    ids = torch.tensor([[2, 3, 4, 5]], device='cuda')
    for name in ('A', 'B', 'C'):
        runtime.create(name)
    a_before, b_before = (state_hash(runtime.get(name)) for name in ('A', 'B'))
    c_before = state_hash(runtime.get('C'))
    baseline = runtime.generate(ids)
    learned = runtime.generate(ids, read_cells=('A', 'B'), update_cell='C',
                               learning_payload=(ids, ids.flip(1)))
    torch.testing.assert_close(baseline['logits'], learned['logits'], atol=0, rtol=0)
    assert state_hash(runtime.get('A')) == a_before
    assert state_hash(runtime.get('B')) == b_before
    assert state_hash(runtime.get('C')) != c_before
    read = runtime.generate(ids, read_cells=('A', 'B', 'C'))
    assert not torch.equal(read['logits'], baseline['logits'])
    runtime.export_bundle(learned['runtime_state'], ('A', 'B', 'C'), output / 'bundle',
                          identity=identity, exporter=export_cell)
    fresh = MultiCellRuntime(runtime.backbone, runtime.memory, runtime.projection, runtime.gate)
    state = fresh.import_bundle(output / 'bundle', identity=identity, importer=import_cell)
    assert state.cache.get_seq_length() == 0
    assert state_hash(fresh.get('C')) == state_hash(runtime.get('C'))
    continued = fresh.generate(learned['pending_token'], runtime_state=state, read_cells=('C',))
    assert continued['runtime_state'].processed == state.processed + 1
    return dict(status='ENGINEERING_SMOKE_PASS', gpu=torch.cuda.get_device_name(),
                torch=str(torch.__version__), identity=identity,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                scope='single-process T4 engineering smoke')


if __name__ == '__main__':
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    result = main(output)
    (output / 'smoke.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
