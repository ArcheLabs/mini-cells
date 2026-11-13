"""Verify same-model migration and explicit independent Cell routing."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import torch

from stage2 import HeadMemory, NeuralMemory, Qwen35Adapter, ROOT

sys.path.insert(0, str(ROOT / 'research/projects/cellular-language-models/experiments'
                      / 'epclm-006-cross-session-cell-portability'))
from cell_artifact import export_cell, import_cell, state_hash, tensor_digest


@torch.no_grad()
def worker(output, role):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model
    model.set_attn_implementation('eager')
    layers = [i for i, kind in enumerate(model.config.layer_types) if kind == 'full_attention']
    memories = {i: HeadMemory(model.model.layers[i].self_attn, NeuralMemory).cuda() for i in layers}
    ids = torch.tensor([[101, 307, 509, 701]], device='cuda')
    cells = {name: {i: [m.create(mode='online_adapt') for m in memories[i].shards]
                    for i in layers} for name in ('A', 'B')}
    session = str(uuid.uuid4())

    def hashes(name):
        return [state_hash(c) for i in layers for c in cells[name][i]]

    def execute(tokens, readers=(), writer=None):
        with ExitStack() as stack:
            for i in layers:
                stack.enter_context(memories[i].attach(
                    model, i, cells[writer or 'A'][i], write=writer is not None,
                    read_cells=[cells[name][i] for name in readers]))
            hidden, state = adapter.encode(tokens, absolute_positions=True)
        assert all(t is None for t in (*state.cache.key_cache, *state.cache.value_cache))
        if not hidden.isfinite().all():
            raise FloatingPointError('Nonfinite hidden states')
        return hidden

    if role == 'source':
        execute(ids, writer='A')
        execute(ids.flip(1), writer='B')
    artifact_hashes = {}
    for name in ('A', 'B'):
        for i in layers:
            for j, memory in enumerate(memories[i].shards):
                path = output.parent / 'cells' / name / f'{i}-{j}'
                identity = dict(**adapter.identity, layer=i, head=j, position='absolute-post-rope',
                                next_position=4, memory_sha256=tensor_digest(memories[i].state_dict()))
                if role == 'source':
                    artifact_hashes[f'{name}/{i}-{j}'] = export_cell(cells[name][i][j], path, identity, session)
                else:
                    holder = SimpleNamespace(memory=memory, parameters=memory.parameters)
                    cells[name][i][j] = import_cell(path, holder, identity)
    before = {name: hashes(name) for name in cells}
    results = {name: execute(ids[:, :1], readers=readers)
               for name, readers in [('base', ()), ('A', ('A',)), ('B', ('B',)), ('AB', ('A', 'B'))]}
    assert before == {name: hashes(name) for name in cells}
    assert not torch.equal(results['A'], results['B'])
    assert all(not torch.equal(results['AB'], results[name]) for name in ('base', 'A', 'B'))
    record = dict(pid=os.getpid(), identity=adapter.identity, layers=layers, states=before,
                  hidden={name: tensor.cpu().tolist() for name, tensor in results.items()})
    if role == 'source':
        record['artifact_hashes'] = artifact_hashes
    else:
        source = json.loads(output.with_name('source.json').read_text())
        assert source['identity'] == adapter.identity and source['states'] == before
        for name, tensor in results.items():
            torch.testing.assert_close(tensor.cpu(), torch.tensor(source['hidden'][name]), atol=1e-5, rtol=1e-4)
        cells['fork'] = {i: [m.fork(c) for m, c in zip(memories[i].shards, cells['A'][i])]
                         for i in layers}
        assert hashes('fork') == before['A']
        execute(ids.flip(1), readers=('A', 'B'), writer='fork')
        assert hashes('fork') != before['A']
        assert all(hashes(name) == before[name] for name in ('A', 'B'))
        execute(ids, readers=('A',), writer='B')
        assert hashes('A') == before['A'] and hashes('B') != before['B']
        record['fork_isolation'] = record['independent_learning'] = 'PASS'
    output.with_name(role + '.json').write_text(json.dumps(record))


if __name__ == '__main__':
    output = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 3:
        worker(output, sys.argv[2])
    else:
        reports = {}
        for role in ('source', 'target'):
            proc = subprocess.run([sys.executable, __file__, str(output), role], check=True, timeout=800)
            reports[role] = json.loads(output.with_name(role + '.json').read_text())
            reports[role]['exit_code'] = proc.returncode
        assert reports['source']['pid'] != reports['target']['pid']
        output.write_text(json.dumps(dict(status='STAGE5_ENGINEERING_PASS', **reports), indent=2))
