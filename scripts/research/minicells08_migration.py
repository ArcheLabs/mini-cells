"""Independent source and target processes for the pinned T4 fixture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from minicells08_smoke import export_cell, import_cell, setup, state_hash, tensor_digest


def recurrent_digest(state):
    return tensor_digest({
        f'{name}.{index}': tensor
        for name in ('conv_states', 'recurrent_states')
        for index, tensor in enumerate(getattr(state.cache, name))
        if tensor is not None
    })


@torch.no_grad()
def read(runtime, ids):
    hidden, _ = runtime.backbone.encode(ids)
    query = runtime.memory.project(hidden.float())[2]
    return runtime.memory.recall(query, runtime.get('C')).cpu()


def worker(output, role):
    runtime, identity = setup()
    ids = torch.tensor([[2, 3, 4, 5]], device='cuda')
    report = {'pid': os.getpid(), 'identity': identity}
    artifact = output / 'cell'
    snapshot = output / 'runtime'
    if role == 'source':
        runtime.create('C')
        result = runtime.generate(ids, update_cell='C', learning_payload=(ids, ids.flip(1)))
        state = result['runtime_state']
        runtime.backbone.export_runtime_state(state, snapshot)
        report['artifact_sha256'] = runtime.export_cell('C', artifact, identity=identity, exporter=export_cell)
        report['pending_token'] = result['pending_token'].tolist()
        report['recurrent_sha256'] = recurrent_digest(state)
        report['cell_state'] = state_hash(runtime.get('C'))
        report['readout'] = read(runtime, ids).tolist()
    else:
        source = json.loads((output / 'source.json').read_text())
        if source['identity'] != identity:
            raise ValueError('Shared weights differ')
        runtime.import_cell('C', artifact, identity=identity, importer=import_cell)
        before = state_hash(runtime.get('C'))
        if before != source['cell_state']:
            raise ValueError('Imported Cell state differs')
        actual = read(runtime, ids)
        expected = torch.tensor(source['readout'])
        report['readout_max_abs'] = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if state_hash(runtime.get('C')) != before:
            raise ValueError('Read mutated Cell')
        state = runtime.backbone.import_runtime_state(snapshot)
        if state.cache.get_seq_length() != 0 or recurrent_digest(state) != source['recurrent_sha256']:
            raise ValueError('Invalid recurrent-only restore')
        token = torch.tensor(source['pending_token'], device='cuda')
        continued = runtime.generate(token, runtime_state=state, read_cells=('C',))
        if continued['runtime_state'].processed != state.processed + 1:
            raise ValueError('Restored session did not advance')
        runtime.create('control')
        control = state_hash(runtime.get('control'))
        runtime.generate(ids, update_cell='C', learning_payload=(ids.flip(1), ids))
        after = state_hash(runtime.get('C'))
        if (after['step'] != before['step'] + 1 or after['fast'] == before['fast']
                or after['momentum'] == before['momentum'] or state_hash(runtime.get('control')) != control):
            raise ValueError('Restored Cell did not learn independently')
        report.update(cell_before=before, cell_after=after, recurrent_sha256=recurrent_digest(state),
                      initial_attention_length=0, continued_token=continued['tokens'].tolist())
    (output / f'{role}.json').write_text(json.dumps(report, indent=2))


def main(output):
    results = {}
    for role in ('source', 'target'):
        process = subprocess.run([sys.executable, __file__, str(output), '--role', role],
                                 check=True, timeout=800)
        results[role] = json.loads((output / f'{role}.json').read_text())
        results[role]['exit_code'] = process.returncode
    if results['source']['pid'] == results['target']['pid']:
        raise RuntimeError('Expected distinct source and target PIDs')
    report = dict(status='CROSS_PROCESS_RESTORE_PASS', **results)
    (output / 'migration.json').write_text(json.dumps(report, indent=2))
    print(report['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--role', choices=('source', 'target'))
    args = parser.parse_args()
    if args.role:
        worker(args.output, args.role)
    else:
        main(args.output)
