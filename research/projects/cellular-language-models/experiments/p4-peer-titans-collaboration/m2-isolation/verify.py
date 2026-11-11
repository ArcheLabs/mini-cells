"""Run the peer isolation matrix with separate writer and reader interpreters."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_m1 import M3, file_hash, initialize, read, state_hash, tensor_hash, write
from e0 import load_checkpoint, save_checkpoint


def static_hash(memory):
    return {k:tensor_hash(v) for k,v in memory.state_dict().items()}


def interface(memory):
    return dict(interface=1, architecture='mag_lite_single_memory',
                source=file_hash(M3/'memory.py'), static=static_hash(memory),
                controls=[memory.max_lr, memory.max_forget])


def storage(tensors):
    return {v.untyped_storage().data_ptr() for v in tensors}


def independent(a, b):
    tensors = lambda c: [*c.fast.values(), *c.momentum.values()]
    assert not storage(tensors(a)) & storage(tensors(b))
    assert a.metadata['cell_id'] != b.metadata['cell_id']
    assert a.metadata is not b.metadata
    assert a.metadata['provenance'] is not b.metadata['provenance']


def equivalent(a, b):
    for name in ('fast','momentum'):
        x,y = getattr(a,name),getattr(b,name)
        assert x.keys() == y.keys() and all(torch.equal(x[k],y[k]) for k in x)
    assert a.update_step == b.update_step


def update(memory, active, peer, keys, values, queries, steps):
    peer_snapshot = copy.deepcopy(state_hash(peer))
    peer_output = read(memory,peer,queries)
    before = state_hash(active)
    for _ in range(steps):
        active,_ = memory.learn(keys[None],values[None],active,'online_adapt')
    after = state_hash(active)
    assert before['fast'] != after['fast'] and before['momentum'] != after['momentum']
    assert active.update_step == before['step']+steps
    assert active.metadata['revision'] == before['metadata']['revision']+steps
    assert state_hash(peer) == peer_snapshot and torch.equal(read(memory,peer,queries),peer_output)
    assert all(v.grad_fn is None for v in [*active.fast.values(),*active.momentum.values()])
    independent(active,peer)
    return active


def restore(memory, folder, expected):
    path = folder/'manifest.json'
    if file_hash(path) != expected['manifest_sha256']:
        raise ValueError('Manifest hash mismatch')
    manifest = json.loads(path.read_text())
    if manifest['metadata']['architecture_id'] != interface(memory)['architecture']:
        raise ValueError('Architecture mismatch')
    if manifest['compatibility'] != interface(memory):
        raise ValueError('Static interface mismatch')
    generation = manifest['generation']
    if len(generation) != 32 or any(c not in '0123456789abcdef' for c in generation):
        raise ValueError('Invalid generation')
    for name in ('fast_weights.pt','momentum.pt'):
        file = folder/generation/name
        if file_hash(file) != manifest['files'][name]:
            raise ValueError('Component hash mismatch')
        tensors = load_checkpoint(file)
        if tensors.keys() != memory.initial.keys():
            raise ValueError('Tensor names mismatch')
        for k,v in tensors.items():
            if list(v.shape) != [1,*memory.initial[k].shape] or v.dtype != memory.initial[k].dtype:
                raise ValueError('Shape or dtype mismatch')
    cell = memory.restore(folder,interface(memory))
    if state_hash(cell) != expected['state']:
        raise ValueError('Restored state mismatch')
    return cell


def queries(memory):
    return torch.eye(memory.dim)[:4]


def writer(output):
    memory = initialize(41035)
    before = static_hash(memory)
    keys = queries(memory)
    vals = F.layer_norm(keys[[2,0,3,1]],(memory.dim,))
    alternative = vals.roll(1,0)
    a,b = (memory.create(mode='online_adapt') for _ in range(2))
    independent(a,b)
    equivalent(a,b)
    assert not storage(memory.initial.values()) & storage(a.fast.values())
    started = time.perf_counter()
    a = update(memory,a,b,keys,vals,keys,128)
    b = update(memory,b,a,keys,alternative,keys,128)
    write_seconds = time.perf_counter()-started
    records = {}

    def save(name, cell):
        started = time.perf_counter()
        output_hash = tensor_hash(read(memory,cell,keys))
        recall_seconds = time.perf_counter()-started
        entry = memory.checkpoint(cell,output/name,interface(memory))
        records[name] = dict(state=state_hash(cell), output=output_hash,
                             recall_seconds=recall_seconds,
                             shapes={kind:{k:dict(shape=list(v.shape),dtype=str(v.dtype))
                                           for k,v in getattr(cell,kind).items()}
                                     for kind in ('fast','momentum')}, **entry)

    save('a',a)
    save('b',b)
    parent = restore(memory,output/'a',records['a'])
    initial_parent = copy.deepcopy(state_hash(parent))
    a1,a2 = memory.fork(parent), memory.fork(parent)
    equivalent(a1,a2)
    independent(a1,a2)
    independent(a1,parent)
    independent(a2,parent)
    for child in (a1,a2):
        assert child.metadata['parent'] == {'cell_id':parent.metadata['cell_id'],
                                             'revision':parent.metadata['revision']}
    a1 = update(memory,a1,a2,keys,vals.roll(2,0),keys,128)
    a2 = update(memory,a2,a1,keys,vals.roll(3,0),keys,128)
    assert state_hash(parent) == initial_parent
    for cell,target in ((a1,vals.roll(2,0)),(a2,vals.roll(3,0))):
        predictions = (F.normalize(read(memory,cell,keys),dim=-1) @ F.normalize(target,dim=-1).T).argmax(-1)
        assert torch.equal(predictions,torch.arange(4))
    assert not torch.equal(read(memory,a1,keys),read(memory,a2,keys))
    save('a1',a1)
    save('a2',a2)
    assert static_hash(memory) == before and all(p.grad is None for p in memory.parameters())
    write(output/'inventory.json',dict(writer_pid=__import__('os').getpid(),cells=records,
                                      static_interface=interface(memory),ab_update_seconds=write_seconds,
                                      source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def negative_checks(memory, output, expected):
    checks = []
    for kind in ('missing_cell','missing_component','manifest_hash','bad_manifest','corrupt_tensor',
                 'component_hash','architecture','interface','shape','dtype'):
        folder = output/'negative'/kind
        receipt = copy.deepcopy(expected)
        if kind != 'missing_cell':
            shutil.copytree(output/'a',folder)
            path = folder/'manifest.json'
            meta = json.loads(path.read_text())
            tensor = folder/meta['generation']/'fast_weights.pt'
            if kind == 'missing_component':
                tensor.unlink()
            elif kind == 'manifest_hash':
                receipt['manifest_sha256'] = '0'*64
            elif kind == 'bad_manifest':
                path.write_text('{')
                receipt['manifest_sha256'] = file_hash(path)
            elif kind == 'component_hash':
                tensor.write_bytes(b'invalid')
            else:
                if kind == 'architecture':
                    meta['metadata']['architecture_id'] = 'wrong'
                elif kind == 'interface':
                    meta['compatibility']['interface'] = 0
                else:
                    if kind == 'corrupt_tensor':
                        tensor.write_bytes(b'invalid')
                    else:
                        values = load_checkpoint(tensor)
                        k = next(iter(values))
                        values[k] = values[k][..., :1] if kind == 'shape' else values[k].double()
                        save_checkpoint(values,tensor)
                    meta['files'][tensor.name] = file_hash(tensor)
                write(path,meta)
                receipt['manifest_sha256'] = file_hash(path)
        try:
            restore(memory,folder,receipt)
        except (FileNotFoundError,ValueError,RuntimeError,EOFError) as error:
            checks.append(dict(case=kind,error=str(error)))
        else:
            raise RuntimeError(f'Accepted invalid checkpoint: {kind}')
    return checks


def reader(output):
    memory = initialize(41035)
    inventory = json.loads((output/'inventory.json').read_text())
    pid = __import__('os').getpid()
    assert pid != inventory['writer_pid']
    cells = {}
    for name,entry in inventory['cells'].items():
        cell = restore(memory,output/name,entry)
        assert tensor_hash(read(memory,cell,queries(memory))) == entry['output']
        cells[name] = cell
    for a,b in __import__('itertools').combinations(cells.values(),2):
        independent(a,b)
    checks = negative_checks(memory,output,inventory['cells']['a'])
    write(output/'result.json',dict(status='PASS',reader_pid=pid,writer_pid=inventory['writer_pid'],
                                    restored=list(cells),negative=checks,torch=torch.__version__,device='cpu',
                                    manifest_sha256=file_hash(output/'inventory.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--phase',choices=['write','read'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert file_hash(M3/'memory.py') == file_hash(M3/'freeze/original-source/memory.py')
    if args.phase:
        (writer if args.phase == 'write' else reader)(args.output)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        for phase in ('write','read'):
            subprocess.run([sys.executable,__file__,str(args.output),'--phase',phase],check=True,timeout=120)
