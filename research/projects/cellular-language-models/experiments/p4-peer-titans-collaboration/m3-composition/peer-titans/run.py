"""P4-M3 peer composition smoke with independent writer and reader."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'m2-isolation'))
from verify import M3, independent, initialize, interface, read, restore, state_hash, tensor_hash, write
from memory import NeuralMemory, state_bytes

SEEDS = (53001, 53002, 53003)
EXPOSURES = 128


def batch_read(memory, a, b, queries):
    packed = replace(a, fast={k:torch.cat((v,b.fast[k])) for k,v in a.fast.items()},
                     momentum={k:torch.cat((v,b.momentum[k])) for k,v in a.momentum.items()})
    return memory.recall(queries,packed)


def evaluate(memory, a, b, q):
    snapshots = [state_hash(c) for c in (a,b)]
    started = time.perf_counter()
    serial = torch.stack([read(memory,c,k) for c,k in zip((a,b),q)])
    serial_seconds = time.perf_counter()-started
    started = time.perf_counter()
    batch = batch_read(memory,a,b,q)
    batch_seconds = time.perf_counter()-started
    torch.testing.assert_close(batch,serial,atol=0,rtol=0)
    crossed = batch_read(memory,a,b,q.flip(0))
    fusion = torch.cat(tuple(batch),dim=-1)
    only_a = torch.cat((batch[0],torch.zeros_like(batch[1])),dim=-1)
    only_b = torch.cat((torch.zeros_like(batch[0]),batch[1]),dim=-1)
    assert torch.equal(only_a[:,:32],fusion[:,:32]) and not only_a[:,32:].any()
    assert torch.equal(only_b[:,32:],fusion[:,32:]) and not only_b[:,:32].any()
    assert snapshots == [state_hash(c) for c in (a,b)]
    return dict(P1=serial,P2=batch,P3=serial,P4=crossed,P2_A=only_a,P2_B=only_b,
                fusion=fusion,serial_seconds=serial_seconds,batch_seconds=batch_seconds)


def generate(seed):
    rng = torch.Generator().manual_seed(seed)
    keys = F.normalize(torch.randn(8,32,generator=rng),dim=-1)
    candidates = F.normalize(torch.randn(8,32,generator=rng),dim=-1)*math.sqrt(32)
    mapping = torch.randperm(8,generator=rng)
    order = torch.stack([torch.randperm(4,generator=rng)+i for i in (0,4)])
    assert len(torch.unique(keys,dim=0)) == 8
    return keys,candidates,mapping,order


def learn(memory, cell, keys, values, order):
    memory.recall(keys[order][None],cell,'read_only')
    return memory.learn(keys[order][None],values[order][None],cell,'online_adapt')[0]


def save(memory, cell, path):
    stats = memory.checkpoint(cell,path,interface(memory))
    assert cell.update_step > 0
    return dict(state=state_hash(cell),static_parameters=sum(p.numel() for p in memory.parameters()),
                static_bytes=sum(p.numel()*p.element_size() for p in memory.parameters()),
                projection_parameters=sum(p.numel() for layer in (memory.key,memory.value,memory.query)
                                          for p in layer.parameters()),
                **state_bytes(cell),**stats)


def episode(root, seed, width):
    keys,candidates,mapping,order = generate(seed)
    values = candidates[mapping]
    memory = initialize(41035)
    a,b = (memory.create(mode='online_adapt') for _ in range(2))
    independent(a,b)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41035)
        union = NeuralMemory(256,32,width).requires_grad_(False)
    p0 = union.create(mode='online_adapt')
    static = (interface(memory),interface(union))
    before_a,before_b = state_hash(a),state_hash(b)
    started = time.perf_counter()
    for _ in range(EXPOSURES):
        a = learn(memory,a,keys,values,order[0])
    assert state_hash(b) == before_b and state_hash(a)['fast'] != before_a['fast']
    learned_a = state_hash(a)
    for _ in range(EXPOSURES):
        b = learn(memory,b,keys,values,order[1])
    assert state_hash(a) == learned_a and state_hash(b)['fast'] != before_b['fast']
    independent(a,b)
    for _ in range(EXPOSURES):
        for chunk in order:
            p0 = learn(union,p0,keys,values,chunk)
    acquire_seconds = time.perf_counter()-started
    assert (a.update_step,b.update_step,p0.update_step)==(128,128,256)
    assert static == (interface(memory),interface(union))
    assert all(t.isfinite().all() for c in (a,b,p0) for t in (*c.fast.values(),*c.momentum.values()))
    q = keys[order]
    result = evaluate(memory,a,b,q)
    result['P0'] = read(union,p0,q.flatten(0,1)).reshape(2,4,32)

    def hits(y, labels):
        scores = F.normalize(y,dim=-1) @ F.normalize(candidates,dim=-1).T
        return scores.argmax(-1).eq(labels)

    expected = mapping[order]
    metrics = {}
    for name in ('P0','P1','P2','P3','P4'):
        h = hits(result[name],expected.flip(0) if name == 'P4' else expected)
        metrics[name] = dict(slot_recall=h.float().mean(1).tolist(),joint=float(h.all(0).float().mean()))
        if name == 'P4':
            confidence = (F.normalize(result[name],dim=-1) @ F.normalize(candidates,dim=-1).T).amax(-1)
            metrics[name]['false_accept_at_0_8'] = (confidence >= .8).float().mean(1).tolist()
    metrics['P2_A'] = [float(hits(result['P2_A'][:,:32],expected[0]).float().mean()),None]
    metrics['P2_B'] = [None,float(hits(result['P2_B'][:,32:],expected[1]).float().mean())]
    assert metrics['P2_A'][0]==metrics['P2']['slot_recall'][0]
    assert metrics['P2_B'][1]==metrics['P2']['slot_recall'][1]
    assert metrics['P1']==metrics['P2']==metrics['P3']
    cells = {name:save(engine,cell,root/name) for name,engine,cell in
             [('a',memory,a),('b',memory,b),('p0',union,p0)]}
    return dict(seed=seed,width=width,keys=tensor_hash(keys),values=tensor_hash(values),
                mapping=mapping.tolist(),write_order=order.tolist(),query_order=order.tolist(),
                presentations={'P0':[128]*8,'peer':[128]*8},calls={'P0':256,'A':128,'B':128},
                read_calls={'acquisition':512,'P0':1,'P1_P3_shared_serial':2,'P2':1,'P4':1,
                            'ablations_additional':0},acquire_seconds=acquire_seconds,
                cells=cells,metrics=metrics,fusion_sha256=tensor_hash(result['fusion']),
                serial_seconds=result['serial_seconds'],batch_seconds=result['batch_seconds'])


def writer(root):
    rows=[]
    for width in (64,129):
        for seed in SEEDS:
            for index in range(5):
                identity=seed*100+index
                try:
                    rows.append(episode(root/f'{width}-{identity}',identity,width))
                except Exception as error:
                    write(root/'writer.json',dict(status='FAIL',rows=rows,failed=[width,identity],error=repr(error)))
                    raise
    write(root/'writer.json',dict(status='PASS',pid=os.getpid(),seeds=SEEDS,rows=rows,
                                  source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                  torch=torch.__version__,device='cpu'))


def reader(root, historical):
    receipt=json.loads((root/'writer.json').read_text())
    assert receipt['pid']!=os.getpid() and len(receipt['rows'])==30
    memory=initialize(41035)
    restored=[]
    for row in receipt['rows']:
        directory=root/f"{row['width']}-{row['seed']}"
        started=time.perf_counter()
        a,b=[restore(memory,directory/name,row['cells'][name]) for name in ('a','b')]
        restore_seconds=time.perf_counter()-started
        independent(a,b)
        keys,_,_,order=generate(row['seed'])
        result=evaluate(memory,a,b,keys[order])
        assert tensor_hash(result['fusion'])==row['fusion_sha256']
        restored.append(dict(seed=row['seed'],width=row['width'],seconds=restore_seconds))
    inventory=json.loads((historical/'inventory.json').read_text())
    a,b=[restore(memory,historical/name,inventory['cells'][name]) for name in ('a','b')]
    q=torch.eye(32)[:4]
    result=evaluate(memory,a,b,torch.stack((q,q)))
    assert [tensor_hash(result['P2'][i]) for i in (0,1)]==[inventory['cells'][n]['output'] for n in ('a','b')]
    write(root/'result.json',dict(status='PASS',writer_pid=receipt['pid'],reader_pid=os.getpid(),
                                  historical='PASS',restored=restored))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--phase',choices=['write','read'])
    args=p.parse_args()
    torch.set_num_threads(1)
    assert hashlib.sha256((M3/'memory.py').read_bytes()).digest()==hashlib.sha256((M3/'freeze/original-source/memory.py').read_bytes()).digest()
    if args.phase=='write':
        writer(args.output)
    elif args.phase=='read':
        reader(args.output,args.historical)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        for phase in ('write','read'):
            subprocess.run([sys.executable,__file__,str(args.output),'--historical',
                            str(args.historical),'--phase',phase],check=True,timeout=240)
