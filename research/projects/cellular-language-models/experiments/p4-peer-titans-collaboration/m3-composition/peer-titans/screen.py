"""Paired CPU screening using the frozen Stage 2 read and persistence helpers."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from run import (M3, NeuralMemory, evaluate, generate, independent, initialize,
                 interface, learn, read, restore, save, state_hash, tensor_hash, write)

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE/'screening.json').read_text())


def rng(seed, stream):
    digest = hashlib.sha256(f'p4-stage3:{seed}:{stream}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], 'big'))


def dataset(seed, n):
    generator = rng(seed, 'association')
    keys = F.normalize(torch.randn(2,16,32,generator=generator),dim=-1)
    values = F.normalize(torch.randn(2,16,32,generator=generator),dim=-1)*math.sqrt(32)
    for domain in range(2):
        values[domain] = values[domain,torch.randperm(16,generator=generator)]
    keys,values = keys[:,:n].contiguous(),values[:,:n].contiguous()
    assert len(torch.unique(keys.flatten(0,1),dim=0))==2*n
    orders = []
    for stream in ('acquire','query','wrong'):
        g = rng(seed,stream)
        orders.append(torch.stack([torch.randperm(16,generator=g) for _ in range(2)]))
    orders = [torch.stack([row[row<n] for row in order]) for order in orders]
    unknown = F.normalize(torch.randn(2,n,32,generator=rng(seed,'unknown')),dim=-1)
    assert not (unknown.flatten(0,1)[:,None]==keys.flatten(0,1)).all(-1).any()
    return keys,values,*orders,unknown


def acquire(memory, cell, keys, values, chunks):
    counts=torch.zeros(len(keys),dtype=torch.long)
    started=time.perf_counter()
    for _ in range(CONFIG['exposure']):
        for chunk in chunks:
            cell=learn(memory,cell,keys,values,chunk)
            counts[chunk]+=1
    assert (counts==CONFIG['exposure']).all()
    assert cell.update_step==CONFIG['exposure']*len(chunks)
    assert all(t.isfinite().all() for t in (*cell.fast.values(),*cell.momentum.values()))
    return cell,dict(calls=cell.update_step,presentations=int(counts.sum()),counts=counts.tolist(),
                     seconds=time.perf_counter()-started)


def score(outputs, candidates, labels, early):
    similarities=F.normalize(outputs,dim=-1)@F.normalize(candidates,dim=-1).T
    confidence,prediction=similarities.max(-1)
    correct=prediction.eq(labels)
    return dict(recall=correct.float().mean(1).tolist(),joint=float(correct.all(0).float().mean()),
                early=[float(correct[i,early[i]].float().mean()) for i in range(2)],
                late=[float(correct[i,~early[i]].float().mean()) for i in range(2)],
                cosine=similarities.gather(-1,labels.unsqueeze(-1)).squeeze(-1).mean(1).tolist(),
                mse=(outputs-candidates[labels]).square().mean((1,2)).tolist(),
                incorrect_accept=((confidence>=CONFIG['threshold'])&~correct).sum(1).tolist())


def rejection(outputs,candidates):
    scores=(F.normalize(outputs,dim=-1)@F.normalize(candidates,dim=-1).T).amax(-1)
    return dict(accepted=(scores>=CONFIG['threshold']).sum(1).tolist(),queries=scores.shape[1],
                scores=scores.tolist())


def episode(path, seed, n, width):
    started=time.perf_counter()
    keys,values,order,query,wrong,unknown=dataset(seed,n)
    candidate=values.flatten(0,1)
    memory=initialize(41035)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41035)
        union=NeuralMemory(256,32,width).requires_grad_(False)
    static=[interface(memory),interface(union)]
    a,b=[memory.create(mode='online_adapt') for _ in range(2)]
    independent(a,b)
    b_before=state_hash(b)
    a,aa=acquire(memory,a,keys[0],values[0],list(order[0].split(CONFIG['chunk'])))
    assert state_hash(b)==b_before
    a_before=state_hash(a)
    b,bb=acquire(memory,b,keys[1],values[1],list(order[1].split(CONFIG['chunk'])))
    assert state_hash(a)==a_before
    independent(a,b)
    p0=union.create(mode='online_adapt')
    schedule=[c for pair in zip(order[0].split(4),(order[1]+n).split(4)) for c in pair]
    p0,uu=acquire(union,p0,keys.flatten(0,1),candidate,schedule)
    assert uu['counts']==aa['counts']+bb['counts']
    assert static==[interface(memory),interface(union)]
    q=torch.stack([keys[i,query[i]] for i in range(2)])
    wrong_q=torch.stack([keys[i,wrong[i]] for i in range(2)])
    labels=query+torch.tensor([[0],[n]])
    early=torch.stack([torch.isin(query[i],order[i,:n//2]) for i in range(2)])
    result=evaluate(memory,a,b,q)
    started_union=time.perf_counter()
    union_output=read(union,p0,q.flatten(0,1)).reshape(2,n,32)
    union_seconds=time.perf_counter()-started_union
    metrics={k:score(result[k],candidate,labels,early) for k in ('P1','P2','P3')}
    metrics['P0']=score(union_output,candidate,labels,early)
    crossed=evaluate(memory,a,b,wrong_q)['P4']
    metrics['P4']=score(crossed,candidate,(wrong+torch.tensor([[0],[n]])).flip(0),early)
    metrics['P2_A']=[metrics['P2']['recall'][0],None]
    metrics['P2_B']=[None,metrics['P2']['recall'][1]]
    wrong_metrics=rejection(crossed,candidate)
    unknown_metrics={ 'P0':rejection(read(union,p0,unknown.flatten(0,1)).reshape(2,n,32),candidate),
                      'P2':rejection(torch.stack([read(memory,c,unknown[i]) for i,c in enumerate((a,b))]),candidate)}
    cells={name:save(engine,c,path/name) for name,engine,c in
           [('a',memory,a),('b',memory,b),('p0',union,p0)]}
    sizes={k:v['fast_weights']+v['momentum'] for k,v in cells.items()}
    assert sizes==dict(a=34048,b=34048,p0={64:34048,129:67848}[width])
    return dict(seed=seed,n=n,width=width,task_hashes=[tensor_hash(t) for t in
                (keys,values,order,query,wrong,unknown)],write_order=order.tolist(),
                query_order=query.tolist(),wrong_order=wrong.tolist(),early_mask=early.tolist(),
                acquisition=dict(a=aa,b=bb,p0=uu),cells=cells,metrics=metrics,
                wrong=wrong_metrics,unknown=unknown_metrics,threshold=CONFIG['threshold'],
                temp_batch_state_bytes=sizes['a']+sizes['b'],read_calls=dict(peer_serial=6,peer_batch=4,union=2),
                serial_seconds=result['serial_seconds'],batch_seconds=result['batch_seconds'],
                union_seconds=union_seconds,episode_seconds=time.perf_counter()-started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                fusion_sha256=tensor_hash(result['fusion']),union_sha256=tensor_hash(union_output))


def writer(output, calibration):
    config_hash=hashlib.sha256((BASE/'screening.json').read_bytes()).hexdigest()
    files=(BASE/'screen.py',BASE/'screening.json',BASE/'run.py',M3/'memory.py',
           BASE.parents[1]/'m2-isolation/verify.py',BASE.parents[1]/'run_m1.py')
    identity=dict(config=CONFIG,config_sha256=config_hash,calibration=calibration,
                  source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  torch=torch.__version__,python=sys.version,device='cpu',dtype='float32',
                  cpu_threads=torch.get_num_threads(),pid=os.getpid())
    write(output/'manifest.json',identity)
    for n in CONFIG['sizes']:
        for width in CONFIG['union_widths']:
            cohorts={'73901':1} if calibration else CONFIG['seed_counts']
            for seed,count in cohorts.items():
                for i in range(count):
                    episode_seed=int(seed)*100+i
                    name=f'{n}-{width}-{episode_seed}'
                    try:
                        result=episode(output/'checkpoints'/name,episode_seed,n,width)
                        result.update(status='PASS',cohort=int(seed),episode=name)
                        write(output/'episodes'/f'{name}.json',result)
                    except BaseException as error:
                        write(output/'episodes'/f'{name}.json',dict(status='FAIL',episode=name,error=repr(error)))
                        raise
    assert hashlib.sha256((BASE/'screening.json').read_bytes()).hexdigest()==config_hash


def reader(output):
    started=time.perf_counter()
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['pid']!=os.getpid()
    assert hashlib.sha256((BASE/'screening.json').read_bytes()).hexdigest()==manifest['config_sha256']
    rows=[json.loads(f.read_text()) for f in sorted((output/'episodes').glob('*.json'))]
    assert len(rows)==(4 if manifest['calibration'] else 80)
    identity={}
    restored=[]
    for r in rows:
        assert r['status']=='PASS'
        n,width,seed=r['n'],r['width'],r['seed']
        keys,values,order,query,wrong,unknown=dataset(seed,n)
        hashes=[tensor_hash(t) for t in (keys,values,order,query,wrong,unknown)]
        assert hashes==r['task_hashes']==identity.setdefault((n,seed),hashes)
        memory=initialize(41035)
        q=torch.stack([keys[i,query[i]] for i in range(2)])
        t=time.perf_counter()
        path=output/'checkpoints'/r['episode']
        a,b=[restore(memory,path/k,r['cells'][k]) for k in ('a','b')]
        independent(a,b)
        assert tensor_hash(evaluate(memory,a,b,q)['fusion'])==r['fusion_sha256']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41035)
            union=NeuralMemory(256,32,width).requires_grad_(False)
        p0=restore(union,path/'p0',r['cells']['p0'])
        assert tensor_hash(read(union,p0,q.flatten(0,1)).reshape(2,n,32))==r['union_sha256']
        restored.append(dict(episode=r['episode'],seconds=time.perf_counter()-t))
    write(output/'restore.json',dict(status='PASS',pid=os.getpid(),writer_pid=manifest['pid'],
                                     episodes=restored,seconds=time.perf_counter()-started))


def run(output,calibration):
    output.mkdir(parents=True,exist_ok=False)
    for phase in ('write','read'):
        args=[sys.executable,__file__,str(output),'--phase',phase]
        if calibration:
            args+=['--calibration']
        subprocess.run(args,check=True,timeout=900)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--calibration',action='store_true')
    parser.add_argument('--phase',choices=['write','read'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    assert (M3/'memory.py').read_bytes()==(M3/'freeze/original-source/memory.py').read_bytes()
    if args.phase=='write':
        writer(args.output,args.calibration)
    elif args.phase=='read':
        reader(args.output)
    else:
        run(args.output,args.calibration)
