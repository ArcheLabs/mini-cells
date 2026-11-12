"""CPU/T4 checks for E3, canonical segments and fresh-process portability."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from portable_model import M2,P,POSITIONS,PortableModel,state_hash
from portability_data import episode,datasets
from cell_artifact import compatibility,CellSlot,export_cell
from harness import lifecycle,tensors,trial,runtime_bytes
from data import batch,file_hash
from e0 import load_checkpoint,masked_loss,save_checkpoint,write
from training import fit

TRAIN={**M2['training'],'seed':P['model_seed'],'batch':P['batch'],'updates':P['updates']}


def gradient(model,rows):
    model.train()
    b=batch(rows,next(model.parameters()).device)
    loss=masked_loss(model(b.input_ids),b.labels)
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    paths={n:float(p.grad.norm()) for n,p in model.named_parameters()
           if n in ('memory.initial.w1','memory.key.weight','memory.value.weight','memory.query.weight')}
    assert all(x>0 for x in paths.values()),paths
    model.zero_grad(set_to_none=True)
    return paths


@torch.no_grad()
def backbone_checks(model,row):
    x=tensors(model,row)
    backbone=model.backbone
    ids=x['target_ids']
    direct=backbone(ids)
    first,runtime=backbone.cached(ids[:,:16])
    rest,continued=backbone.cached(ids[:,16:],runtime)
    torch.testing.assert_close(torch.cat((first,rest),1),direct,**P['tolerance'])
    altered=ids.clone()
    altered[:,20:]=(altered[:,20:]+7)%1024
    torch.testing.assert_close(backbone(altered)[:,:20],direct[:,:20],**P['tolerance'])
    original_pair=backbone(torch.cat((ids,ids)))
    perturbed_pair=backbone(torch.cat((ids,altered)))
    torch.testing.assert_close(original_pair[0],perturbed_pair[0],atol=0,rtol=0)
    long=ids.repeat(1,3)
    _,long_runtime=backbone.cached(long)
    assert all(pair[0].shape[2]==model.model_config['window'] for pair in long_runtime.kv_cache.values())
    packet=x['query_packets'][:,0]
    canonical=model.encode_cell_segment(packet)
    conditioned,_=backbone.cached(packet,runtime)
    difference=float((canonical-conditioned[:,-1]).abs().max())
    assert difference>0
    torch.testing.assert_close(canonical,model.encode_cell_segment(packet),atol=0,rtol=0)
    k,v,q=model.memory.project(canonical[:,None])
    assert k.shape==v.shape==q.shape==(1,1,model.memory.dim)
    return dict(full_cached_max_abs=float((torch.cat((first,rest),1)-direct).abs().max()),
                i0_i1_max_abs=difference,window_tokens=model.model_config['window'],
                isolated_streams=True,causal=True,**runtime_bytes(continued))


def prepare_restore(out,model,row,identity,expected):
    x=tensors(model,row)
    payload=dict(target=x['target_ids'].cpu(),packets=x['query_packets'].cpu(),expected=expected.cpu())
    with torch.no_grad():
        first,rt=model.backbone.cached(x['target_ids'][:,:16])
        tail,_=model.backbone.cached(x['target_ids'][:,16:],rt)
        payload.update(mamba={k:tuple(t.cpu() for t in v) for k,v in rt.mamba_state.items()},
                       kv={k:tuple(t.cpu() for t in v) for k,v in rt.kv_cache.items()},
                       position=rt.position,expected_tail=tail.cpu())
    save_checkpoint(payload,out/'restore-input.pt')
    write(out/'restore-identity.json',identity)


@torch.no_grad()
def restore_check(out,device,fixture,checkpoint):
    from baselines import Runtime
    model=PortableModel(fixture).to(device).eval()
    model.load_state_dict(load_checkpoint(checkpoint)['model'],strict=True)
    identity=json.loads((out/'restore-identity.json').read_text())
    assert identity==compatibility(model,file_hash(checkpoint))
    data=load_checkpoint(out/'restore-input.pt')
    slot=CellSlot(model,identity)
    cell=slot.replace(out/'artifacts/correct')
    before=state_hash(cell)
    hidden,_=model.backbone.cached(data['target'].to(device))
    actual=slot.read(hidden[:,POSITIONS],data['packets'].to(device)).cpu()
    torch.testing.assert_close(actual,data['expected'],**P['tolerance'])
    assert state_hash(cell)==before
    backbone=model.backbone
    moved=lambda group:{k:tuple(t.to(device) for t in pair) for k,pair in group.items()}
    runtime=Runtime(id(backbone),data['position'],moved(data['mamba']),moved(data['kv']),backbone._version())
    tail,_=backbone.cached(data['target'][:,16:].to(device),runtime)
    torch.testing.assert_close(tail.cpu(),data['expected_tail'],**P['tolerance'])
    write(out/'fresh-restore.json',dict(status='PASS',pid=os.getpid(),
                                       cell_output_max_abs=float((actual-data['expected']).abs().max()),
                                       session_output_max_abs=float((tail.cpu()-data['expected_tail']).abs().max())))


def fresh_process(out,device,fixture,checkpoint):
    argv=[sys.executable,__file__,str(out),'--device',device,'--restore',str(checkpoint)]
    if fixture:
        argv+=['--fixture']
    subprocess.run(argv,check=True,timeout=180)
    result=json.loads((out/'fresh-restore.json').read_text())
    assert result['pid']!=os.getpid()
    return result


def checks(out,device):
    out.mkdir(parents=True,exist_ok=True)
    model=PortableModel().to(device).eval()
    params=sum(p.numel() for p in model.backbone.parameters())
    assert params==10135808
    sample=episode('validation',62900,0)
    evidence=backbone_checks(model,sample)
    del model
    model=PortableModel(True).to(device)
    gradients=gradient(model,[sample])
    ledger=None
    if device=='cpu':
        train=[episode('train',62901,i) for i in range(4)]
        result=fit(model,train,[sample],{**TRAIN,'updates':1},out/'fixture-training',{'fixture':'epclm006'})
        ledger=result['ledger']
        assert (ledger['committed_updates'],ledger['input_tokens'],ledger['query_target_tokens'])==(1,256,16)
    model.eval()
    path=out/'fixture.pt'
    save_checkpoint({'model':dict(model.state_dict())},path)
    identity=compatibility(model,file_hash(path))
    _,expected=trial(model,sample,identity,out/'artifacts')
    life=lifecycle(model,sample,identity,out/'artifacts')
    prepare_restore(out,model,sample,identity,expected)
    fresh=fresh_process(out,device,True,path)
    return dict(status='PASS',device=device,e3_parameters=params,backbone=evidence,
                gradient_paths=gradients,fixture_ledger=ledger,lifecycle=life,fresh_restore=fresh,
                official_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--restore',type=Path)
    p.add_argument('--fixture',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.restore:
        restore_check(args.output,args.device,args.fixture,args.restore)
    else:
        write(args.output/'preflight.json',checks(args.output,args.device))
