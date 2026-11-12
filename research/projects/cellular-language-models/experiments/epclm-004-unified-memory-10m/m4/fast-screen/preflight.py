"""Synthetic fixture checks for masks, causal sessions and process restoration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from screening import (Factorial,MASKS,P,cross,datasets,fresh,train_diagnostic,
                       tensor_digest,write,episode,MQARConfig,sessions,wrapper,frozen_hash)
from adapter import save_checkpoint,load_checkpoint

MODEL=dict(vocab=1024,dim=16,layers=2,heads=2,kv_heads=2,ffn=32,window=8)
FUSION=dict(ssm_layers=1,state_size=4,memory_dim=4,memory_hidden=8,chunk=16,seed=51017)


def fixture(arm,device):
    return Factorial(arm,MODEL,FUSION).to(device)


def row():
    return episode(MQARConfig(bindings=4,length=64,queries=2,interference=1.,seed=50991),
                   'validation',0)


def verify_cached(model,ids):
    model.eval()
    model.memory_mode='online_adapt'
    with torch.no_grad():
        full=model(ids)
        model.reset_runtime()
        runtime=None
        cell=None
        parts=[]
        for part in ids.split(model.chunk,1):
            logits,cell,_,runtime=model.forward_state(part,cell,disabled=model.disabled,
                                                      runtime=runtime,cache=True)
            parts.append(logits)
        actual=torch.cat(parts,1)
        torch.testing.assert_close(actual,full,**P['cache_tolerance'])
        model.reset_runtime()
        try:
            model.forward_state(ids[:,:model.chunk],cell,disabled=model.disabled,runtime=runtime,cache=True)
        except ValueError as error:
            assert 'session' in str(error)
        else:
            raise AssertionError('Stale Runtime accepted')
    return float((actual-full).abs().max())


def run_checks(output,device):
    output.mkdir(parents=True,exist_ok=True)
    sample=row()
    prefix,suffix,positions,targets,padding=sessions([sample],16,device)
    assert prefix.shape==(1,64) and suffix.shape==(1,16)
    assert not ((suffix[:,:,None]==targets[:,None,:]).any())
    assert padding['prefix_padding']==7 and padding['query_padding']==9
    results={}
    for arm in MASKS:
        model=fixture(arm,device)
        ids=torch.tensor([sample['input_ids']],device=device)
        before=frozen_hash(model)
        counts={k:0 for k in ('swa','ssm','memory')}
        hooks=[]
        for k,m in [('swa',model.swa.final_norm),('ssm',model.ssm.final_norm),('memory',model.memory_norm)]:
            def observe(_m,_i,_o,key=k):
                counts[key]+=1
            hooks.append(m.register_forward_hook(observe))
        with torch.no_grad():
            logits,cell,gates,_=model.forward_state(ids,disabled=model.disabled)
        for h in hooks:h.remove()
        assert all(counts[k]==int(k not in model.disabled) for k in counts)
        assert (cell is None)==('memory' in model.disabled)
        assert torch.allclose(gates.sum(-1),torch.ones_like(gates[...,0]))
        for name in model.disabled:assert gates[...,model.branch_names.index(name)].eq(0).all()
        assert logits.shape==(1,64,1024)
        changed=ids.clone()
        changed[:,49:]=(changed[:,49:]+1)%1024
        with torch.no_grad():
            torch.testing.assert_close(model(changed)[:,:49],logits[:,:49],**P['cache_tolerance'])
        gradients=train_diagnostic(model,[sample])
        assert frozen_hash(model)==before
        parity=verify_cached(model,ids)
        folder=output/arm
        model.eval()
        matrix,resources=cross(model,[sample],folder/'sessions',dict(tokenizer='synthetic-fixture'))
        save_checkpoint({'model':dict(model.state_dict())},folder/'static.pt')
        subprocess.run([sys.executable,__file__,str(output),'--device',device,'--restore',arm],
                       check=True,timeout=180)
        restored=json.loads((folder/'fresh.json').read_text())
        assert all(r['pid']!=os.getpid() for r in restored.values())
        results[arm]=dict(status='PASS',calls=counts,gradient_norms=gradients,
                          cache_max_abs_error=parity,resources=resources,
                          conditions={k:r['state'] for k,r in matrix.items()},restore=restored)
    return dict(status='PASS',device=device,arms=results,optimizer_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--restore',choices=list(MASKS))
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.restore:
        model=fixture(args.restore,args.device).eval()
        folder=args.output/args.restore
        model.load_state_dict(load_checkpoint(folder/'static.pt')['model'])
        write(folder/'fresh.json',fresh(model,folder/'sessions'))
    else:
        write(args.output/'cpu-preflight.json',run_checks(args.output,args.device))
