"""L1-C contextual, state and cross-process engineering gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from l1c import (MASKS,P,TRAIN,build,gradient_probe,train_sessions,expected_order,
                 frozen_hash,load_checkpoint,tensor_digest,write)
from dataset import CONFIG,bank,render
from evaluation import input_batch,at_query,persist,restore,evaluate_batch
from model import cell_hash
from adapter import save_checkpoint


def change_key(ids,position):
    copy=ids.clone()
    copy[:,position]=8+(copy[:,position]-8+11)%256
    return copy


def contextual_check(model,row,device):
    a,b,q,_,_=input_batch([row],device)
    query=int(q[0,0])
    with torch.no_grad():
        wa=model.context(a)
        wb=model.context(b)
        fact=model.context(change_key(a,2))
        recall=model.context(change_key(b,query-1))
        k,v,_=model.memory.project(wa)
        k_changed,v_changed,_=model.memory.project(fact)
        _,_,projected_q=model.memory.project(wb)
        _,_,q_changed=model.memory.project(recall)
        distances=dict(fact_context=float((wa[:,3]-fact[:,3]).norm()),
                       fact_key=float((k[:,3]-k_changed[:,3]).norm()),
                       fact_value=float((v[:,3]-v_changed[:,3]).norm()),
                       query_context=float((wb[:,query]-recall[:,query]).norm()),
                       query_projection=float((projected_q[:,query]-q_changed[:,query]).norm()))
        assert min(distances.values())>1e-7,distances
        later=b.clone()
        later[:,query+1:]=(later[:,query+1:]+7)%1024
        torch.testing.assert_close(wb[:,:query+1],model.context(later)[:,:query+1],**P['tolerance'])
        written=[]
        queried=[]
        key_hook=model.memory.key.register_forward_pre_hook(lambda _,x:written.append(x[0].detach()))
        query_hook=model.memory.query.register_forward_pre_hook(lambda _,x:queried.append(x[0].detach()))
        try:
            cell,_,_=model.write_a(a)
            assert cell.update_step==17
            torch.testing.assert_close(torch.cat(written,1),wa,**P['tolerance'])
            written.clear();queried.clear()
            before=cell_hash(cell)
            output=model.read_b(b,cell)[0]
            assert cell_hash(cell)==before
            assert len(queried)==1
            torch.testing.assert_close(queried[0],wb,**P['tolerance'])
        finally:
            key_hook.remove();query_hook.remove()
        changed=model.read_b(later,cell)[0]
        torch.testing.assert_close(output[:,:query+1],changed[:,:query+1],**P['tolerance'])
        # With both stateful branches masked, an unrelated A session cannot
        # influence B and no Cell is carried from A.
        no_memory=build('F00',device,True).eval()
        different_a=change_key(a,2)
        a1=no_memory(torch.cat((a,b),1))
        a2=no_memory(torch.cat((different_a,b),1))
        torch.testing.assert_close(a1,a2,atol=0,rtol=0)
    return distances


def checks(output,device):
    output.mkdir(parents=True,exist_ok=True)
    sample=render(CONFIG,'validation',0)
    train=[render(CONFIG,'train',i) for i in range(2)]
    evidence={}
    for arm in MASKS:
        model=build(arm,device,True)
        frozen=frozen_hash(model)
        init=tensor_digest(model.state_dict())
        grads=gradient_probe(model,[sample])
        a,b,_,_,_=input_batch([sample],device)
        model.eval()
        # Compare the same stream when the other episode is different.
        second=render(CONFIG,'validation',1)
        other_a,other_b,_,_,_=input_batch([second],device)
        with torch.no_grad():
            joined_a=torch.cat((a,other_a))
            joined_b=torch.cat((b,other_b))
            cells,_,_=model.write_a(joined_a)
            changed=joined_a.clone()
            changed[1,2]=8+(int(changed[1,2])-8+1)%256
            other_cells,_,_=model.write_a(changed)
            left=model.read_b(joined_b,cells)[0][0]
            right=model.read_b(joined_b,other_cells)[0][0]
            torch.testing.assert_close(left,right,atol=0,rtol=0)
        counts={'ssm':0,'memory':0,'swa':0}
        hooks=[]
        for name,module in [('ssm',model.ssm.final_norm),('memory',model.memory_norm),('swa',model.swa.final_norm)]:
            def count(_m,_i,_o,key=name):
                counts[key]+=1
            hooks.append(module.register_forward_hook(count))
        with torch.no_grad():
            model(torch.cat((a,b),1))
        for hook in hooks:hook.remove()
        assert counts['swa']==(1 if 'memory' in model.disabled else 2)
        assert counts['ssm']==int('ssm' not in model.disabled)
        assert counts['memory']==int('memory' not in model.disabled)
        context=contextual_check(model,sample,device) if arm=='F11' else None
        if arm=='F11':
            model.train()
            a,b,q,target,_=input_batch([sample],device)
            cell,_,_=model.write_a(a)
            loss=torch.nn.functional.cross_entropy(at_query(model.read_b(b,cell)[0],q).flatten(0,1),target.flatten())
            params=[cell.fast['w1'],model.memory.initial['w1'],model.memory.key.weight,
                    model.memory.value.weight,model.memory.query.weight]
            paths=[float(g.norm()) for g in torch.autograd.grad(loss,params)]
            assert min(paths)>0 and all(torch.isfinite(torch.tensor(paths)))
        else:
            paths=None
        folder=output/arm
        folder.mkdir(exist_ok=True)
        ledger=None
        if device=='cpu':
            identity=dict(fixture='l1c-cpu',arm=arm)
            result=train_sessions(model,train,[sample],folder/'fit',identity,float('inf'),
                                  {**TRAIN,'batch':2,'updates':1})
            ledger=result['ledger']
            assert (ledger['committed_updates'],ledger['input_tokens'],ledger['query_target_tokens'])==(1,1024,16)
            assert load_checkpoint(folder/'fit/checkpoint.pt')['identity']==identity
        assert frozen_hash(model)==frozen
        model.eval()
        persist(model,[sample],folder/'session',dict(fixture='l1c',arm=arm),P['tolerance'])
        save_checkpoint({'model':dict(model.state_dict())},folder/'fixture.pt')
        subprocess.run([sys.executable,__file__,str(output),'--device',device,'--restore-arm',arm],
                       check=True,timeout=180)
        proof=json.loads((folder/'fresh.json').read_text())
        assert proof['pid']!=os.getpid()
        measured=evaluate_batch(model,[sample],arm=='F11')
        evidence[arm]=dict(initial_sha256=init,gradient_norms=grads,forward_calls=counts,
                           meta_gradient_paths=paths,context=context,fixture_ledger=ledger,
                           restore=proof,gates=measured['gates'])
    assert len({r['initial_sha256'] for r in evidence.values()})==1
    return dict(status='PASS',device=device,arms=evidence,official_optimizer_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--restore-arm',choices=list(MASKS))
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.restore_arm:
        folder=args.output/args.restore_arm
        model=build(args.restore_arm,args.device,True).eval()
        model.load_state_dict(load_checkpoint(folder/'fixture.pt')['model'])
        write(folder/'fresh.json',restore(model,folder/'session',P['tolerance']))
    else:
        train,validation,evaluation,data=bank()
        order=expected_order(train)
        assert len({episode for row in order for episode in row['ids']})==128
        result=checks(args.output,args.device)
        write(args.output/'preflight.json',result)
        write(args.output/'dataset-manifest.json',data)
