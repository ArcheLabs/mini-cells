"""Sequential gates. Stop before E3 when direct Titans acquisition fails."""
import argparse
import json
from pathlib import Path
import uuid
import torch
from torch.nn import functional as F
from retrieval import *

def data_gate():
    pairs={}
    for split,seed in [('train',73001),('validation',73002)]:
        rows=[episode(split,seed,i) for i in range(4096 if split=='train' else 512)]
        pairs[split]={(int(k),int(v)) for r in rows for k,v in zip(r['keys'],r['values'])}
        assert set(k for k,v in pairs[split])==set(range(SYMBOLS))
        assert set(v for k,v in pairs[split])==set(range(SYMBOLS))
        for row in rows:
            oracle=dict(zip(row['keys'].tolist(),row['values'].tolist()))
            assert row['candidates'][row['targets']].tolist()==[oracle[k] for k in row['queries'].tolist()]
    assert not pairs['train'] & pairs['validation']
    partition=torch.Generator().manual_seed(73000)
    keys=torch.randperm(SYMBOLS,generator=partition).reshape(4,64)
    vals=torch.randperm(SYMBOLS,generator=partition).reshape(4,64)
    kg={int(k):g for g,row in enumerate(keys) for k in row}
    vg={int(v):g for g,row in enumerate(vals) for v in row}
    assert all((vg[v]-kg[k])%4==OFFSETS[s] for s,ps in pairs.items() for k,v in ps)
    return dict(status='PASS',oracle=1.,random=1/8,train_pairs=len(pairs['train']),
                validation_pairs=len(pairs['validation']),test='SEALED',dropout=0)


def direct_fixture():
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory=NeuralMemory(32,32,64)
            self.model_config={'fixture':'direct-v2'}
    torch.manual_seed(41035)
    model=Fixture().eval()
    k=torch.eye(32)[:8]
    v=F.layer_norm(k,(32,))
    labels=torch.tensor([2,0,3,1,6,4,7,5])
    return model,k,v,labels


def direct_write(model,k,v,mapping):
    c=model.memory.create(1,'online_adapt')
    initial=c.detached()
    trace=[]
    for _ in range(128):
        c,diag=model.memory.learn(k[None],v[mapping][None],c,'online_adapt')
        trace.append({n:float(x.mean()) for n,x in diag.items()})
    delta=sum(float((x.detach()-initial.fast[n]).square().sum()) for n,x in c.fast.items())**.5
    return c,trace,delta


def direct_read(model,k,cell):
    before=state_hash(cell)
    output=model.memory.recall(k[None],cell)[0]
    if state_hash(cell)!=before:
        raise RuntimeError("Read-only Cell mutated")
    return output


def direct_gate(out):
    model,k,v,labels=direct_fixture()
    correct,trace,delta=direct_write(model,k,v,labels)
    wrong,_,_=direct_write(model,k,v,labels.roll(1))
    y=direct_read(model,k,correct)
    logits=rank(y,v)
    negative=rank(direct_read(model,k,wrong),v)
    accuracy=float(logits.argmax(-1).eq(labels).float().mean())
    control=float(negative.argmax(-1).eq(labels).float().mean())
    ranks=(logits>logits.gather(1,labels[:,None])).sum(1)+1
    diagnostic=dict(ranks=ranks.tolist(),predictions=logits.argmax(-1).tolist(),
                    targets=labels.tolist(),cosine=F.cosine_similarity(y,v[labels]).tolist())
    if accuracy!=1 or control>=accuracy or delta<=0:
        return dict(status='BLOCKED',correct=accuracy,permuted=control,fast_delta=delta,trace=trace,per_association=diagnostic)
    identity=compatibility(model,'direct-fixture')
    identity['cell_interface_version']='separated-kvq-v2'
    export_cell(correct,out/'cell',identity,uuid.uuid4().hex)
    torch.testing.assert_close(direct_read(model,k,import_cell(out/'cell',model,identity)),y,atol=0,rtol=0)
    return dict(status='PASS',correct=accuracy,permuted=control,fast_delta=delta,trace=trace,
                cosine=float(F.cosine_similarity(y,v[labels]).mean()),
                momentum_norm=sum(float(t.square().sum()) for t in correct.momentum.values())**.5,
                restore_max_abs=0.)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    report={'data':data_gate(),'direct':direct_gate(out)}
    if report['direct']['status']=='PASS':
        model=Retrieval()
        row=episode('train',73001,0)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.003)
        history=[]
        for _ in range(64):
            optimizer.zero_grad(set_to_none=True)
            logits,trace=model(row)
            loss=F.cross_entropy(logits,row['targets'])
            loss.backward()
            gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
            assert gradients and all(torch.isfinite(torch.tensor(list(gradients.values()))))
            optimizer.step()
            accuracy=float(logits.argmax(-1).eq(row['targets']).float().mean())
            history.append(dict(loss=float(loss.detach()),accuracy=accuracy))
            if accuracy==1:
                break
        with torch.no_grad():
            logits,_=model(row)
            accuracy=float(logits.argmax(-1).eq(row['targets']).float().mean())
        report['fixed_episode']=dict(status='PASS' if accuracy==1 else 'BLOCKED',history=history,
                                     gradient_norms=gradients,trace=trace,
                                     backbone='frozen E3 initialization')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    run(args.output)
