"""Sequential retrieval and portability screening. Independent Test is sealed."""
import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from retrieval import BASE, Retrieval, compatibility, episode, export_cell, import_cell, rank, state_hash
from preflight007 import run as preflight
from e0 import load_checkpoint, save_checkpoint, write, rng, restore_rng
from data import digest, file_hash

P=json.loads((BASE/'config.json').read_text())


def row_on(split,seed,index,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v
            for k,v in episode(split,seed,index).items()}


@torch.no_grad()
def evaluate(model,rows):
    results=[]
    for row in rows:
        k,v,q,candidates=model.represent(row)
        correct,trace=model.acquire_vectors(k,v,P['inner_updates'],'online_adapt')
        wrong,_=model.acquire_vectors(k,v.roll(1,0),P['inner_updates'],'online_adapt')
        before=state_hash(correct)
        read=model.memory.recall(q[None],correct)[0]
        good=rank(read,candidates)
        bad=rank(model.memory.recall(q[None],wrong)[0],candidates)
        # C0 is a uniform no-Cell control. Candidate order is independently shuffled.
        empty=torch.zeros_like(good)
        assert state_hash(correct)==before
        target=row['targets']
        metrics={}
        for name,logits in [('C0',empty),('C1',good),('C2',bad),('C3',good)]:
            pred=logits.argmax(-1)
            metrics[name]=dict(accuracy=float(pred.eq(target).float().mean()),
                               prediction=pred.tolist(),nll=float(F.cross_entropy(logits,target)))
        score=good.gather(1,target[:,None])
        other=good.masked_fill(F.one_hot(target,8).bool(),float('-inf')).amax(1)
        results.append(dict(id=row['id'],metrics=metrics,targets=target.tolist(),trace=trace,
                            read_cosine=F.cosine_similarity(read,candidates[target]).tolist(),
                            margin=(score[:,0]-other).tolist(),logit_delta=float((good-bad).abs().mean())))
    return results


def analyze(rows):
    rng=np.random.default_rng(P['bootstrap_seed'])
    a={c:np.array([r['metrics'][c]['accuracy'] for r in rows]) for c in ('C0','C1','C2')}
    result={c:float(v.mean()) for c,v in a.items()}
    result['C1_nll']=float(np.mean([r['metrics']['C1']['nll'] for r in rows]))
    for c in ('C0','C2'):
        diff=a['C1']-a[c]
        draws=rng.integers(len(diff),size=(P['bootstrap_replicates'],len(diff)))
        result['C1-'+c]=dict(mean=float(diff.mean()),ci95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist())
    result['unique_candidate_predictions']=len({p for r in rows for p in r['metrics']['C1']['prediction']})
    return result


def build(device):
    torch.manual_seed(P['model_seed'])
    return Retrieval().to(device)


@torch.no_grad()
def fresh(out):
    model=build('cuda').eval()
    checkpoint=out/'checkpoint.pt'
    model.load_state_dict(load_checkpoint(checkpoint)['model'])
    identity=compatibility(model,file_hash(checkpoint))
    identity['cell_interface_version']='separated-kvq-v2'
    cell=import_cell(out/'cell',model,identity)
    row=row_on('validation',P['validation_seed'],0,'cuda')
    _,_,q,candidates=model.represent(row)
    before=state_hash(cell)
    actual=rank(model.memory.recall(q[None],cell)[0],candidates)
    torch.testing.assert_close(actual.cpu(),load_checkpoint(out/'expected.pt'),atol=2e-5,rtol=2e-5)
    assert state_hash(cell)==before
    write(out/'fresh.json',dict(status='PASS',source_runtime_transferred=False,read_only=True))


def run(out,started):
    out.mkdir(parents=True,exist_ok=True)
    proof=out/'cpu'
    if not (proof/'report.json').exists():
        preflight(proof)
    gates=json.loads((proof/'report.json').read_text())
    if any(gates.get(k,{}).get('status')!='PASS' for k in ('data','fixed_episode')):
        failed=next(k for k in ('data','fixed_episode') if gates.get(k,{}).get('status')!='PASS')
        write(out/'decision.json',dict(status='BLOCKED',gate=failed,official_updates=0))
        return
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Tesla T4 required after CPU gates')
    device='cuda'
    model=build(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=P['lr'])
    train=[row_on('train',P['train_seed'],i,device) for i in range(4096)]
    val=[row_on('validation',P['validation_seed'],i,device) for i in range(P['validation_episodes'])]
    def update(group):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits=torch.cat([model(row)[0] for row in group])
        loss=F.cross_entropy(logits,torch.cat([row['targets'] for row in group]))
        assert loss.isfinite()
        loss.backward()
        gradients={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert len(gradients)==sum(p.requires_grad for p in model.parameters())
        assert np.isfinite(list(gradients.values())).all()
        return float(loss.detach()),gradients
    t=time.monotonic()
    update(train[:4])
    torch.cuda.synchronize()
    step=time.monotonic()-t
    t=time.monotonic()
    evaluate(model.eval(),val[:4])
    torch.cuda.synchronize()
    estimate=1.5*(P['updates']*step+5*len(val)*(time.monotonic()-t)/4)+600
    deadline=started+P['execution_seconds']
    write(out/'budget.json',dict(projected_seconds=estimate,optimizer_updates=0,peak_bytes=torch.cuda.max_memory_allocated()))
    if time.monotonic()+estimate>=deadline:
        write(out/'decision.json',dict(status='BLOCKED',gate='T4_BUDGET',official_updates=0))
        return
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    identity=dict(source=source,config_sha256=digest(P),train_sha256=digest([r['id'] for r in train]),
                  validation_sha256=digest([r['id'] for r in val]),gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__,cuda=torch.version.cuda,precision='fp32',tf32=False)
    write(out/'manifest.json',identity)
    checkpoint=out/'checkpoint.pt'
    history=[]
    curves=[]
    start=0
    if checkpoint.exists():
        saved=load_checkpoint(checkpoint)
        assert saved['identity']==identity
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        history,curves=saved['history'],saved['curves']
        start=len(history)
    if not curves:
        curves.append(analyze(evaluate(model.eval(),val)))
    for i in range(start,P['updates']):
        if time.monotonic()>=deadline:
            raise TimeoutError('Screening deadline; last durable checkpoint retained')
        group=train[i*4:i*4+4]
        loss,gradients=update(group)
        optimizer.step()
        history.append(dict(update=i+1,loss=loss,gradients=gradients,ids=[r['id'] for r in group]))
        if (i+1)%256==0:
            rows=evaluate(model.eval(),val)
            curves.append(analyze(rows))
            write(out/f'validation-{i+1}.json',rows)
        if (i+1)%32==0:
            save_checkpoint(dict(identity=identity,model=dict(model.state_dict()),optimizer=optimizer.state_dict(),
                                 history=history,curves=curves,rng=rng(True)),checkpoint)
    final=curves[-1]
    passed=(final['C1']>=P['candidate_top1_min'] and final['unique_candidate_predictions']>1
            and all(final[c]<=P['control_max'] and final['C1-'+c]['ci95'][0]>0 for c in ('C0','C2'))
            and final['C1']>curves[0]['C1'] and final['C1_nll']<curves[0]['C1_nll'])
    write(out/'training.json',dict(history=history,curves=curves,targets=16*len(history)))
    if not passed:
        write(out/'decision.json',dict(status='BLOCKED',gate='HELDOUT_RETRIEVAL',metrics=final))
        return
    with torch.no_grad():
        row=val[0]
        k,v,q,candidates=model.eval().represent(row)
        cell,_=model.acquire_vectors(k,v,P['inner_updates'],'online_adapt')
        identity=compatibility(model,file_hash(checkpoint))
        identity['cell_interface_version']='separated-kvq-v2'
        if not (out/'cell').exists():
            export_cell(cell,out/'cell',identity,uuid.uuid4().hex)
        save_checkpoint(rank(model.memory.recall(q[None],cell)[0],candidates).cpu(),out/'expected.pt')
    subprocess.run([sys.executable,__file__,str(out),'--fresh'],check=True,
                          timeout=max(1,min(180,deadline-time.monotonic())))
    write(out/'portability.json',dict(C3='source-side canonical read recorded in validation-1024.json',
                                    C3_reset='fresh canonical query against the carried Cell',
                                    C4=json.loads((out/'fresh.json').read_text())))
    write(out/'decision.json',dict(status='SCREENING_COMPLETE',metrics=final,test='SEALED',confirmation='PENDING'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--started',type=float,default=time.monotonic())
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.fresh:
        fresh(args.output)
    else:
        run(args.output,args.started)
