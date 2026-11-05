"""FP32 query-target trainer, complete episodes and source-bound atomic checkpoints."""
import csv
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model,dev,path=None):
    model.eval();x,y,metadata=dev;device=next(model.parameters()).device;rows=[]
    with torch.inference_mode():
        for start in range(0,len(x),16):
            ids=torch.as_tensor(x[start:start+16],device=device);labels=torch.as_tensor(y[start:start+16],device=device)
            logits,_=model(ids);valid=labels!=-100;loss=F.cross_entropy(logits[valid],labels[valid],reduction='none')
            assert torch.isfinite(loss).all(),'nonfinite evaluation loss'
            predictions=logits[valid].argmax(-1).cpu().tolist();losses=loss.cpu().tolist();offset=0
            for episode in metadata[start:start+16]:
                for q in episode:
                    rows.append({**q,'prediction':predictions[offset],'correct':int(predictions[offset]==q['target']),'nll':losses[offset]});offset+=1
    if path:
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    groups={'all':rows}
    for row in rows:
        groups.setdefault(row['condition'],[]).append(row)
        band='key_inside16' if row['binding_in16'] else ('key_inside64_out16' if row['binding_in64'] else 'key_outside64')
        groups.setdefault(band,[]).append(row)
    for name,group in groups.items():
        episodes={}
        for r in group:episodes.setdefault(r['episode'],[]).append(r['correct'])
        means=np.array([np.mean(v) for v in episodes.values()])
        summary[name]={'episode_mean_em':float(means.mean()),'episode_sd':float(means.std(ddof=1)) if len(means)>1 else None,
                       'query_nll':float(np.mean([r['nll'] for r in group])),'episodes':len(episodes),'queries':len(group)}
    for name,group in [('repeat_first',[r for r in rows if r['task']=='repeat' and r['occurrence']==1]),
                       ('repeat_later',[r for r in rows if r['task']=='repeat' and r['occurrence']>1])]:
        summary[name]={'query_em':float(np.mean([r['correct'] for r in group])),'queries':len(group)}
    model.train();return summary


def fit(model,train,dev,p,folder,identity,deadline,atomic_save,checkpoint_hook=None):
    folder.mkdir(exist_ok=True);path=folder/'checkpoint.pt';device=next(model.parameters()).device
    c=p['training'];optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=tuple(c['betas']),weight_decay=c['weight_decay'])
    progress={'updates':0,'targets':0,'rows':[],'evaluations':{},'training_seconds':0.0}
    if path.exists():
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved['identity']!=identity:raise ValueError('checkpoint source/protocol/data identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);progress=saved['progress']
    if progress['updates']==c['updates_per_arm'] and (folder/'result.json').exists():
        previous=json.loads((folder/'result.json').read_text())
        assert previous['checkpoint_sha256']==file_sha(path) and previous['identity']==identity
        return previous
    def save():
        atomic_save({'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'progress':progress},path)
        write(folder/'progress.json',progress)
        if checkpoint_hook:checkpoint_hook(path)
    if '0' not in progress['evaluations']:progress['evaluations']['0']=evaluate(model,dev)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    x,y,_=train;batch=c['batch_episodes'];steps=c['updates_per_arm'];model.train()
    try:
        for step in range(progress['updates'],steps):
            if time.monotonic()>=deadline:raise TimeoutError('calibration training deadline; fixed budget not reduced')
            ids=torch.as_tensor(x[step*batch:(step+1)*batch],device=device);labels=torch.as_tensor(y[step*batch:(step+1)*batch],device=device)
            if device.type=='cuda':
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            logits,cache=model(ids);valid=labels!=-100;count=int(valid.sum());assert count==batch*8
            loss=F.cross_entropy(logits[valid],labels[valid],reduction='sum')/count
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite FP32 loss; no update')
            loss.backward()
            grads=[param.grad for param in model.parameters() if param.requires_grad]
            if any(g is None for g in grads) or not torch.stack([g.isfinite().all() for g in grads]).all():
                raise FloatingPointError('nonfinite/missing FP32 gradient; no clipping or update')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip'],error_if_nonfinite=True)
            before=model.base.backbone.embedding.weight.detach().clone() if step<2 else None
            optimizer.step()
            changed=float((model.base.backbone.embedding.weight.detach()-before).norm()) if before is not None else None
            if before is not None and not changed>0:raise FloatingPointError('optimizer failed to change parameters in initial two updates')
            if not torch.stack([param.isfinite().all() for param in model.parameters()]).all():raise FloatingPointError('nonfinite updated parameters; previous atomic checkpoint preserved')
            if device.type=='cuda':torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            progress['updates']=step+1;progress['targets']+=count;progress['training_seconds']+=seconds
            if device.type=='cuda':progress['peak_training_allocated_bytes']=max(progress.get('peak_training_allocated_bytes',0),torch.cuda.max_memory_allocated())
            progress['rows'].append({'update':step+1,'targets':progress['targets'],'loss':float(loss.detach()),'train_query_em':float((logits[valid].argmax(-1)==labels[valid]).float().mean()),'gradient_norm':float(norm),
                 'initial_parameter_change_norm':changed,'state_norm':float(torch.stack([s.ssm_state.detach().norm() for s in cache['mamba']]).norm()),'seconds':seconds})
            if step+1 in c['evaluation_updates']:progress['evaluations'][str(step+1)]=evaluate(model,dev)
            if (step+1)%c['checkpoint_every_updates']==0 or step+1==steps:
                save();print({'arm':folder.name,'updates':step+1,'targets':progress['targets'],'loss':progress['rows'][-1]['loss']},flush=True)
        assert progress['targets']==c['targets_per_arm']
    except TimeoutError:
        save();write(folder/'progress.json',progress);raise
    # No failed parameter state is saved when numerical checks raise.
    save();write(folder/'progress.json',progress)
    final=evaluate(model,dev,folder/'predictions.csv')
    control_x=dev[0].copy();control_x[(control_x>=72)&(control_x<136)]=7
    no_binding=evaluate(model,(control_x,dev[1],dev[2]))
    report={'status':'COMPLETE','identity':identity,'updates':progress['updates'],'targets':progress['targets'],
            'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':progress['training_seconds'],'query_targets_per_second':progress['targets']/max(progress['training_seconds'],1e-9),
            'processed_input_tokens':c['episodes_per_arm']*c['full_sequence_length'],
            'nonpadding_input_tokens':int((x!=0).sum()),'update_seconds_p50':float(np.percentile([r['seconds'] for r in progress['rows']],50)),
            'update_seconds_p95':float(np.percentile([r['seconds'] for r in progress['rows']],95)),
            'peak_training_allocated_bytes':progress.get('peak_training_allocated_bytes'),
            'no_binding_control':no_binding,'checkpoint_sha256':file_sha(path),'checkpoint_bytes':path.stat().st_size,'metrics':final}
    write(folder/'result.json',report);return report


def paired_analysis(folder,arms,samples=1000,seed=2710102):
    data={}
    for arm in arms:
        with (folder/arm/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
        episodes={}
        for row in rows:
            episodes.setdefault(row['episode'],[]).append((int(row['query']),int(row['target']),int(row['correct'])))
        data[arm]=episodes
    base=data[arms[0]];result={}
    for arm in arms[1:]:
        assert data[arm].keys()==base.keys(),'unpaired development episodes'
        deltas=[]
        for eid,rows in base.items():
            a=data[arm][eid];assert [(q,y) for q,y,_ in a]==[(q,y) for q,y,_ in rows],'unpaired query targets'
            deltas.append(np.mean([v for _,_,v in a])-np.mean([v for _,_,v in rows]))
        delta=np.array(deltas);rng=np.random.default_rng(seed)
        means=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(samples)]
        result[arm]={'paired_em_difference':float(delta.mean()),'episode_difference_sd':float(delta.std(ddof=1)),
             'development_episode_bootstrap_interval95':np.percentile(means,[2.5,97.5]).tolist(),
             'model_seeds':1,'scope':'episode sampling only, not training-seed variance or formal hypothesis test'}
    return result
