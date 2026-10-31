"""Resumable document TBPTT with explicit per-lane cursors and state."""
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from native_clm_runtime import lr_multiplier,sha256_file

def atomic_save(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp'); torch.save(value,temporary); temporary.replace(path)

def rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda']: torch.cuda.set_rng_state_all(value['cuda'])

@torch.no_grad()
def evaluate(model,documents,steps=None,ablation=None,deadline=float("inf")):
    if not documents: raise ValueError("empty evaluation")
    model.eval(); rows=[]
    for document in documents:
        if time.monotonic()>deadline: raise TimeoutError('validation budget exhausted')
        state=model.init_state(1); total=0.; count=0
        if ablation=='swap':
            other=documents[(documents.index(document)+1)%len(documents)]
            for prior in other['tokens'][:-1]: state,_=model.step(state,[prior])
        for token,target in zip(document['tokens'][:-1],document['tokens'][1:]):
            if time.monotonic()>deadline: raise TimeoutError('validation budget exhausted')
            if ablation=='reset': state=model.reset(state)
            if ablation=='entry': state=torch.cat((state[:,:1],torch.zeros_like(state[:,1:])),1)
            state,logits=model.step(state,[token],steps,communication=ablation!='communication')
            loss=F.cross_entropy(logits,torch.tensor([target],device=logits.device))
            if not torch.isfinite(loss): raise FloatingPointError('validation NLL is non-finite')
            total+=float(loss); count+=1
        rows.append({'story_id':document['id'],'nll_sum':total,'tokens':count,'nll':total/count})
    nll=sum(r['nll_sum'] for r in rows)/sum(r['tokens'] for r in rows)
    return {'nll':nll,'ppl':math.exp(min(nll,700)),'stories':rows}

def train(model,documents,identity,path,target_tokens,deadline,seed,batch=4,unroll=64,stop_after_updates=None,on_checkpoint=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    total_steps=max(1,math.ceil(target_tokens/(batch*unroll)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(step,total_steps,min(100,max(1,total_steps//10))))
    scaler=torch.amp.GradScaler('cuda',enabled=next(model.parameters()).is_cuda)
    order=list(range(len(documents))); random.shuffle(order)
    cursor=0; lanes=[None]*batch; positions=[0]*batch; state=model.init_state(batch)
    tokens=updates=0; curve=[]
    restore_started=time.monotonic()
    if Path(path).exists():
        sidecar=Path(str(path)+'.sha256')
        if not sidecar.exists() or sidecar.read_text().strip()!=sha256_file(Path(path)): raise ValueError('checkpoint hash mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity: raise ValueError('checkpoint source/config/data/protocol mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        order,cursor,lanes,positions=saved['order'],saved['cursor'],saved['lanes'],saved['positions']
        state=saved['state'].to(model.embedding.weight); tokens,updates=saved['tokens'],saved['updates']
        curve=saved['curve']; restore_rng(saved['rng'])
    restore_seconds=time.monotonic()-restore_started
    checkpoint_seconds=0.
    started=time.monotonic(); initial_tokens=tokens
    def save():
        nonlocal checkpoint_seconds
        checkpoint_started=time.monotonic()
        atomic_save({'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng(),
                     'order':order,'cursor':cursor,'lanes':lanes,'positions':positions,'state':state.detach().cpu(),
                     'tokens':tokens,'updates':updates,'curve':curve},path)
        Path(str(path)+'.sha256').write_text(sha256_file(Path(path))+'\n')
        checkpoint_seconds+=time.monotonic()-checkpoint_started
        if on_checkpoint is not None: on_checkpoint()
    model.train()
    while tokens<target_tokens and time.monotonic()<deadline-10 and (stop_after_updates is None or updates<stop_after_updates):
        optimizer.zero_grad(set_to_none=True); state=state.detach(); losses=[]; actual=0
        for _ in range(unroll):
            if tokens+actual>=target_tokens: break
            active=[]; inputs=[]; targets=[]; resets=[]
            for lane in range(batch):
                reset=lanes[lane] is None or positions[lane]>=len(documents[lanes[lane]]['tokens'])-1
                if reset:
                    if cursor==len(order): random.shuffle(order); cursor=0
                    lanes[lane]=order[cursor]; cursor+=1; positions[lane]=0
                doc=documents[lanes[lane]]['tokens']; p=positions[lane]
                inputs.append(doc[p]); targets.append(doc[p+1]); resets.append(reset)
                active.append(tokens+actual+lane<target_tokens)
            mask=torch.tensor(active,device=state.device)
            state=model.reset(state,torch.tensor(resets,device=state.device))
            with torch.autocast(device_type=state.device.type,dtype=torch.float16,enabled=state.is_cuda):
                next_state,logits=model.step(state,inputs)
                per_token=F.cross_entropy(logits,torch.tensor(targets,device=state.device),reduction='none')
            if not torch.isfinite(per_token).all(): raise FloatingPointError('non-finite loss')
            state=torch.where(mask[:,None,None],next_state,state)
            losses.append(per_token[mask].sum()); actual+=int(mask.sum())
            for lane,enabled in enumerate(active): positions[lane]+=int(enabled)
        loss=sum(losses)/actual
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        tokens+=actual; updates+=1
        curve.append({'tokens':tokens,'nll':float(loss.detach()),'gradient_norm':float(norm),'state_norm':float(state.detach().norm())})
        if updates%10==0: save()
    save()
    return {'status':'COMPLETE' if tokens>=target_tokens else 'INCOMPLETE','tokens':tokens,'target_tokens':target_tokens,
            'checkpoint_save_seconds':checkpoint_seconds,'checkpoint_restore_seconds':restore_seconds,
            'tokens_per_second':(tokens-initial_tokens)/max(time.monotonic()-started,1e-9),'curve':curve,
            'peak_vram_bytes':torch.cuda.max_memory_allocated() if state.is_cuda else None}

def paired(left,right,seed=713,draws=1000):
    a={r['story_id']:r for r in left['stories']}; b={r['story_id']:r for r in right['stories']}
    if a.keys()!=b.keys(): raise ValueError('unpaired documents')
    ids=sorted(a); generator=np.random.default_rng(seed); estimates=[]
    def difference(sample):
        return sum(a[k]['nll_sum']-b[k]['nll_sum'] for k in sample)/sum(a[k]['tokens'] for k in sample)
    for _ in range(draws): estimates.append(difference(generator.choice(ids,len(ids))))
    delta=difference(ids)
    return {'nll_difference':delta,'relative_change':delta/right['nll'],'story_bootstrap_95':np.quantile(estimates,[.025,.975]).tolist(),
            'training_seeds':1,'between_seed_variance':'NOT_EVALUATED','scientific_decision':False}
