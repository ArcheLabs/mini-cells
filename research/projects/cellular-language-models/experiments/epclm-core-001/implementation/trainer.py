"""Resumable document TBPTT with explicit per-lane cursors and state."""
import hashlib
import io
import json
import os
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from native_clm_runtime import lr_multiplier,sha256_file
from model import check_numerics

def atomic_save(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(value,stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)

def save_checkpoint(value, path):
    payload = io.BytesIO()
    torch.save(value, payload)
    content = payload.getvalue()
    # The payload and its checksum become visible in one atomic file replacement.
    atomic_save({'format': 'core001.checkpoint.v2', 'payload': content,
                 'sha256': hashlib.sha256(content).hexdigest()}, path)


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') == 'core001.checkpoint.v2':
        content = saved['payload']
        if hashlib.sha256(content).hexdigest() != saved['sha256']:
            raise ValueError('checkpoint hash mismatch')
        return torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    # Existing v0.1 checkpoints remain readable with their original sidecar.
    sidecar = Path(str(path) + '.sha256')
    if not sidecar.exists() or sidecar.read_text().strip() != sha256_file(Path(path)):
        raise ValueError('legacy checkpoint hash mismatch')
    return saved

def rng(include_cuda=True):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if include_cuda else []}

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

class NumericalFailure(FloatingPointError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def metric(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def gradient_metrics(model):
    gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise FloatingPointError('no model gradients')
    finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
    norm = torch.stack([g.float().norm() for g in gradients]).norm()
    return finite, float(norm)


def train(model,documents,identity,path,target_tokens,deadline,seed,batch=4,unroll=64,
          stop_after_updates=None,on_checkpoint=None,checkpoint_interval=10,
          precision='fp16',initial_scale=1024.,max_consecutive_overflows=3):
    if checkpoint_interval < 1 or max_consecutive_overflows < 1 or not math.isfinite(initial_scale) or initial_scale <= 0 or precision not in ('fp16','fp32'):
        raise ValueError('invalid checkpoint/precision settings')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    total_steps=max(1,math.ceil(target_tokens/(batch*unroll)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(step,total_steps,min(100,max(1,total_steps//10))))
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16' and next(model.parameters()).is_cuda,init_scale=initial_scale)
    amp_enabled=scaler.is_enabled()
    order=list(range(len(documents))); random.shuffle(order)
    cursor=0; lanes=[None]*batch; positions=[0]*batch; state=model.init_state(batch)
    tokens=updates=0; curve=[]; diagnostics=[]; consecutive_overflows=0; fp32_probes={}
    restore_started=time.monotonic()
    if Path(path).exists():
        saved=load_checkpoint(path)
        if saved['identity']!=identity or saved.get('precision',precision)!=precision:
            raise ValueError('checkpoint source/config/data/protocol/precision mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        order,cursor,lanes,positions=saved['order'],saved['cursor'],saved['lanes'],saved['positions']
        state=saved['state'].to(model.embedding.weight); tokens,updates=saved['tokens'],saved['updates']
        curve=saved['curve']; diagnostics=saved.get('overflow_diagnostics',[])
        consecutive_overflows=saved.get('consecutive_overflows',0); restore_rng(saved['rng'])
    last_saved_updates=updates if Path(path).exists() else -1
    initial_updates=updates
    restore_seconds=time.monotonic()-restore_started
    checkpoint_seconds=0.
    started=time.monotonic(); initial_tokens=tokens

    def save():
        nonlocal checkpoint_seconds,last_saved_updates
        checkpoint_started=time.monotonic()
        save_checkpoint({'identity':identity,'precision':precision,'model':model.state_dict(),
                         'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                         'scaler':scaler.state_dict(),'rng':rng(state.is_cuda),'order':order,
                         'cursor':cursor,'lanes':lanes,'positions':positions,'state':state.detach().cpu(),
                         'tokens':tokens,'updates':updates,'curve':curve,
                         'overflow_diagnostics':diagnostics,'consecutive_overflows':consecutive_overflows},path)
        last_saved_updates=updates
        checkpoint_seconds+=time.monotonic()-checkpoint_started
        if on_checkpoint is not None: on_checkpoint()

    def record(event):
        diagnostics.append(event)
        destination=Path(path).with_suffix('.diagnostics.json')
        temporary=destination.with_suffix('.tmp')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with temporary.open('w') as stream:
            json.dump(diagnostics,stream,indent=2,allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(destination)
        print(json.dumps(event,allow_nan=False),flush=True)

    def rollout(use_amp):
        nonlocal state,order,cursor,lanes,positions
        state=state.detach(); losses=[]; actual=0; signature=hashlib.sha256()
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
            signature.update(repr((inputs,targets,resets,active)).encode())
            mask=torch.tensor(active,device=state.device)
            state=model.reset(state,torch.tensor(resets,device=state.device))
            with torch.autocast(device_type=state.device.type,dtype=torch.float16,enabled=use_amp):
                next_state,logits=model.step(state,inputs)
                per_token=F.cross_entropy(logits,torch.tensor(targets,device=state.device),reduction='none')
            check_numerics(torch.isfinite(per_token).all(), 'non-finite loss')
            state=torch.where(mask[:,None,None],next_state,state)
            losses.append((per_token*mask).sum()); actual+=sum(active)
            for lane,enabled in enumerate(active): positions[lane]+=int(enabled)
        return sum(losses)/actual,actual,signature.hexdigest()

    model.train()
    if consecutive_overflows>=max_consecutive_overflows:
        raise NumericalFailure('saved checkpoint exhausted consecutive overflow allowance',diagnostics)
    while tokens<target_tokens and time.monotonic()<deadline-10 and (stop_after_updates is None or updates<stop_after_updates):
        # A rejected update is a transaction: replay the same state, documents and RNG.
        previous=(state.detach(),order.copy(),cursor,lanes.copy(),positions.copy(),rng(state.is_cuda))
        def rollback():
            nonlocal state,order,cursor,lanes,positions
            state,old_order,cursor,old_lanes,old_positions,random_state=previous
            order=old_order.copy(); lanes=old_lanes.copy(); positions=old_positions.copy()
            restore_rng(random_state); optimizer.zero_grad(set_to_none=True)

        optimizer.zero_grad(set_to_none=True)
        loss,actual,signature=rollout(amp_enabled)
        scale=scaler.get_scale()
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        finite,norm=gradient_metrics(model)
        if not finite or not math.isfinite(norm):
            event={'stage':'GRADIENT_OVERFLOW','update':updates+1,'committed_updates':updates,
                   'attempt_tokens':actual,'committed_tokens':tokens,'loss':metric(loss.detach()),
                   'state_norm':metric(state.detach().norm()),'gradient_norm':metric(norm),
                   'grad_scaler_scale':scale,'input_sha256':signature,'classification':'FP32_NONFINITE'}
            if amp_enabled:
                # unscale_ registered actual Inf/NaN gradients. GradScaler.step then
                # skips optimizer.step, and update performs its normal backoff.
                if not finite:
                    scaler.step(optimizer); scaler.update()
                else:
                    # Finite elements can still overflow an FP32 norm reduction.
                    # Explicitly skip rather than letting a zero found_inf permit a step.
                    scaler.update(new_scale=scale*scaler.get_backoff_factor())
                event['next_scale']=scaler.get_scale()
                rollback()
                cached=fp32_probes.get(updates)
                if cached is not None:
                    if cached['input_sha256']!=signature: raise ValueError('FP32 diagnostic input changed across retry')
                    event['fp32']=dict(cached); event['reused_fp32_diagnostic']=True
                    probe_ok=cached['finite']
                    event['classification']='MIXED_PRECISION_PATH'
                else:
                    try:
                        probe_loss,probe_tokens,probe_signature=rollout(False)
                        probe_loss.backward()
                        probe_finite,probe_norm=gradient_metrics(model)
                        probe_ok=probe_finite and math.isfinite(probe_norm)
                        if probe_tokens!=actual or probe_signature!=signature:
                            raise ValueError('FP32 diagnostic did not replay the same update')
                        event['fp32']={'loss':metric(probe_loss.detach()),'state_norm':metric(state.detach().norm()),
                                       'gradient_norm':metric(probe_norm),'tokens':probe_tokens,
                                       'input_sha256':probe_signature,'finite':probe_ok,'optimizer_updates':0}
                        event['classification']='MIXED_PRECISION_PATH' if probe_ok else 'MODEL_OR_TBPTT_INSTABILITY'
                    except FloatingPointError as error:
                        probe_ok=False
                        event['fp32']={'finite':False,'error':str(error),'optimizer_updates':0}
                        event['classification']='MODEL_OR_TBPTT_INSTABILITY'
                    if probe_ok: fp32_probes[updates]=dict(event['fp32'])
                rollback()
            else:
                probe_ok=False; rollback()
            consecutive_overflows+=1
            event['consecutive_overflows']=consecutive_overflows
            record(event); save()
            if not amp_enabled or not probe_ok or consecutive_overflows>=max_consecutive_overflows:
                raise NumericalFailure('non-finite gradients: '+event['classification'],diagnostics)
            continue
        # Never clip non-finite gradients or disable finite-error detection.
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
            record({'stage':'NONFINITE_PARAMETERS','update':updates+1,'committed_tokens':tokens,
                    'loss':metric(loss.detach()),'gradient_norm':metric(norm),'grad_scaler_scale':scale})
            raise NumericalFailure('non-finite parameters after optimizer step; checkpoint not advanced',diagnostics)
        scheduler.step(); tokens+=actual; updates+=1; consecutive_overflows=0; fp32_probes.clear()
        curve.append({'tokens':tokens,'nll':float(loss.detach()),'gradient_norm':norm,
                      'state_norm':float(state.detach().norm()),'grad_scaler_scale':scaler.get_scale()})
        if updates%checkpoint_interval==0: save()
    if last_saved_updates!=updates: save()
    return {'status':'COMPLETE' if tokens>=target_tokens else 'INCOMPLETE','tokens':tokens,'updates':updates,
            'resumed_from_tokens':initial_tokens,'resumed_from_updates':initial_updates,
            'target_tokens':target_tokens,'new_tokens':tokens-initial_tokens,'elapsed_seconds':time.monotonic()-started,
            'checkpoint_save_seconds':checkpoint_seconds,'checkpoint_restore_seconds':restore_seconds,
            'tokens_per_second':(tokens-initial_tokens)/max(time.monotonic()-started,1e-9),'curve':curve,
            'overflow_diagnostics':diagnostics,'peak_vram_bytes':torch.cuda.max_memory_allocated() if state.is_cuda else None}


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
