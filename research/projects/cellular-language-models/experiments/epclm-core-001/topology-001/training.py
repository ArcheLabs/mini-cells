"""Fresh-only version of existing document TBPTT; same ordering, reset, loss and LR rules."""
import math
import random
import time
import torch
from torch.nn import functional as F
from native_clm_runtime import lr_multiplier


def sync(model):
    if next(model.parameters()).is_cuda:torch.cuda.synchronize()


def observation(model):
    return [{k:v.detach().cpu().tolist() for k,v in row.items()} for row in model.observation]


@torch.no_grad()
def evaluate(model,documents,deadline):
    if not documents:raise ValueError('empty evaluation split')
    model.eval();rows=[];started=time.monotonic()
    for doc in documents:
        state=model.init_state(1);losses=[]
        for token,target in zip(doc['tokens'][:-1],doc['tokens'][1:]):
            if time.monotonic()>=deadline:raise TimeoutError('evaluation budget exhausted')
            state,logits=model.step(state,[token])
            losses.append(F.cross_entropy(logits,torch.tensor([target],device=state.device),reduction='sum'))
        total=float(torch.stack(losses).sum());count=len(losses)
        if not math.isfinite(total):raise FloatingPointError('nonfinite evaluation NLL')
        rows.append({'document_id':doc['id'],'prediction_tokens':count,'total_nll':total,'mean_nll':total/count})
    total=sum(r['total_nll'] for r in rows);count=sum(r['prediction_tokens'] for r in rows);mean=total/count
    return {'status':'COMPLETE','documents':rows,'total_documents':len(rows),'prediction_tokens':count,
            'total_nll':total,'mean_nll':mean,'ppl':math.exp(mean) if mean<700 else None,'evaluation_seconds':time.monotonic()-started}


def train(model,documents,validation,protocol,deadline,entry,persist):
    plan=protocol['training'];seed=protocol['seed'];rng=random.Random(seed)
    batch=plan['batch'];unroll=plan['tbptt'];target=plan['prediction_tokens_per_arm']
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],betas=tuple(plan['betas']),weight_decay=plan['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(step,plan['updates'],plan['warmup_updates']))
    order=list(range(len(documents)));rng.shuffle(order);cursor=0;lanes=[None]*batch;positions=[0]*batch
    state=model.init_state(batch);tokens=updates=0;curve=[];progression={};started=time.monotonic();compute_seconds=0.
    assert not optimizer.state and tokens==updates==0
    entry.update(training_status='RUNNING',fresh_initialization=True,training_metrics=curve,validation_progression=progression)
    if state.is_cuda:torch.cuda.reset_peak_memory_stats()
    try:
        while tokens<target:
            if time.monotonic()>=deadline:raise TimeoutError('training budget exhausted; incomplete arm must restart from zero')
            model.train();sync(model);begin=time.monotonic();state=state.detach();optimizer.zero_grad(set_to_none=True)
            losses=[];actual=0;learning_rate=optimizer.param_groups[0]['lr']
            for tick in range(unroll):
                if tokens+actual>=target:break
                inputs=[];targets=[];resets=[]
                for lane in range(batch):
                    reset=lanes[lane] is None or positions[lane]>=len(documents[lanes[lane]]['tokens'])-1
                    if reset:
                        if cursor==len(order):rng.shuffle(order);cursor=0
                        lanes[lane]=order[cursor];cursor+=1;positions[lane]=0
                    doc=documents[lanes[lane]]['tokens'];pos=positions[lane]
                    inputs.append(doc[pos]);targets.append(doc[pos+1]);resets.append(reset)
                active=[tokens+actual+lane<target for lane in range(batch)]
                mask=torch.tensor(active,device=state.device);state=model.reset(state,torch.tensor(resets,device=state.device))
                model.observe=(tick==unroll-1 or tokens+actual+sum(active)>=target)
                next_state,logits=model.step(state,inputs)
                losses.append((F.cross_entropy(logits,torch.tensor(targets,device=state.device),reduction='none')*mask).sum())
                state=torch.where(mask[:,None,None],next_state,state);actual+=sum(active)
                for lane,enabled in enumerate(active):positions[lane]+=int(enabled)
            model.observe=False
            loss=sum(losses)/actual
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite FP32 loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip'],error_if_nonfinite=True)
            optimizer.step()
            if not bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all()):raise FloatingPointError('nonfinite parameters')
            scheduler.step();tokens+=actual;updates+=1;sync(model);duration=time.monotonic()-begin;compute_seconds+=duration
            row={'update':updates,'prediction_tokens':tokens,'update_prediction_tokens':actual,'loss':float(loss.detach()),
                 'learning_rate':learning_rate,'gradient_norm':float(norm),'update_seconds':duration,'tokens_per_second':actual/max(duration,1e-12),
                 'training_compute_seconds':compute_seconds,'state_observations':observation(model)}
            curve.append(row);entry.update(prediction_tokens=tokens,optimizer_updates=updates)
            if updates%25==0:print({'arm':model.topology,'update':updates,'tokens':tokens,'loss':row['loss']},flush=True);persist()
            if updates in plan['validation_milestones']:
                try:progression[str(updates)]=evaluate(model,validation,deadline)
                except Exception as error:progression[str(updates)]={'status':'FAILED','error':str(error)}
            model.observation=[]
        if tokens!=target or updates!=plan['updates']:raise ValueError('exact prediction token/update budget not met')
        entry['training_status']='COMPLETE'
    finally:
        model.observe=False
        entry.update(prediction_tokens=tokens,optimizer_updates=updates,training_compute_seconds=compute_seconds,
                     training_wall_seconds=time.monotonic()-started,tokens_per_second=tokens/max(compute_seconds,1e-12),
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if state.is_cuda else None,
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if state.is_cuda else None)
        persist()
