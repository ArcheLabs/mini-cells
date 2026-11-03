"""Resumable complete-document groups using existing atomic checkpoint writer."""
import copy,math,random,time
import numpy as np
import torch
from torch.nn import functional as F
from dependencies import shared,detach
from precision import select

def targets(group):return sum(len(d['ids'])-1 for d in group)

def identity(config,arm,source,data):return shared.sha({'config':config,'arm':arm,'source':source,'data':data})

def save(path,model,opt,scaler,progress,identity,policy):
    shared.native._atomic_save({'format':'minicells.clm-v04.checkpoint.v1','identity':identity,'precision_policy':policy,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'scheduler':{'kind':'committed-target cosine','committed_targets':progress['targets']},'progress':progress,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},path)

def train(model,groups,p,deadline,checkpoint,run_identity,resume=True,checkpoint_hook=None):
    device=model.token.weight.device;opt=torch.optim.AdamW(model.parameters(),lr=p['optimizer']['lr'],betas=tuple(p['optimizer']['betas']),weight_decay=p['optimizer']['weight_decay'])
    policy=select(p['precision'].get('requested','auto'),device,p['precision'].get('diagnostic_bf16',False));bf16=policy['selected_precision']=='bf16';amp=policy['selected_precision']!='fp32';dtype=torch.bfloat16 if bf16 else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=policy['scaler_enabled'],init_scale=p['precision']['initial_scale'])
    progress={'targets':0,'updates':0,'next_group':0,'rows':[],'overflows':[],'status':'INCOMPLETE'}
    model.generator=torch.Generator(device=device).manual_seed(p['smoke']['seed']+99)
    if resume and checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved.get('precision_policy')!=policy:raise ValueError('checkpoint precision policy differs; implicit mixed-protocol continuation forbidden')
        if saved['identity']!=run_identity:raise ValueError('incompatible source/config/data checkpoint; restart not allowed silently')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);progress=saved['progress']
        torch.set_rng_state(saved['cpu_rng'].cpu());random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        if saved['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        if 'router_rng' in progress:model.generator.set_state(progress['router_rng'].cpu())
    total=p.get('schedule_target_budget',sum(targets(g) for g in groups));initial_parameters=[v.detach().clone() for cell in getattr(model,'cells',[]) for v in cell.parameters()]
    initial_targets=progress['targets']
    began=time.monotonic();model.train();model.noise=p['router']['noise']
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        for i in range(progress['next_group'],len(groups)):
            if progress['targets']>=p.get('session_stop_targets',float('inf')):break
            if time.monotonic()>=deadline:break
            group=groups[i];n=targets(group)
            if progress['targets']+n>total:raise ValueError('group crosses frozen scheduler target budget')
            frac=(progress['targets']+n)/total;warm=p['optimizer']['warmup_fraction']
            lr=p['optimizer']['lr']*(frac/warm if frac<warm else .5*(1+math.cos(math.pi*(frac-warm)/(1-warm))))
            for entry in opt.param_groups:entry['lr']=lr
            rng=model.generator.get_state().clone();cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None
            started=time.monotonic()
            for attempt in range(p['precision']['max_consecutive_overflows']):
                opt.zero_grad(set_to_none=True);model.generator.set_state(rng);torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
                lm_sum=aux_sum=0.;forward_events=[];backward_events=[]
                for document in group:
                    cache=model.reset(document['doc_id']);ids=document['ids']
                    for start in range(0,len(ids)-1,p['model']['chunk']):
                        if time.monotonic()>=deadline:raise TimeoutError('training phase deadline; current uncommitted group discarded')
                        end=min(start+p['model']['chunk'],len(ids)-1);x=torch.tensor([ids[start:end]],device=device);y=torch.tensor(ids[start+1:end+1],device=device)
                        fe=None
                        if device.type=='cuda':fe=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));fe[0].record()
                        with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                            logits,cache=model(x,cache);lm=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1));aux=model.balance;loss=lm+p['router']['balance_alpha']*aux
                        if fe is not None:fe[1].record();forward_events.append(fe)
                        if not bool(loss.isfinite()):raise FloatingPointError('nonfinite loss')
                        be=None
                        if device.type=='cuda':be=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));be[0].record()
                        scaler.scale(loss*((end-start)/n)).backward();cache=detach(cache)
                        if be is not None:be[1].record();backward_events.append(be)
                        lm_sum+=float(lm.detach())*(end-start);aux_sum+=float(aux.detach())*(end-start)
                scaler.unscale_(opt)
                finite=bool(torch.stack([v.grad.isfinite().all() for v in model.parameters() if v.grad is not None]).all())
                if finite:break
                progress['overflows'].append({'group':i,'attempt':attempt,'scale':scaler.get_scale(),'targets_committed':progress['targets']})
                if not scaler.is_enabled():raise FloatingPointError('nonfinite non-FP16 gradients; no optimizer step')
                scaler.step(opt);scaler.update()
                if attempt+1==p['precision']['max_consecutive_overflows']:raise FloatingPointError('consecutive FP16 overflow tolerance exceeded')
            norms={name:float(torch.stack([v.grad.detach().float().norm() for v in module.parameters() if v.grad is not None]).norm()) for name,module in model.named_children() if any(v.grad is not None for v in module.parameters())}
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['optimizer']['clip'],error_if_nonfinite=True)
            optimizer_events=None
            if device.type=='cuda':optimizer_events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));optimizer_events[0].record()
            scaler.step(opt);scaler.update()
            if optimizer_events is not None:optimizer_events[1].record()
            if not all(bool(v.isfinite().all()) for v in model.parameters()):raise FloatingPointError('nonfinite parameter after step; checkpoint not advanced')
            progress['targets']+=n;progress['updates']+=1;progress['next_group']=i+1
            progress['rows'].append({'targets':progress['targets'],'lm_loss':lm_sum/n,'balance_loss':aux_sum/n,'total_loss':lm_sum/n+p['router']['balance_alpha']*aux_sum/n,'gradient_norm':float(norm),'module_gradients':norms,'forward_gpu_ms':sum(a.elapsed_time(b) for a,b in forward_events) if device.type=='cuda' else None,'backward_gpu_ms':sum(a.elapsed_time(b) for a,b in backward_events) if device.type=='cuda' else None,'optimizer_gpu_ms':optimizer_events[0].elapsed_time(optimizer_events[1]) if optimizer_events is not None else None,'lr':lr,'seconds':time.monotonic()-started})
            progress['router_rng']=model.generator.get_state();save(checkpoint,model,opt,scaler,progress,run_identity,policy)
            if checkpoint_hook is not None:checkpoint_hook(checkpoint)
        progress['status']='COMPLETE' if progress['next_group']==len(groups) else 'INCOMPLETE'
    except TimeoutError:progress['status']='INCOMPLETE'
    finally:
        progress['router_rng']=model.generator.get_state()
        # Never serialize failed/uncommitted parameter states over a good checkpoint.
        model.noise=0.;model.eval()
    elapsed=time.monotonic()-began
    current=[v.detach() for cell in getattr(model,'cells',[]) for v in cell.parameters()]
    changes=[float((a-b).float().norm()) for a,b in zip(current,initial_parameters)]
    return {k:v for k,v in progress.items() if k!='router_rng'}, {'precision':policy['selected_precision'],'precision_policy':policy,'session_seconds':elapsed,'session_targets':progress['targets']-initial_targets,'session_targets_per_second':(progress['targets']-initial_targets)/max(elapsed,1e-9),'expert_parameter_tensor_update_norms':changes,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
