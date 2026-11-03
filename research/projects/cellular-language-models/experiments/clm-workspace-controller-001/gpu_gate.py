"""Bounded real-GPU evidence only; cannot pass on CPU or old Smoke metrics."""
import copy,json,time,statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from models import Model
from precision import select,autocast
from training import train
from evaluate import profile

TOLERANCE={'fp32':{'atol':2e-5,'rtol':2e-4,'mean':2e-6},'fp16':{'atol':.05,'rtol':.2,'mean':.005},'bf16':{'atol':.15,'rtol':.4,'mean':.02},'route_tie_gap':.002,'relative_denominator_floor':.1}

def error(a,b):
    diff=(a.float()-b.float()).abs()
    return {'maximum_absolute':float(diff.max()),'mean_absolute':float(diff.mean()),'maximum_relative':float((diff/a.float().abs().clamp_min(.1)).max()),'nonfinite':int((~torch.isfinite(a)).sum()+(~torch.isfinite(b)).sum())}

def accepted(e,precision):
    t=TOLERANCE[precision];return e['nonfinite']==0 and e['maximum_absolute']<=t['atol'] and e['mean_absolute']<=t['mean'] and e['maximum_relative']<=t['rtol']


def correctness(configs,p,deadline):
    if not torch.cuda.is_available():return {'status':'BLOCKED','reason':'target CUDA runtime unavailable'}
    device=torch.device('cuda');policy=select(p['precision'].get('requested','auto'),device);results={}
    for arm,entry in configs.items():
        model=Model(arm,entry['config'],17).to(device).eval();model.trace_tensors=True;cases=[]
        for length in (1,31,64,129,255,273):
            if time.monotonic()>=deadline:raise TimeoutError('GPU correctness deadline')
            ids=(torch.arange(length,device=device)[None]%100)+4
            def run(parts,chosen):
                cache=model.reset();outputs=[];steps=[[],[]]
                with torch.no_grad(),autocast(chosen,device):
                    for chunk in parts:
                        logits,cache=model(chunk,cache);outputs.append(logits)
                        for k,row in enumerate(model.tensor_trace.get('steps',[])):steps[k].append(row)
                return torch.cat(outputs,1),cache,steps
            full,cache,full_trace=run([ids],policy);full_encoder=model.tensor_trace.get('encoder_output');prefill,_,cached_trace=run([ids[:,:max(1,length//2)],ids[:,max(1,length//2):]] if length>1 else [ids],policy)
            sequential,_,single_trace=run(list(ids.split(1,1)),policy)
            fp32,_,_=run([ids],select('fp32',device));comparisons={'prefill_cached':error(full,prefill),'token_decode':error(full,sequential),'fp32_reference':error(fp32,full)}
            changed=ids.clone();cut=max(1,length//2);changed[:,cut:]=103
            causal,changed_cache,changed_trace=run([changed],policy);changed_encoder=model.tensor_trace.get('encoder_output');comparisons['causality']=error(full[:,:cut],causal[:,:cut])
            if full_encoder is not None:comparisons['encoder_causality']=error(full_encoder[:,:cut],changed_encoder[:,:cut])
            for step,rows in enumerate(full_trace):
                if rows:
                    comparisons['cell_causality_step'+str(step)]=error(rows[0]['delta'][:,:cut],changed_trace[step][0]['delta'][:,:cut])
                    if 'logits' in rows[0]:comparisons['router_causality_step'+str(step)]=error(rows[0]['logits'][:,:cut],changed_trace[step][0]['logits'][:,:cut])
            for name in ('encoder_kv','workspace_kv'):
                if cache.get(name) is not None:
                    take=max(0,cut-(length-cache[name][0].shape[2]))
                    if take:comparisons[name+'_causality']=error(cache[name][0][:,:,:take],changed_cache[name][0][:,:,:take])
            route_differences=[]
            for label,other in (('cached',cached_trace),('single',single_trace)):
                for k,records in enumerate(other):
                    if not records:continue
                    normal=full_trace[k][0]
                    if 'indices' not in normal:continue
                    selected=torch.cat([r['indices'] for r in records],1)
                    flips=(normal['indices'].sort(-1).values!=selected.sort(-1).values).any(-1)
                    sorted_logits=normal['logits'].float().sort(-1,descending=True).values;gap=sorted_logits[...,1]-sorted_logits[...,2]
                    route_differences.append({'execution':label,'step':k,'changed_paths':int(flips.sum()),'non_tie_changed_paths':int((flips&(gap>TOLERANCE['route_tie_gap'])).sum()),'minimum_selection_boundary_gap':float(gap.min())})
            ok=all(accepted(e,policy['selected_precision']) for e in comparisons.values()) and not any(r['non_tie_changed_paths'] for r in route_differences)
            cases.append({'length':length,'status':'PASS' if ok else 'FAIL','errors':comparisons,'route_differences':route_differences})
        results[arm]={'status':'PASS' if all(c['status']=='PASS' for c in cases) else 'FAIL','cases':cases}
        del model;torch.cuda.empty_cache()
    return {'status':'PASS' if all(r['status']=='PASS' for r in results.values()) else 'FAIL','policy':policy,'tolerances':TOLERANCE,'models':results}


def benchmark(configs,groups,p,out,deadline):
    reports={};device=torch.device('cuda')
    for arm in ('C2','T0-F'):
        reports[arm]={}
        for requested in ('fp32','fp16','bf16'):
            if time.monotonic()>=deadline:raise TimeoutError('precision benchmark deadline')
            q=copy.deepcopy(p);q['precision'].update(requested=requested,diagnostic_bf16=requested=='bf16')
            try:
                policy=select(requested,device,requested=='bf16');model=Model(arm,configs[arm]['config'],17).to(device)
                directory=out/(arm+'-'+requested);directory.mkdir(parents=True,exist_ok=True)
                # Independent short runs include cold first group; report steady samples separately.
                torch.cuda.synchronize();started=time.monotonic()
                progress,performance=train(model,groups[:5],q,deadline,directory/'checkpoint.pt','precision-diagnostic-'+arm+'-'+requested,resume=False)
                torch.cuda.synchronize();elapsed=time.monotonic()-started;rows=progress['rows'][1:]
                timings=[r['seconds'] for r in rows]
                reports[arm][requested]={'status':'PASS' if progress['status']=='COMPLETE' else 'FAIL','policy':policy,'first_loss':progress['rows'][0]['lm_loss'],'overflow_count':len(progress['overflows']),'effective_targets_per_second':progress['targets']/elapsed,'median_group_seconds':statistics.median(timings),'p95_group_seconds':sorted(timings)[-1],'forward_gpu_ms':[r['forward_gpu_ms'] for r in rows],'backward_gpu_ms':[r['backward_gpu_ms'] for r in rows],'optimizer_gpu_ms':[r['optimizer_gpu_ms'] for r in rows],'peak_vram':performance['peak_allocated_bytes'],'inference':profile(model,q,deadline,policy)}
            except Exception as failure:reports[arm][requested]={'status':'FAIL','error':type(failure).__name__+': '+str(failure)}
            finally:
                if 'model' in locals():del model
                torch.cuda.empty_cache()
    selected=select(p['precision'].get('requested','auto'),device)['selected_precision']
    ratios={arm:reports[arm].get(selected,{}).get('effective_targets_per_second',0)/max(1e-9,reports[arm].get('fp32',{}).get('effective_targets_per_second',0)) for arm in reports}
    return {'status':('PASS' if min(ratios.values())>=.8 else 'BLOCKED') if all(reports[a].get(selected,{}).get('status')=='PASS' for a in reports) else 'FAIL','target_vs_fp32_throughput_ratio':ratios,'performance_rule':'selected policy must achieve >=80% of FP32 throughput in both model families, otherwise request explicit precision decision; no silent fallback','hardware':select('auto',device),'models':reports,'native_claim':'capability-based policy; raw kernel names recorded, no native BF16 claim from framework compatibility'}


def controller_equivalence(config,p,deadline):
    from dependencies import BASE,frozen
    historical=frozen('gpu_smoke_reference',BASE/'reports/pre-main-gate/frozen_smoke_models.py');device=torch.device('cuda');policy=select(p['precision'].get('requested','auto'),device);results={}
    for arm in ('C2','C3'):
        if time.monotonic()>=deadline:raise TimeoutError('controller regression deadline')
        before=historical.Model(arm,config,17).to(device);after=Model(arm,config,17).to(device);after.load_state_dict(before.state_dict())
        ids=torch.arange(64,device=device)[None]%100+4;captures=[];calls=[];losses=[];outputs=[]
        for model in (before,after):
            route=model.route;records=[];updates=[]
            def wrapped(*args,**kwargs):
                value=route(*args,**kwargs);records.append((value[0].detach().clone(),value[1].detach().clone()));return value
            model.route=wrapped;hook=model.controller.register_forward_hook(lambda m,a,o:updates.append(o.detach().clone()))
            with autocast(policy,device):logits,_=model(ids);loss=F.cross_entropy(logits.reshape(-1,config['vocab']),ids.reshape(-1))
            (loss*1024).backward();captures.append(records);calls.append(updates);outputs.append(logits.detach());losses.append(float(loss.detach()));hook.remove()
        differences=[]
        for name,value in before.named_parameters():
            other=dict(after.named_parameters())[name]
            if value.grad is not None:differences.append(error(value.grad/1024,other.grad/1024))
        e=error(outputs[0],outputs[1]);same_routes=all(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) for a,b in zip(captures[0],captures[1]))
        good=accepted(e,policy['selected_precision']) and same_routes and len(calls[0])==2 and len(calls[1])==1 and all(accepted(d,policy['selected_precision']) for d in differences)
        results[arm]={'status':'PASS' if good else 'FAIL','logits':e,'routes_and_gates_equal':same_routes,'controller_call_counts':[len(calls[0]),len(calls[1])],'first_state':error(calls[0][0],calls[1][0]),'losses':losses,'max_parameter_gradient_error':max(d['maximum_absolute'] for d in differences)}
        del before,after;torch.cuda.empty_cache()
    return {'status':'PASS' if all(v['status']=='PASS' for v in results.values()) else 'FAIL','policy':policy,'models':results}
