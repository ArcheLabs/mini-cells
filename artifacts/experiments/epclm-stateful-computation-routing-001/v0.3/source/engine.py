"""Complete-document optimizer groups, chunk-local credit and deterministic retries."""
import copy
import math
import time
import torch
from torch.nn import functional as F
from dependencies import shared
from models import detach
from data import count


def check_time(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('phase deadline reached')


def group_update(model,documents,optimizer,config,baseline,generator,deadline):
    old_rng=generator.get_state().clone();total=count(documents);device=next(model.parameters()).device
    baseline=[0.,0.] if isinstance(baseline,(float,int)) else baseline
    optimizer.zero_grad(set_to_none=True);model.train();model.sample=True
    ce_sum=0.;returns_sum=[0.,0.];action_count=0;forward_seconds=backward_seconds=0.;eligible=0;writes=0;gpu_forward=[];gpu_backward=[]
    try:
        for document in documents:
            cache=model.reset(document['doc_id']);ids=document['ids'];chunks=[]
            for start in range(0,len(ids)-1,config['model']['chunk']):
                check_time(deadline);end=min(start+config['model']['chunk'],len(ids)-1)
                began=time.monotonic()
                x=torch.tensor([ids[start:end]],device=device);y=torch.tensor(ids[start+1:end+1],device=device)
                events=None
                if device.type=='cuda':
                    events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
                logits,cache,actions=model(x,cache,generator)
                if events is not None:events[1].record();gpu_forward.append(events)
                losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y,reduction='none')
                chunks.append((losses,actions));ce_sum+=float(losses.detach().sum());forward_seconds+=time.monotonic()-began
                writes+=int(model.arm in ('M1','M2'))
                if len(chunks)%2==0:cache=detach(cache)
            for i in range(0,len(chunks),2):
                objective=0
                for j in range(i,min(i+2,len(chunks))):
                    losses,actions=chunks[j];objective=objective+losses.sum()/total
                    if model.arm!='T0':
                        future=chunks[j+1][0].detach().mean() if j+1<len(chunks) else 0.
                        returns=losses.detach()+future
                        old=losses.new_tensor(baseline)[None,None,:]
                        objective=objective+config['router']['pg_weight']*((returns[None,:,None]-old)*actions).sum()/(2*total)
                        for k in range(2):returns_sum[k]+=float(returns.sum())
                        action_count+=len(losses)
                    if model.arm in ('M1','M2') and j%2==0 and j+1<len(chunks):eligible+=1
                if not bool(objective.isfinite()):raise FloatingPointError('nonfinite objective')
                began=time.monotonic();events=None
                if device.type=='cuda':events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
                objective.backward()
                if events is not None:events[1].record();gpu_backward.append(events)
                backward_seconds+=time.monotonic()-began
        gradients={name:float(torch.stack([p.grad.detach().norm() for p in module.parameters() if p.grad is not None]).norm()) for name,module in model.named_children() if any(p.grad is not None for p in module.parameters())}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['clip'],error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(p.isfinite().all()) for p in model.parameters()):raise FloatingPointError('nonfinite parameters after update; invalid checkpoint')
        beta=config['router']['baseline_beta'];new_baseline=[beta*baseline[k]+(1-beta)*returns_sum[k]/max(1,action_count) for k in range(2)]
        return {'targets':total,'mean_nll':ce_sum/total,'gradient_norm':float(norm),'baseline':new_baseline,'module_gradient_norms':gradients,'ce_head_gradient_norm':gradients.get('head'),'router_policy_query_gradient_norm':gradients.get('query'),'gradient_scope':'head has CE-only gradients, Router query has score-function gradients; other modules may combine both','forward_host_seconds':forward_seconds,'backward_host_seconds':backward_seconds,'forward_gpu_ms':sum(a.elapsed_time(b) for a,b in gpu_forward) if gpu_forward else None,'backward_gpu_ms':sum(a.elapsed_time(b) for a,b in gpu_backward) if gpu_backward else None,'state_writes':writes,'writes_with_next_chunk_gradient':eligible,'credit_assignment':'TRUNCATED_ROUTER_CREDIT_ASSIGNMENT'}
    except Exception:
        optimizer.zero_grad(set_to_none=True);generator.set_state(old_rng);raise


def optimizer(model,config):
    p=config['optimizer'];return torch.optim.AdamW(model.parameters(),lr=p['lr'],betas=tuple(p['betas']),weight_decay=p['weight_decay'])


def train(model,document_groups,config,deadline,out):
    opt=optimizer(model,config);device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(config['router_seed']);baseline=[0.,0.];tokens=0;rows=[]
    total=sum(count(g) for g in document_groups);began=time.monotonic();status='INCOMPLETE'
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        for group in document_groups:
            p=config['optimizer'];progress=(tokens+count(group))/total;warm=p['warmup_fraction']
            multiplier=min(1.,(tokens+count(group))/(total*warm)) if progress<warm else .5*(1+math.cos(math.pi*(progress-warm)/(1-warm)))
            for item in opt.param_groups:item['lr']=p['lr']*multiplier
            start=time.monotonic();row=group_update(model,group,opt,config,baseline,generator,deadline)
            baseline=row['baseline'];tokens+=row['targets'];rows.append({**row,'committed_targets':tokens,'seconds':time.monotonic()-start,'learning_rate':opt.param_groups[0]['lr'],'doc_ids':[d['doc_id'] for d in group]})
            shared.write(out/'training-metrics.json',{'status':'IN_PROGRESS','rows':rows,'targets':tokens})
        status='COMPLETE'
    finally:
        report={'status':status,'targets':tokens,'updates':len(rows),'seconds':time.monotonic()-began,'rows':rows,'tokens_per_second':tokens/max(time.monotonic()-began,1e-9),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
        shared.write(out/'training-metrics.json',report)
    return report


@torch.no_grad()
def evaluate(model,documents,config,deadline,mode='normal',sample=False):
    model.eval();model.mode=mode;model.sample=sample;device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(config['evaluation']['sample_seed'])
    rows=[];began=time.monotonic();model.calls=model.writes=0;model.trace=[];model.update_records=[];model.diagnostics=True
    states=[]
    try:
        for document in documents:
            cache=model.reset(document['doc_id']);ids=document['ids'];total=0.;targets=0
            for start in range(0,len(ids)-1,config['model']['chunk']):
                check_time(deadline);end=min(start+config['model']['chunk'],len(ids)-1)
                logits,cache,_=model(torch.tensor([ids[start:end]],device=device),cache,generator)
                y=torch.tensor(ids[start+1:end+1],device=device)
                total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y,reduction='sum'));targets+=end-start
            if hasattr(model,'commit'):model.commit(cache)
            rows.append({'doc_id':document['doc_id'],'total_nll':total,'targets':targets})
            if cache['state'] is not None:
                s=cache['state'][0].float();unit=F.normalize(s,dim=-1)
                states.append({'doc_id':document['doc_id'],'norm':s.norm(dim=-1).tolist(),'cosine':(unit@unit.T).tolist(),'singular_values':torch.linalg.svdvals(s).tolist()})
        nll=sum(r['total_nll'] for r in rows)/sum(r['targets'] for r in rows)
        return {'status':'COMPLETE','mode':mode,'routing':'categorical_fixed_seed' if sample else 'argmax','mean_nll':nll,'ppl':math.exp(nll) if nll<700 else None,'documents':rows,'seconds':time.monotonic()-began,'cell_core_calls':model.calls,'workspace_writes':model.writes,'routes':model.trace,'state_updates':model.update_records,'states':states}
    finally:model.mode='normal';model.sample=False;model.diagnostics=False


def paired(a,b,config):
    import numpy as np
    if [r['doc_id'] for r in a['documents']]!=[r['doc_id'] for r in b['documents']]:raise ValueError('paired documents differ')
    x=np.array([[r['total_nll'],r['targets']] for r in a['documents']]);y=np.array([[r['total_nll'],r['targets']] for r in b['documents']])
    if not np.array_equal(x[:,1],y[:,1]):raise ValueError('paired target counts differ')
    rng=np.random.default_rng(config['evaluation']['bootstrap_seed']);values=[]
    for _ in range(config['evaluation']['bootstrap_repetitions']):
        i=rng.integers(0,len(x),len(x));values.append((x[i,0].sum()-y[i,0].sum())/y[i,0].sum())
    delta=(a['mean_nll']-b['mean_nll'])/b['mean_nll'];lo,hi=np.quantile(values,[.025,.975]);threshold=config['evaluation']['threshold']
    return {'relative_nll':delta,'ci95':[float(lo),float(hi)],'status':'OBSERVED_IMPROVEMENT' if delta<=-threshold and hi<0 else 'OBSERVED_DEGRADATION' if delta>=threshold and lo>0 else 'NO_CLEAR_DIFFERENCE','scope':'document sampling conditional on single fitted seed; no training variance or multiplicity correction'}


@torch.no_grad()
def base_profile(model,config,deadline):
    import statistics
    device=next(model.parameters()).device;model.eval();model.sample=False;model.mode='normal';model.diagnostics=False
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    prefill=[];decode=[];sample=torch.full((1,32),4,device=device,dtype=torch.long)
    for i in range(13):
        check_time(deadline);sync();start=time.perf_counter();_,cache,_=model(sample,model.reset('latency'));sync()
        if i>=3:prefill.append(time.perf_counter()-start)
        sync();start=time.perf_counter();model(sample[:,:1],cache);sync()
        if i>=3:decode.append(time.perf_counter()-start)
    return {'status':'MEASURED','precision':'FP32','prefill_tokens':32,'decode_context_tokens':32,'batch':1,'warmup':3,'repetitions':10,'prefill_median_seconds':statistics.median(prefill),'decode_median_seconds':statistics.median(decode),'decode_p95_seconds':sorted(decode)[-1],'decode_tokens_per_second':1/statistics.median(decode),'scope':'same model execution path; diagnostics disabled; initialized document cache per repetition'}


def route_summary(records,cells):
    import numpy as np
    counts=np.zeros(cells,dtype=int);transition=np.zeros((cells,cells),dtype=int);bins={};previous=None;entropies=[];magnitudes=[];state_terms=[]
    for row in records:
        i=row['cell'];counts[i]+=1;entropies.append(row['entropy']);key=str(row['position']//32)
        if key not in bins:bins[key]=np.zeros(cells,dtype=int)
        bins[key][i]+=1
        if row['step']==1 and previous is not None:transition[previous,i]+=1
        previous=i
        if row['update_magnitude'] is not None:magnitudes.append(row['update_magnitude'])
        if row['router_state_contribution_norm'] is not None:state_terms.append(row['router_state_contribution_norm'])
    total=max(1,int(counts.sum()))
    return {'selection_counts':counts.tolist(),'selection_proportions':(counts/total).tolist(),'dead_cell_ratio':float((counts==0).mean()),'load_imbalance_max_over_mean':float(counts.max()/max(1e-9,counts.mean())),'entropy_mean':float(np.mean(entropies)) if entropies else None,'transition_counts':transition.tolist(),'repeat_second_step_ratio':float(np.trace(transition)/max(1,transition.sum())),'position_bin_counts':{k:v.tolist() for k,v in bins.items()},'update_magnitude_mean':float(np.mean(magnitudes)) if magnitudes else None,'router_state_contribution_norm_mean':float(np.mean(state_terms)) if state_terms else None,'scope':'argmax fixed-validation diagnostic; no specialization conclusion'}


@torch.no_grad()
def profile(model,config,deadline):
    result=base_profile(model,config,deadline);check_time(deadline)
    from torch.profiler import profile as capture,ProfilerActivity
    activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if model.token.weight.is_cuda else [])
    ids=torch.full((1,2*config['model']['chunk']),4,device=model.token.weight.device,dtype=torch.long)
    with capture(activities=activities,record_shapes=True) as prof:
        model(ids,model.reset('profile'))
        if model.token.weight.is_cuda:torch.cuda.synchronize()
    events=prof.events();top=sorted(prof.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:12]
    result['kernel_launches']=sum('CUDA' in str(e.device_type) for e in events)
    result['host_hotspots']=[{'operator':e.key,'calls':e.count,'self_cpu_us':e.self_cpu_time_total,'self_device_us':getattr(e,'self_device_time_total',0)} for e in top]
    result['profiler_scope']='one frozen forward of two logical chunks; diagnostic overhead excluded from latency measurement; no training benchmark claim'
    return result
