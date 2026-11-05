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
    old_rng=generator.get_state().clone();cpu_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None
    total=count(documents);device=next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True);model.train();model.sample=True;ce_sum=0.;new_baseline=baseline
    try:
        for document in documents:
            cache=model.reset(document['doc_id']);ids=document['ids']
            for start in range(0,len(ids)-1,config['model']['chunk']):
                check_time(deadline);end=min(start+config['model']['chunk'],len(ids)-1)
                x=torch.tensor([ids[start:end]],device=device);y=torch.tensor([ids[start+1:end+1]],device=device)
                logits,cache,actions=model(x,cache,generator)
                token_losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none')
                loss=token_losses.mean()
                objective=loss
                if actions:
                    detached=token_losses.detach();returns=detached.flip(0).cumsum(0).flip(0)
                    remaining=torch.arange(end-start,0,-1,device=device,dtype=detached.dtype)
                    advantage=returns-new_baseline*remaining
                    logp=torch.stack(actions).reshape(end-start,2)
                    objective=objective+config['router']['pg_weight']*(advantage[:,None]*logp).sum()/(end-start)
                    beta=config['router']['baseline_beta'];new_baseline=beta*new_baseline+(1-beta)*float(loss.detach())
                if not bool(objective.isfinite()):raise FloatingPointError('nonfinite objective')
                (objective*((end-start)/total)).backward();ce_sum+=float(loss.detach())*(end-start)
                cache=detach(cache) # Includes every T0 layer KV, recurrent Workspace KV and Cell State.
        gradients={name:float(torch.stack([p.grad.detach().norm() for p in module.parameters() if p.grad is not None]).norm()) for name,module in model.named_children() if any(p.grad is not None for p in module.parameters())}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['clip'],error_if_nonfinite=True)
        # FP32: there is no loss-scaler overflow to back off. Fail closed before mutation.
        if not all(bool(p.isfinite().all()) for p in model.parameters()):raise FloatingPointError('nonfinite parameters before update')
        optimizer.step()
        if not all(bool(p.isfinite().all()) for p in model.parameters()):raise FloatingPointError('nonfinite parameters after update; checkpoint invalid')
        return {'targets':total,'mean_nll':ce_sum/total,'gradient_norm':float(norm),'baseline':new_baseline,'module_gradient_norms':gradients}
    except Exception:
        optimizer.zero_grad(set_to_none=True);generator.set_state(old_rng);torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
        # Caller commits no counters/baseline and never resumes this failed group.
        raise


def optimizer(model,config):
    p=config['optimizer'];return torch.optim.AdamW(model.parameters(),lr=p['lr'],betas=tuple(p['betas']),weight_decay=p['weight_decay'])


def train(model,document_groups,config,deadline,out):
    opt=optimizer(model,config);device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(config['router_seed']);baseline=0.;tokens=0;rows=[]
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
    rows=[];began=time.monotonic();model.calls=model.writes=0;model.trace=[];model.diagnostics=True
    states=[]
    try:
        for document in documents:
            cache=model.reset(document['doc_id']);ids=document['ids'];total=0.;targets=0
            for start in range(0,len(ids)-1,config['model']['chunk']):
                check_time(deadline);end=min(start+config['model']['chunk'],len(ids)-1)
                logits,cache,_=model(torch.tensor([ids[start:end]],device=device),cache,generator)
                y=torch.tensor(ids[start+1:end+1],device=device)
                total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y,reduction='sum'));targets+=end-start
            rows.append({'doc_id':document['doc_id'],'total_nll':total,'targets':targets})
            if cache['state'] is not None:
                s=cache['state'][0].float();unit=F.normalize(s,dim=-1)
                states.append({'doc_id':document['doc_id'],'norm':s.norm(dim=-1).tolist(),'cosine':(unit@unit.T).tolist(),'singular_values':torch.linalg.svdvals(s).tolist()})
        nll=sum(r['total_nll'] for r in rows)/sum(r['targets'] for r in rows)
        return {'status':'COMPLETE','mode':mode,'routing':'categorical_fixed_seed' if sample else 'argmax','mean_nll':nll,'ppl':math.exp(nll) if nll<700 else None,'documents':rows,'seconds':time.monotonic()-began,'cell_core_calls':model.calls,'workspace_writes':model.writes,'routes':model.trace,'states':states}
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
def profile(model,config,deadline):
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
