"""Paired windows, sampled/greedy gaps and separate causal interventions."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F
from .accounting import ComputeCounter,summarize
from .reuse import paired_bootstrap_delta,make_batch

def diagnostics(actions,cell_count):
    flat=actions.detach().cpu().reshape(-1,actions.shape[-1]);a=cell_count+1
    counts=torch.bincount(flat.flatten(),minlength=a).double();prob=counts/counts.sum()
    unique,freq=torch.unique(flat,dim=0,return_counts=True);previous=torch.full_like(flat,a);previous[:,1:]=flat[:,:-1]
    transitions=torch.bincount((previous*a+flat).flatten(),minlength=(a+1)*a).reshape(a+1,a)
    ctx=flat==cell_count;has=ctx.any(-1);first=torch.where(has,ctx.long().argmax(-1)+1,torch.zeros(len(flat),dtype=torch.long))
    return {'no_context_fraction':float((~has).double().mean()),'first_context_step_histogram':torch.bincount(first,minlength=flat.shape[-1]+1).tolist(),'first_context_step_note':'1-based; bin 0 means no context read',
            'module_counts':counts.tolist(),'module_usage_entropy':float(-(prob*prob.clamp_min(1e-20).log()).sum()),
            'unique_paths':len(unique),'modal_path_fraction':float(freq.max()/len(flat)),
            'repeated_path_fraction':sum(len(set(r))<len(r) for r in flat.tolist())/len(flat),
            'context_read_fraction':float((flat==cell_count).float().mean()),'unused_cell_fraction':float((counts[:cell_count]==0).double().mean()),
            'start_and_predecessor_counts':transitions.tolist(),'path_histogram':{','.join(map(str,r)):int(f) for r,f in zip(unique.tolist(),freq.tolist())}}

@torch.inference_mode()
def evaluate(model,tokens,starts,groups,c,policy='greedy',*,edge_mode='normal',shuffle_order=False,edge_permutation=None,context_noop=False):
    was_training=model.training;model.eval();device=next(model.parameters()).device
    rng=torch.Generator(device=device).manual_seed(c['evaluation']['sampled_seed']);perm=np.random.default_rng(c['evaluation']['intervention_seed'])
    nll=[];paths=[];counts=[];changed=0;context_deltas=[];counter=ComputeCounter(model)
    try:
        with counter:
            for offset in range(0,len(starts),c['training']['batch_size']):
                x,y=make_batch(tokens,np.asarray(starts[offset:offset+c['training']['batch_size']]),c['training']['sequence_length'],device)
                replay=None
                if shuffle_order or context_noop:
                    counter.enabled=False;natural=model(x,policy='greedy');counter.enabled=True
                    replay=natural.actions.clone()
                    if shuffle_order:
                        for row in replay.reshape(-1,replay.shape[-1]):row.copy_(row[torch.as_tensor(perm.permutation(len(row)),device=device)])
                    changed+=int((replay!=natural.actions).any(-1).sum())
                out=model(x,policy=policy,generator=rng,replay=replay,edge_mode=edge_mode,edge_permutation=edge_permutation,
                    capture_context_states=True,context_mode='zero_replay' if context_noop else 'normal')
                counter.enabled=False;context_deltas.append(context_state_effects(model,out,y));counter.enabled=True
                losses=F.cross_entropy(out.logits.float().flatten(0,1),y.flatten(),reduction='none').reshape(y.shape).mean(-1)
                nll.extend(losses.cpu().tolist());paths.append(out.actions.cpu());counts.append(summarize(model,out,counter)['unique_touched_weights']['mean'])
        allpaths=torch.cat(paths);diag=diagnostics(allpaths,c['model']['cell_count']) if model.arm!='M0' else {'scope':'MoE paths are per-layer expert sets'}
        return {'nll':sum(nll)/len(nll),'perplexity':math.exp(sum(nll)/len(nll)),'sample_nll':nll,'story_groups':list(groups),'window_starts':list(map(int,starts)),
                'policy':policy,'routing':diag,'context_read_nll_probe':merge_context_effects(context_deltas),'actions':allpaths.tolist(),'compute':{**counter.report(len(starts)*c['training']['sequence_length']),
                    'controller_flops':sum(v for k,v in counter.by_module.items() if 'controller' in k or 'router' in k),'attention_flops':counter.by_module.get('context.qk_av',0),
                    'unique_touched_weights_mean':sum(counts)/len(counts),'non_gemm_flops':None},'order_changed_fraction':changed/(len(starts)*c['training']['sequence_length']) if shuffle_order else None}
    finally:model.train(was_training)

def intervention_suite(model,tokens,starts,groups,c):
    if model.arm not in ('R1','R2'):return {}
    base=evaluate(model,tokens,starts,groups,c);results={}
    settings=[('same_multiset_order_permuted',{'shuffle_order':True}),
              ('context_noop_same_path',{'context_noop':True})]
    if model.arm=='R2':
        settings += [('connection_removed',{'edge_mode':'zero'}),
                     ('connection_predecessor_shifted',{'edge_mode':'shuffle'})]
        for i in range(c['evaluation'].get('connection_permutation_repeats',3)):
            seed=c['evaluation']['intervention_seed']+i
            ids=np.random.default_rng(seed).permutation(c['model']['cell_count']+2)
            settings.append(('connection_prior_row_permuted_seed_'+str(seed),
                {'edge_mode':'permutation','edge_permutation':torch.tensor(ids,device=next(model.parameters()).device)}))
    for label,kwargs in settings:
        row=evaluate(model,tokens,starts,groups,c,**kwargs)
        row['paired_effect']=paired_bootstrap_delta(row['sample_nll'],base['sample_nll'],group_ids=groups,
            seed=c['evaluation']['bootstrap_seed'],replicates=c['evaluation']['bootstrap_replicates'])
        if 'edge_permutation' in kwargs:row['predecessor_row_permutation']=kwargs['edge_permutation'].tolist()
        if label in ('same_multiset_order_permuted','context_noop_same_path'):
            if not np.array_equal(np.sort(np.asarray(row['actions']),axis=-1),np.sort(np.asarray(base['actions']),axis=-1)):
                raise AssertionError('fixed-path intervention changed operation multiset')
        results[label]=row
    return {'natural':base,'interventions':results,'scientific_decision':False,
            'interpretation':'order effects alone show noncommutativity, not useful connections; compare R1 and R2 symmetrically'}

@torch.inference_mode()
def context_state_effects(model,out,labels):
    """Intermediate LM-head probe, not a trained intermediate objective or causal claim."""
    deltas=[];before=[];after=[]
    for event in out.context_events or []:
        y=labels.flatten()[event['rows']]
        a=F.cross_entropy(F.linear(model.final(event['before']),model.embedding.weight).float(),y,reduction='none')
        b=F.cross_entropy(F.linear(model.final(event['after']),model.embedding.weight).float(),y,reduction='none')
        before.extend(a.cpu().tolist());after.extend(b.cpu().tolist());deltas.extend((b-a).cpu().tolist())
    return {'calls':len(deltas),'before_sum':sum(before),'after_sum':sum(after),'delta_sum':sum(deltas)}

def merge_context_effects(rows):
    calls=sum(r['calls'] for r in rows)
    return {'calls':calls,'mean_nll_before':sum(r['before_sum'] for r in rows)/calls if calls else None,
            'mean_nll_after':sum(r['after_sum'] for r in rows)/calls if calls else None,
            'mean_nll_after_minus_before':sum(r['delta_sum'] for r in rows)/calls if calls else None,
            'scope':'read-visited training/evaluation states only; intermediate untuned head proxy, excluded from timing and cost; final causal contrast uses context_noop_same_path'}
