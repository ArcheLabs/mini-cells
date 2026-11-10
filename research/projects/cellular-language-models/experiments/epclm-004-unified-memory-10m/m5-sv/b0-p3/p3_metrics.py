"""Candidate-only objective, three output spaces and mapping-cluster uncertainty."""
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from p3_data import BASE,forward_batch,write,file_hash
sys.path.insert(0,str(BASE.parent/'b0-p2'))
from b0_models import runtime_report


def candidate_loss(logits,row):
    positions=row['query_positions'];candidates=row['candidates']
    if not candidates:raise ValueError('Empty candidate set')
    targets=[row['labels'][i] for i in positions]
    if not targets or any(t not in candidates for t in targets):raise ValueError('Target is not a candidate')
    scores=logits[positions][:,candidates]
    labels=torch.tensor([candidates.index(t) for t in targets],device=scores.device)
    return F.cross_entropy(scores,labels,reduction='sum')


def statistics(scores,row):
    z=scores.detach().double().cpu();c=row['candidates'];targets=torch.tensor([row['labels'][i] for i in row['query_positions']])
    if not z.isfinite().all():raise FloatingPointError('Nonfinite Query logits')
    n=len(targets);cand=z[:,c];local=torch.tensor([c.index(int(t)) for t in targets]);gold=z[torch.arange(n),targets]
    fullz=z.logsumexp(-1);valuez=z[:,264:520].logsumexp(-1);candidatez=cand.logsumexp(-1)
    lp=z-fullz[:,None];vlp=z[:,264:520]-valuez[:,None];clp=cand-candidatez[:,None]
    full=fullz-gold;type_loss=fullz-valuez;candidate=valuez-candidatez;assoc=candidatez-gold
    if not torch.allclose(full,type_loss+candidate+assoc,atol=1e-9,rtol=0):raise ValueError('Three-part NLL decomposition failed')
    predictions={'active':torch.tensor(c)[cand.argmax(-1)],'value':z[:,264:520].argmax(-1)+264,'full':z.argmax(-1)}
    r={'mapping_hash':row['mapping_hash'],'episode_id':row['id'],'queries':n,'candidate_count':len(c),
       'active_nll':float(assoc.mean()),'value_nll':float((valuez-gold).mean()),'full_nll':float(full.mean()),
       'type_nll':float(type_loss.mean()),'candidate_nll':float(candidate.mean()),
       'random_active_accuracy':1/len(c),'random_active_nll':float(np.log(len(c))),
       'value_mass':float((valuez-fullz).exp().mean()),'candidate_mass':float((candidatez-fullz).exp().mean()),
       'candidate_given_value_mass':float((candidatez-valuez).exp().mean()),
       'active_gold_probability':float(clp[torch.arange(n),local].exp().mean()),
       'active_gold_rank':(1+(cand>gold[:,None]).sum(-1)+((cand==gold[:,None])&(torch.tensor(c)[None]<targets[:,None])).sum(-1)).tolist(),
       'targets':targets.tolist(),'oracle_exact_match':sum(dict(row['bindings'])[k]==int(t) for k,t in zip(row['query_keys'],targets))/n,'execution':'complete valid Episode prefix;trailing PAD excluded'}
    for name,pred in predictions.items():
        counts=Counter(pred.tolist());r[name+'_correct']=int((pred==targets).sum());r[name+'_predictions']=pred.tolist()
        r[name+'_dominant_rate']=max(counts.values())/n;r[name+'_unique_predictions']=len(counts)
    r['active_top5_correct']=sum(rank<=5 for rank in r['active_gold_rank'])
    r['active_entropy']=float(-(clp.exp()*clp).sum(-1).mean());r['value_entropy']=float(-(vlp.exp()*vlp).sum(-1).mean())
    r['full_entropy']=float(-(lp.exp()*lp).sum(-1).mean())
    return r,{'full_log_probabilities':lp.float().numpy(),'value_log_probabilities':vlp.float().numpy(),
        'candidate_log_probabilities':clp.float().numpy()}


def aggregate(records,draws=2000,seed=96031):
    if not records:raise ValueError('Empty evaluation')
    queries=sum(r['queries'] for r in records);groups=defaultdict(list)
    for r in records:groups[r['mapping_hash']].append(r)
    means={key:sum(r[key]*r['queries'] for r in records)/queries for key in
        ('active_nll','value_nll','full_nll','type_nll','candidate_nll','random_active_accuracy','random_active_nll',
         'value_mass','candidate_mass','candidate_given_value_mass','active_gold_probability','active_entropy','value_entropy','full_entropy')}
    clusters=[]
    for mapping,rows in groups.items():
        count=sum(r['queries'] for r in rows);clusters.append({'mapping_hash':mapping,
            'active_accuracy':sum(r['active_correct'] for r in rows)/count,
            'active_nll':sum(r['active_nll']*r['queries'] for r in rows)/count,
            'random_accuracy':sum(r['random_active_accuracy']*r['queries'] for r in rows)/count})
    sample=np.array([[c[k] for k in ('active_accuracy','active_nll','random_accuracy')] for c in clusters])
    rng=np.random.default_rng(seed);boot=sample[rng.integers(len(sample),size=(draws,len(sample)))].mean(1)
    eligible=[r for r in records if r['candidate_count']>1]
    distributions={name:dict(sorted(Counter(p for r in records for p in r[name+'_predictions']).items())) for name in ('active','value','full')}
    return {**means,'queries':queries,'episodes':len(records),'mapping_clusters':len(groups),
        **{name+'_exact_match':sum(r[name+'_correct'] for r in records)/queries for name in ('active','value','full')},
        'active_top5_accuracy':sum(r['active_top5_correct'] for r in records)/queries,
        'active_gold_rank_mean':float(np.mean([rank for r in records for rank in r['active_gold_rank']])),
        'prediction_distribution':distributions,'dominant_token_rate':{name:max(counts.values())/queries for name,counts in distributions.items()},
        'unique_predicted_tokens':{name:len(counts) for name,counts in distributions.items()},
        'collapse_episode_fraction':sum(r['active_dominant_rate']>=.9 for r in eligible)/len(eligible) if eligible else 0.,
        'active_bootstrap_95ci':np.quantile(boot[:,0],[.025,.975]).tolist(),
        'active_nll_bootstrap_95ci':np.quantile(boot[:,1],[.025,.975]).tolist(),
        'advantage_bootstrap_95ci':np.quantile(boot[:,0]-boot[:,2],[.025,.975]).tolist(),
        'bootstrap_unit':'complete Mapping/Episode cluster','clusters':clusters,'records':records,
        'uniform_value_accuracy':1/256,'uniform_value_nll':float(np.log(256)),
        'uniform_full_accuracy':1/8192,'uniform_full_nll':float(np.log(8192)),'oracle_exact_match':1.}


def memory_snapshot(model,row,stream,norms):
    if not hasattr(model,'last_runtime'):return {'status':'NOT_APPLICABLE'}
    runtime=model.last_runtime;states=[pair[1][stream].float() for pair in runtime.mamba.values()]
    if runtime.position!=row['valid_length']:raise ValueError('PAD/state leakage')
    result={'mapping_hash':row['mapping_hash'],'stream':stream,'valid_prefix':runtime.position,'padding_processed':False,
        'episode_reset_count':1,'summary_norms':norms,'state_norms':[float(t.norm()) for t in states],
        'state_activation':[{'mean':float(t.mean()),'std':float(t.std()),'max_abs':float(t.abs().max()),'finite':bool(t.isfinite().all())} for t in states],
        'state_update_frequency_per_block':int(np.ceil(runtime.position/model.chunk)),
        'state_update_norm':'NOT_AVAILABLE: original forward exposes only final detached state',
        'state_saturation':'NOT_AVAILABLE: no calibrated threshold',
        'runtime_resources_batch_total':runtime_report(model),'causal_store_contribution':'NOT_AVAILABLE: no intervention'}
    if any(not entry['finite'] for entry in result['state_activation']):raise FloatingPointError('Nonfinite recurrent state')
    if runtime.stores:
        store=runtime.stores[stream];metrics=store.metrics(runtime.position);positions={k:(2+3*i,3+3*i) for i,(k,v) in enumerate(row['bindings'])};reads=[]
        for trace in model.last_trace:
            if trace['stream']!=stream:continue
            pos=trace['query_position'];key=row['input_ids'][pos-1];kp,vp=positions[key];selected=trace['selected_positions']
            reads.append({**trace,'key_hit_top1':selected[0]==kp,'key_hit_top5':kp in selected[:5],
                'value_hit_top1':selected[0]==vp,'value_hit_top5':vp in selected[:5],
                'correct_key_hit':kp in selected,'correct_fact_value_hit':vp in selected,
                'key_value_joint_hit':kp in selected and vp in selected,
                'any_gold_value_token_hit':any(p>=0 and row['input_ids'][p]==dict(row['bindings'])[key] for p in selected),
                'correct_binding_retrieval':'NOT_AVAILABLE: fact-position co-selection is not semantic binding recall',
                'retrieval_score':'NOT_AVAILABLE: original trace does not export scores'})
        result['store']={'write_attempts':metrics['seen'],'accepted_writes':metrics['writes'],'rejected_writes':metrics['seen']-metrics['writes'],
            'occupancy':metrics['occupancy'],'evictions':metrics['replacements'],'query_layer_retrieval_attempts':len(reads),
            'retrieval_records':reads,'predictor':{'ema_norm':float(runtime.predictor[stream].norm()),'score_mean':metrics['score_mean'],'score_variance':metrics['score_variance']},'raw_metrics':metrics}
    else:result['store']={'status':'NOT_APPLICABLE'}
    return result


@torch.no_grad()
def evaluate(model,rows,folder,*,full_probabilities=False,draws=2000):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);model.eval();records=[];memory=[];probabilities=[];norms=[];handles=[]
    if any(row['split']!='validation' for row in rows):raise PermissionError('Validation only;Test is sealed')
    def observe(module,args,output):norms.append(output.detach().float().flatten(1).norm(dim=1))
    if hasattr(model,'summary'):handles=[layer.register_forward_hook(observe) for layer in model.summary.values()]
    try:
        for offset in range(0,len(rows),8):
            group=rows[offset:offset+8];logits=forward_batch(model,group)
            observed=torch.stack(norms).T.cpu().tolist() if norms else [[] for _ in group];norms.clear()
            for stream,row in enumerate(group):
                r,a=statistics(logits[stream,row['query_positions']],row);records.append(r)
                memory.append(memory_snapshot(model,row,stream,observed[stream]));probabilities.append(a)
    finally:
        for handle in handles:handle.remove()
    result=aggregate(records,draws);write(folder/'metrics.json',result);write(folder/'memory-diagnostics.json',memory)
    width=max(len(row['candidates']) for row in rows);candidate=[];candidate_ids=[]
    for row,a in zip(rows,probabilities):
        count=len(row['query_positions']);k=len(row['candidates']);candidate.append(np.pad(a['candidate_log_probabilities'],((0,0),(0,width-k)),constant_values=-np.inf))
        candidate_ids.extend([row['candidates']+[-1]*(width-k)]*count)
    arrays={'candidate_log_probabilities':np.concatenate(candidate),'candidate_ids':np.array(candidate_ids),
        'value_log_probabilities':np.concatenate([a['value_log_probabilities'] for a in probabilities]),
        'targets':np.array([t for r in records for t in r['targets']]),
        'mapping_hashes':np.array([r['mapping_hash'] for r in records for _ in range(r['queries'])]),
        **{name+'_predictions':np.array([v for r in records for v in r[name+'_predictions']]) for name in ('active','value','full')}}
    if full_probabilities:arrays['full_log_probabilities']=np.concatenate([a['full_log_probabilities'] for a in probabilities])
    np.savez_compressed(folder/'probabilities.npz',**arrays)
    write(folder/'probability-manifest.json',{'sha256':file_hash(folder/'probabilities.npz'),'full_vocabulary_array':full_probabilities,
        'queries':len(arrays['targets']),'candidate_padding':'-1 IDs and -inf log probabilities;excluded from all computations'})
    return result


def stage1_gate(evaluation,online,p):
    g=p['gate'];checks={
        'validation_accuracy':evaluation['active_exact_match']>=g['validation_accuracy_min'],
        'ci_above_actual_random':evaluation['active_bootstrap_95ci'][0]>evaluation['random_active_accuracy'],
        'nll_advantage':evaluation['random_active_nll']-evaluation['active_nll']>=g['nll_improvement_min'],
        'all_stage_online_accuracy':online['correct']/online['queries']>=g['training_online_accuracy_min'],
        'no_severe_active_collapse':evaluation['collapse_episode_fraction']<g['collapse_episode_fraction'],
        'numerical_status':online['numerically_finite']}
    return {'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,'online_scope':g['training_online_scope']}
