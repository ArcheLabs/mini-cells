"""Read-only full-Episode observations and stable NLL decomposition."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'b0/src'))
from b0_metrics import evaluate as base_evaluate
from b0_models import runtime_report
from data import file_hash
from e0 import write


def probability_metrics(logits,targets,episodes,seed=86003,draws=2000):
    scores=torch.as_tensor(logits,dtype=torch.float64)
    target=torch.as_tensor(targets,dtype=torch.long)
    logp=scores.log_softmax(-1);p=logp.exp();valid=scores[:,264:520].logsumexp(-1)-scores.logsumexp(-1)
    full=-logp[torch.arange(len(target)),target];type_nll=-valid;conditional=full-type_nll
    if not torch.allclose(full,type_nll+conditional,atol=1e-10,rtol=0):raise ValueError('NLL decomposition failed')
    pred=scores.argmax(-1);value_pred=scores[:,264:520].argmax(-1)+264
    counts=Counter(pred.tolist());n=len(target)
    ranks=1+(scores>scores[torch.arange(n),target,None]).sum(-1)
    ranks+=((scores==scores[torch.arange(n),target,None])&(torch.arange(scores.shape[-1])[None]<target[:,None])).sum(-1)
    cluster=[]
    for episode in dict.fromkeys(episodes):
        mask=torch.tensor([e==episode for e in episodes]);cluster.append({'episode_id':episode,
            'exact_match':float((pred[mask]==target[mask]).double().mean()),'nll':float(full[mask].mean()),
            'type_nll':float(type_nll[mask].mean()),'conditional_nll':float(conditional[mask].mean())})
    values=np.array([[r[k] for k in ('exact_match','nll','type_nll','conditional_nll')] for r in cluster])
    rng=np.random.default_rng(seed);sample=values[rng.integers(len(values),size=(draws,len(values)))].mean(1)
    intervals={k:np.quantile(sample[:,i],[.025,.975]).tolist() for i,k in enumerate(('exact_match','nll','type_nll','conditional_nll'))}
    result={'full_vocabulary_nll':float(full.mean()),'type_nll':float(type_nll.mean()),'value_conditional_nll':float(conditional.mean()),
        'value_constrained_exact_match':float((value_pred==target).double().mean()),
        'prediction_entropy':float(-(p*logp).sum(-1).mean()),'unique_predicted_tokens':len(counts),
        'dominant_token_rate':max(counts.values())/n,'top20_predicted_tokens':counts.most_common(20),
        'prediction_counts':dict(sorted(counts.items())),
        'value_prediction_fraction':float(((pred>=264)&(pred<520)).double().mean()),
        'key_prediction_fraction':float(((pred>=8)&(pred<264)).double().mean()),
        'noise_special_prediction_fraction':float(((pred<8)|((pred>=520)&(pred<536))).double().mean()),
        'other_token_prediction_fraction':float((pred>=536).double().mean()),
        'gold_rank_histogram':dict(sorted(Counter(ranks.tolist()).items())),
        'episode_metrics':cluster,'episode_bootstrap_95ci':intervals,'bootstrap_draws':draws,
        'bootstrap_unit':'Episode;8 Queries are clustered','uniform_value_conditional_nll':float(np.log(256)),
        'decomposition_max_abs_error':float((full-type_nll-conditional).abs().max())}
    return result,logp.float().numpy()


def memory_snapshot(model,row,summary_norms):
    if not hasattr(model,'last_runtime'):return {'status':'NOT_APPLICABLE','architecture':'Full Context Transformer'}
    runtime=model.last_runtime
    states=[pair[1].float() for pair in runtime.mamba.values()]
    result={'episode_id':row['id'],'position':runtime.position,'fresh_episode_state':runtime.position==512,
        'runtime_resources':runtime_report(model),'summary_norms':summary_norms,'state_norms':[float(s.norm()) for s in states],
        'state_activation': [{'mean':float(s.mean()),'std':float(s.std()),'max_abs':float(s.abs().max()),'finite':bool(s.isfinite().all())} for s in states],
        'state_update_norm':'NOT_AVAILABLE: original forward exposes only final detached state',
        'state_update_frequency':'16 chunks per Mamba block per512-token Episode',
        'state_saturation':'NOT_AVAILABLE: no calibrated saturation threshold',
        'state_reset':'New runtime created by every complete model(input_ids) call'}
    if runtime.stores:
        store=runtime.stores[0];metrics=store.metrics(runtime.position)
        positions={8+k:(2+3*i,3+3*i) for i,(k,v) in enumerate(row['bindings'])}
        reads=[]
        for trace in model.last_trace:
            q=trace['query_position'];key=row['input_ids'][q-1];kp,vp=positions[key]
            selected=trace['selected_positions'];reads.append({**trace,'correct_key_position':kp,'correct_value_position':vp,
                'key_hit_top1':selected[0]==kp,'key_hit_top5':kp in selected[:5],
                'value_hit_top1':selected[0]==vp,'value_hit_top5':vp in selected[:5],
                'correct_key_retrieved':kp in selected,'correct_value_retrieved':vp in selected})
        result['store']={'write_attempts':metrics['seen'],'accepted_writes':metrics['writes'],
            'rejected_writes':metrics['seen']-metrics['writes'],'occupancy':metrics['occupancy'],
            'evictions':metrics['replacements'],'query_layer_retrieval_attempts':len(reads),'retrieval_records':reads,
            'key_hit_rate':sum(r['correct_key_retrieved'] for r in reads)/len(reads) if reads else None,
            'correct_value_hit_rate':sum(r['correct_value_retrieved'] for r in reads)/len(reads) if reads else None,
            'retrieval_scores':'NOT_AVAILABLE: original trace exposes selected positions only',
            'causal_effect_on_prediction':'NOT_AVAILABLE: no intervention/ablation authorized in this run',
            'predictor':{'final_ema_norm':float(runtime.predictor.float().norm()),'innovation_mean':metrics['score_mean'],
                'innovation_variance':metrics['score_variance'],'score_histogram':metrics['score_histogram']},
            'raw_metrics':metrics,
            'interpretation':'Position overlap with factual Key/Value tokens, not semantic retrieval correctness or a causal benefit claim'}
    else:result['store']={'status':'NOT_APPLICABLE'}
    return result


def evaluate(model,rows,folder,seed=86003,draws=2000):
    if any(row['split']=='test' for row in rows):raise PermissionError('Test is forbidden')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);captured=[];telemetry=[];norms=[];handles=[]
    def summary_hook(module,args,output):norms.append(output.detach().float().norm())
    if hasattr(model,'summary'):
        handles.extend(layer.register_forward_hook(summary_hook) for layer in model.summary.values())
    def observer(module,args,output):
        row=rows[len(captured)];positions=row['query_positions'];captured.append(output[0,positions].detach().cpu())
        observed=torch.stack(norms).cpu().tolist() if norms else [];norms.clear()
        telemetry.append(memory_snapshot(model,row,observed))
    handles.append(model.register_forward_hook(observer))
    try:
        result=base_evaluate(model,rows,1,save_probabilities=folder/'probabilities.npz',bootstrap_seed=seed,bootstrap_draws=draws)
    finally:
        for handle in handles:handle.remove()
    labels=[y for row in rows for y in row['labels'] if y!=-100];episodes=[r['id'] for r in rows for _ in range(8)]
    quality,logp=probability_metrics(torch.cat(captured).numpy(),labels,episodes,seed,draws)
    if abs(quality['full_vocabulary_nll']-result['nll'])>2e-5:raise ValueError('Full NLL disagreement')
    result.update(quality)
    with np.load(folder/'probabilities.npz') as saved:arrays={k:saved[k] for k in saved.files}
    arrays['log_probabilities']=logp
    np.savez_compressed(folder/'probabilities.npz',**arrays)
    result['full_query_probability_artifact']['sha256']=file_hash(folder/'probabilities.npz')
    result['full_query_probability_artifact']['bytes']=(folder/'probabilities.npz').stat().st_size
    write(folder/'metrics.json',result);write(folder/'memory-diagnostics.json',telemetry)
    return result
