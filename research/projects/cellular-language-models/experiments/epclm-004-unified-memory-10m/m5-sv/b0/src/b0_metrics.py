"""The same complete-Episode forward and Query-only metrics for all B0 models."""
from collections import Counter
import math
from pathlib import Path
import random
import numpy as np
import torch
from b0_models import runtime_report
from data import batch, file_hash
from e0 import masked_loss


def binomial_tail(correct,total,p=1/256):
    if not 0<=correct<=total or not 0<p<1:raise ValueError('invalid binomial counts')
    if correct==0:return 1.
    logs=[math.lgamma(total+1)-math.lgamma(k+1)-math.lgamma(total-k+1)+k*math.log(p)+(total-k)*math.log1p(-p) for k in range(correct,total+1)]
    maximum=max(logs);return min(1.,math.exp(maximum)*sum(math.exp(x-maximum) for x in logs))


def confidence(correct,total):
    if not total:raise ValueError('empty confidence sample')
    z=1.959963984540054;p=correct/total;den=1+z*z/total
    mid=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,mid-half),min(1.,mid+half)]


def query_metrics(logits,labels):
    if logits.ndim!=2 or len(logits)!=len(labels) or not len(labels):raise ValueError('empty/misaligned Query logits')
    if not logits.isfinite().all() or not ((labels>=264)&(labels<520)).all():raise ValueError('invalid Query logits or target')
    logp=logits.log_softmax(-1);prob=logp.exp();gold=logits.gather(1,labels[:,None])[:,0]
    ids=torch.arange(logits.shape[-1],device=logits.device)
    ranks=1+(logits>gold[:,None]).sum(-1)+((logits==gold[:,None])&(ids[None]<labels[:,None])).sum(-1)
    pred=logits.argmax(-1);value_pred=logits[:,264:520].argmax(-1)+264
    other=logits.clone();other.scatter_(1,labels[:,None],float('-inf'))
    return {'nll':float(-logp.gather(1,labels[:,None]).mean()),'correct':int((pred==labels).sum()),
        'value_correct':int((value_pred==labels).sum()),'top5_correct':int((ranks<=5).sum()),
        'gold_rank':ranks.cpu().tolist(),'gold_probability':prob.gather(1,labels[:,None])[:,0].cpu().tolist(),
        'valid_value_probability_mass':prob[:,264:520].sum(-1).cpu().tolist(),
        'invalid_output_count':int(((pred<264)|(pred>=520)).sum()),
        'gold_logit_margin':(gold-other.max(-1).values).cpu().tolist(),
        'prediction_ids':pred.cpu().tolist(),'value_prediction_ids':value_pred.cpu().tolist(),
        'target_ids':labels.cpu().tolist()}


@torch.no_grad()
def evaluate(model,rows,batch_size=1,*,save_probabilities=None,bootstrap_seed=86003,bootstrap_draws=2000):
    if batch_size!=1:raise ValueError('B0 freezes Batch Size=1')
    device=next(model.parameters()).device;model.eval();details=[];arrays=[];distribution=Counter();loss=0.;targets=correct=values=top5=invalid=0
    for row in rows:
        if row.get('split')=='test':raise PermissionError('B0 must not evaluate Test')
        b=batch([row],device)
        logits=model(b.input_ids)  # Exactly the training forward, without cached/split calls.
        if logits.shape[:2]!=b.labels.shape:raise ValueError('model logits shape differs')
        q=b.query_mask
        if not torch.equal(q,b.labels!=-100):raise ValueError('Query supervision mask differs')
        scores=logits[q];gold=b.labels[q];r=query_metrics(scores,gold);n=len(gold)
        loss+=float(masked_loss(logits,b.labels))*n;targets+=n;correct+=r['correct'];values+=r['value_correct'];top5+=r['top5_correct'];invalid+=r['invalid_output_count']
        distribution.update(r['prediction_ids']);details.append({'episode_id':row['id'],'query_positions':row['query_positions'],'targets':n,**r})
        if save_probabilities is not None:arrays.append(scores.softmax(-1).cpu().numpy())
    if not targets:raise ValueError('no Query targets')
    episode_accuracy=np.array([r['correct']/r['targets'] for r in details]);rng=np.random.default_rng(bootstrap_seed)
    boot=episode_accuracy[rng.integers(0,len(details),(bootstrap_draws,len(details)))].mean(-1)
    all_values=lambda key:[v for r in details for v in r[key]]
    result={'nll':loss/targets,'query_exact_match':correct/targets,'full_vocab_exact_match':correct/targets,
        'value_constrained_exact_match':values/targets,'top5_accuracy':top5/targets,'supervised_targets':targets,'query_targets':targets,
        'correct':correct,'gold_rank_mean':float(np.mean(all_values('gold_rank'))),
        'gold_probability_mean':float(np.mean(all_values('gold_probability'))),
        'valid_value_probability_mass_mean':float(np.mean(all_values('valid_value_probability_mass'))),
        'invalid_output_rate':invalid/targets,'gold_logit_margin_mean':float(np.mean(all_values('gold_logit_margin'))),
        'query_prediction_counts':{str(k):v for k,v in sorted(distribution.items())},
        'full_vocab_random_reference':1/model.config.vocab,'value_random_reference':1/256,
        'wilson_query_ci95_nominal':confidence(correct,targets),'binomial_p_greater_than_1_over_256_nominal':binomial_tail(correct,targets),
        'nominal_test_warning':'Queries within an Episode are dependent; the query-level binomial test/CI is a requested nominal reference, not independent-trial evidence.',
        'episode_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_unit':'episode','bootstrap_seed':bootstrap_seed,'bootstrap_draws':bootstrap_draws,
        'execution_path':'one model.forward over the complete Episode, fresh Runtime per call',
        'episodes':details,'runtime':runtime_report(model)}
    if save_probabilities is not None:
        path=Path(save_probabilities);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:np.savez_compressed(stream,probabilities=np.concatenate(arrays),
            targets=np.array(all_values('target_ids')),episode_ids=np.array([r['episode_id'] for r in details for _ in range(r['targets'])]))
        result['full_query_probability_artifact']={'path':str(path),'sha256':file_hash(path),'bytes':path.stat().st_size,'shape':[targets,model.config.vocab]}
    return result
