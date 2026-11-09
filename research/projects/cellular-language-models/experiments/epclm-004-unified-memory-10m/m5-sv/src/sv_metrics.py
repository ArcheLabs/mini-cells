"""Episode-aware scientific metrics; undefined quantities stay explicit."""
import math
import numpy as np


def safe_ratio(a,b): return a/b if b else None


def admission(stored, required, future):
    s,g,u=set(stored),set(required),set(future)
    return {'coverage':safe_ratio(len(s&g),len(g)), 'precision':safe_ratio(len(s&u),len(s)),
            'recall':safe_ratio(len(s&u),len(u))}


def retrieval(ranked, required):
    g=set(required); ranked=list(dict.fromkeys(r for r in ranked if r is not None))
    rank=next((i+1 for i,r in enumerate(ranked) if r in g),None)
    return {'hit_at_1':int(bool(ranked) and ranked[0] in g), 'hit_at_k':int(rank is not None),
            'recall_at_k':safe_ratio(len(set(ranked)&g),len(g)), 'rr':1/rank if rank else 0.}


def discrimination(scores, failures):
    scores=np.asarray(scores,dtype=float); labels=np.asarray(failures,dtype=int)
    if len(scores)!=len(labels) or not np.isfinite(scores).all():raise ValueError('invalid scores/labels')
    npos=int(labels.sum()); nneg=len(labels)-npos
    result={'n':len(labels),'unrecoverable':npos,'recoverable':nneg,
            'score_distribution':{}}
    for value,name in ((0,'recoverable'),(1,'unrecoverable')):
        group=scores[labels==value]
        result['score_distribution'][name]={'n':len(group),'mean':float(group.mean()) if len(group) else None,
            'quantiles':np.quantile(group,[0,.25,.5,.75,1]).tolist() if len(group) else None}
    if not npos or not nneg:
        return {**result,'auroc':None,'auprc':None,'undefined_reason':'single recoverability class'}
    # Average ranks for tied scores; AP integrates precision at tied thresholds.
    order=np.argsort(-scores,kind='stable'); y=labels[order]; s=scores[order]
    tp=fp=area=ap=0.; last_tp=last_fp=0.
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[j]==s[i]:j+=1
        tp+=int(y[i:j].sum());fp+=(j-i)-int(y[i:j].sum())
        area+=(fp-last_fp)*(tp+last_tp)/2
        ap+=(tp-last_tp)/npos*tp/(tp+fp)
        last_tp,last_fp=tp,fp;i=j
    return {**result,'auroc':area/(npos*nneg),'auprc':ap,'auprc_definition':'average precision at distinct score thresholds'}


def summarize_queries(rows, e6=None):
    n=len(rows); hits=[r for r in rows if r['retrieval']['hit_at_k']]
    result={'queries':n,'em':safe_ratio(sum(r['correct_on'] for r in rows),n),
            'em_off':safe_ratio(sum(r['correct_off'] for r in rows),n),
            'delta_em':safe_ratio(sum(r['correct_on']-r['correct_off'] for r in rows),n),
            'conditional_em':safe_ratio(sum(r['correct_on'] for r in hits),len(hits))}
    for key in ('hit_at_1','hit_at_k','recall_at_k','rr'):
        result[key if key!='rr' else 'mrr']=safe_ratio(sum(r['retrieval'][key] for r in rows),n)
    if e6 is not None:
        wrong=[r for r in rows if not e6[r['query_id']]];right=[r for r in rows if e6[r['query_id']]]
        result.update(rescue_rate=safe_ratio(sum(r['correct_on'] for r in wrong),len(wrong)),
                      regression_rate=safe_ratio(sum(not r['correct_on'] for r in right),len(right)))
    return result


def bootstrap_paired(left, right, seed=83101, draws=2000):
    """Each dictionary maps Episode ID to its query-mean measurement."""
    if set(left)!=set(right) or not left:raise ValueError('unpaired/empty episodes')
    ids=sorted(left);differences=np.array([left[k]-right[k] for k in ids]);rng=np.random.default_rng(seed)
    boot=np.array([differences[rng.integers(0,len(ids),len(ids))].mean() for _ in range(draws)])
    return {'episodes':len(ids),'paired_difference':float(differences.mean()),
            'ci95':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_seed':seed,'draws':draws,'unit':'episode'}


def hierarchical_paired(pairs, seed=83101, draws=2000):
    """Resample paired training seeds, then paired episodes within each seed."""
    if len(pairs)<3:raise ValueError('three paired seeds required')
    rng=np.random.default_rng(seed);arrays=[]
    for left,right in pairs:
        if set(left)!=set(right) or not left:raise ValueError('unpaired episodes')
        arrays.append(np.array([left[k]-right[k] for k in sorted(left)]))
    means=np.array([a.mean() for a in arrays]);boot=[]
    for _ in range(draws):
        samples=[]
        for i in rng.integers(0,len(arrays),len(arrays)):
            a=arrays[i];samples.append(a[rng.integers(0,len(a),len(a))].mean())
        boot.append(np.mean(samples))
    return {'paired_difference':float(means.mean()),'seed_differences':means.tolist(),
            'seed_std':float(means.std(ddof=1)),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'bootstrap_seed':seed,'draws':draws,'unit':'paired seed then episode'}
