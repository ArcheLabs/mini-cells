"""Paired document-cluster estimates and explicit single-seed screening decisions."""
import math
import numpy as np


def comparisons(arms,protocol):
    plan=protocol['evaluation'];reference=arms.get('R1',{}).get('holdout',{})
    if reference.get('status')!='COMPLETE':return {'status':'INCOMPLETE','reason':'R1 full holdout unavailable'}
    ids=[r['document_id'] for r in reference['documents']]
    draws=np.random.default_rng(plan['bootstrap_seed']).integers(0,len(ids),(plan['bootstrap_replicates'],len(ids)))
    result={}
    pairs=[(name,'R1') for name in protocol['arms'] if name!='R1']+[('SR','R2'),('T4','R2'),('T4','SR')]
    for candidate,baseline in pairs:
        a=arms.get(candidate,{}).get('holdout',{});b=arms.get(baseline,{}).get('holdout',{})
        if a.get('status')!='COMPLETE' or b.get('status')!='COMPLETE':
            result[candidate+'-'+baseline]={'status':'INCOMPLETE'};continue
        left={r['document_id']:r for r in a['documents']};right={r['document_id']:r for r in b['documents']}
        if set(left)!=set(ids) or set(right)!=set(ids):raise ValueError('unpaired holdout document IDs')
        if any(left[i]['prediction_tokens']!=right[i]['prediction_tokens'] for i in ids):raise ValueError('unpaired document target counts')
        tokens=np.array([left[i]['prediction_tokens'] for i in ids]);delta=np.array([left[i]['total_nll']-right[i]['total_nll'] for i in ids])
        estimates=delta[draws].sum(1)/tokens[draws].sum(1);mean=float(delta.sum()/tokens.sum())
        result[candidate+'-'+baseline]={'status':'COMPLETE','mean_delta_nll':mean,'relative_ppl_improvement':1-math.exp(mean),
            'bootstrap_95_ci':np.quantile(estimates,[.025,.975]).tolist(),'positive_document_fraction':float((delta>0).mean()),
            'negative_document_fraction':float((delta<0).mean()),'tied_document_fraction':float((delta==0).mean()),
            'unweighted_mean_document_delta_nll':float((delta/tokens).mean())}
    return {'status':'COMPLETE' if all(v['status']=='COMPLETE' for v in result.values()) else 'INCOMPLETE',
            'comparisons':result,'bootstrap_replicates':len(draws),'seed':plan['bootstrap_seed'],'document_order':ids,
            'shared_cluster_draws_across_all_comparisons':True,'estimand':'token-weighted NLL difference; negative favors candidate',
            'single_seed_limit':'document bootstrap does not estimate training-seed variance; development evidence only'}


def screen(arms,paired,protocol):
    rules=protocol['screen'];checks={};base=arms.get('R1',{})
    for name in ('R2','SR','T4'):
        arm=arms.get(name,{});pair=paired.get('comparisons',{}).get(name+'-R1',{})
        try:
            if pair['status']!='COMPLETE':raise ValueError('paired comparison incomplete')
            tests={'holdout_improvement':pair['relative_ppl_improvement']>=rules['minimum_holdout_ppl_improvement'],
                   'paired_ci':pair['bootstrap_95_ci'][1]<rules['delta_nll_ci_upper_max'],
                   'validation':arm['validation']['ppl']/base['validation']['ppl']<=rules['maximum_validation_ppl_ratio'],
                   'causal_propagation':arm['propagation']['status']=='COMPLETE' and not arm['propagation']['causal_cone_violations'] and base['propagation']['status']=='COMPLETE' and not base['propagation']['causal_cone_violations'],
                   'latency':arm['inference']['decode']['median_seconds']/base['inference']['decode']['median_seconds']<=rules['maximum_decode_latency_ratio']}
            checks[name]={'status':'EVALUATED','criteria':tests,'eligible':all(tests.values())}
        except (KeyError,TypeError,ValueError,ZeroDivisionError) as error:checks[name]={'status':'INCOMPLETE','reason':str(error),'eligible':False}
    eligible=[n for n,v in checks.items() if v['eligible']]
    selected=min(eligible,key=lambda n:arms[n]['holdout']['mean_nll']) if eligible else None
    complete=all(arms.get(n,{}).get('training_status')=='COMPLETE' for n in protocol['arms']) and all(v['status']=='EVALUATED' for v in checks.values())
    return {'status':'TOPOLOGY_CANDIDATE_SELECTED' if selected and complete else ('NO_CLEAR_TOPOLOGY_WINNER' if complete else 'INCOMPLETE_SCREEN'),
            'selected':selected if complete else None,'checks':checks,'rules':rules,'scientific_confirmation':False,
            'next_phase_automatic':False,'recommendation':'multi-seed follow-up only after review; no automatic training'}
