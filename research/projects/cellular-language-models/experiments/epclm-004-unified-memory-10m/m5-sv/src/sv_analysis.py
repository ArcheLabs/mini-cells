"""Aggregate retained observations without changing the frozen success criteria."""
import math
import hashlib
import random
from pathlib import Path
import json
import numpy as np
from sv_metrics import summarize_queries, discrimination, paired_seed_episode_bootstrap, safe_ratio
from e0 import write


def episode_measurements(result, e6):
    base={q['query_id']:q['correct_on'] for q in e6['queries']}
    summary=summarize_queries(result['queries'],base)
    queries=result['queries'];summary.update(episode_id=result['episode_id'],condition=result['condition'])
    for key in ('coverage','precision','recall'):
        values=[q['admission'][key] for q in queries if q['admission'][key] is not None]
        summary['admission_'+key]=sum(values)/len(values) if values else None
    summary['nll']=sum(q['nll_on'] for q in queries)/len(queries)
    summary['query_ppl']=math.exp(summary['nll']) if summary['nll']<700 else None
    stores=result['resources']['stores']
    summary['store_occupancy']=sum(s['occupancy'] for s in stores)
    summary['store_write_count']=sum(s['writes'] for s in stores)
    summary['store_replacement_rate']=safe_ratio(sum(s['replacements'] for s in stores),sum(s['full_attempts'] for s in stores))
    summary['average_stored_token_age']=safe_ratio(sum((s['average_stored_age'] or 0)*s['occupancy'] for s in stores),sum(s['occupancy'] for s in stores))
    summary['store_read_preparation_seconds']=sum(q['store_read_timing']['store_read_preparation_seconds'] for q in queries)/len(queries)
    summary['store_update_seconds']=result['prefill_timing']['store_update_host_seconds']
    summary['query_latency_seconds']=sum(q['latency_on_seconds'] for q in queries)/len(queries)
    summary['query_count']=len(queries)
    summary['resource_bytes']=sum(result['resources'][k] for k in (
        'mamba_runtime_bytes','kv_bytes','predictor_bytes','store_payload_bytes','store_metadata_serialized_bytes','summary_materialized_bytes_per_chunk'))
    return summary


def condition_report(records, baseline_records):
    episodes=[];scores=[];failures=[];all_queries=[];base={r['episode_id']:r for r in baseline_records}
    for r in records:
        e6=base[r['episode_id']];episodes.append(episode_measurements(r,e6));all_queries.extend(r['queries'])
        for label in e6['recoverability']:
            if label['record_id'] in r['record_scores']:
                scores.append(r['record_scores'][label['record_id']]);failures.append(1-label['correct'])
    totals={q['query_id']:q['correct_on'] for r in baseline_records for q in r['queries']}
    no_history=[q for r in baseline_records for q in r.get('no_history',[])]
    guesses=[]
    for r in baseline_records:
        for q in r['queries']:
            seed=int(hashlib.sha256(('83101:'+q['query_id']).encode()).hexdigest()[:16],16)
            guesses.append(int(264+random.Random(seed).randrange(256)==q['expected_value_id']))
    report={'controls':{'no_history_em':safe_ratio(sum(q['correct'] for q in no_history),len(no_history)),
                'random_guess_expected_em':1/256,'random_guess_realized_em':safe_ratio(sum(guesses),len(guesses)),
                'random_guess_seed_rule':'sha256(83101:query_id), uniform over256 known value tokens'},'episodes':episodes,'query_summary':summarize_queries(all_queries,totals),
            'recoverability':discrimination(scores,failures),'episode_count':len(episodes),
            'aggregation':'query summary is descriptive; inferential unit is paired Episode within paired seed',
            'resources':{'static_parameters':records[0]['resources']['static_parameters'],
                'static_weight_bytes':records[0]['resources']['static_weight_bytes'],
                'mean_dynamic_bytes':float(np.mean([e['resource_bytes'] for e in episodes])),
                'mean_query_latency_seconds':float(np.mean([e['query_latency_seconds'] for e in episodes])),
                'mean_prefill_seconds':float(np.mean([r['prefill_seconds'] for r in records])),
                'by_first_episode':records[0]['resources']}}
    report['query_summary']['nll']=sum(q['nll_on'] for q in all_queries)/len(all_queries)
    report['query_summary']['query_ppl']=math.exp(report['query_summary']['nll']) if report['query_summary']['nll']<700 else None
    report['query_summary']['ppl_scope']='masked synthetic Query targets, not natural-language perplexity'
    return report


def seed_summary(values):
    a=np.array(values,dtype=float);mean=float(a.mean());sd=float(a.std(ddof=1)) if len(a)>1 else None
    # Student-t(2) critical value; descriptive seed interval, not a bootstrap.
    half=4.3026527299*sd/math.sqrt(3) if len(a)==3 else None
    return {'per_seed':a.tolist(),'mean':mean,'sample_std':sd,'seed_t_ci95':[mean-half,mean+half] if half is not None else None}


def scientific_decision(root, protocol, adequacy):
    root=Path(root);seeds=protocol['seeds'];conditions=protocol['conditions'];rules=protocol['decision']
    primary=rules['primary_long_conditions'];draws=rules['bootstrap_draws'];boot_seed=rules['bootstrap_seed']
    reports={};tables={};comparisons={};all_checks=[];condition_tables={}
    for cap in protocol['capacity_tiers']:
        tables[str(cap)]={}
        for arm in ('E6','E7-Random','E7-FIFO','E7-Innovation','E7-Oracle'):
            per_seed=[];reports[cap,arm]=[]
            for seed in seeds:
                outcomes=[]
                for condition in conditions:
                    actual_cap=0 if arm=='E6' else cap
                    path=root/str(seed['model_seed'])/arm/str(actual_cap)/(condition['name']+'.json')
                    r=json.loads(path.read_text());outcomes.append((condition['name'],r))
                reports[cap,arm].append(dict(outcomes))
                chosen=[e for name,r in outcomes if name in primary for e in r['episodes']]
                per_seed.append({k:float(np.mean([e[k] for e in chosen])) for k in ('em','em_off','delta_em','admission_coverage','hit_at_k','mrr')})
            tables[str(cap)][arm]={k:seed_summary([r[k] for r in per_seed]) for k in per_seed[0]}
        condition_tables[str(cap)]={}
        for condition in conditions:
            name=condition['name'];condition_tables[str(cap)][name]={}
            for arm in ('E6','E7-Random','E7-FIFO','E7-Innovation','E7-Oracle'):
                records=[reports[cap,arm][i][name] for i in range(3)]
                condition_tables[str(cap)][name][arm]={key:seed_summary([float(np.mean([e[key] for e in r['episodes']])) for r in records]) for key in ('em','delta_em','admission_coverage','hit_at_k','mrr')}
                if arm=='E7-Innovation':
                    for control in ('E6','E7-Random','E7-FIFO'):
                        pairs=[({e['episode_id']:e['em'] for e in records[i]['episodes']}, {e['episode_id']:e['em'] for e in reports[cap,control][i][name]['episodes']}) for i in range(3)]
                        condition_tables[str(cap)][name][arm][control+':paired_em']=paired_seed_episode_bootstrap(pairs,boot_seed,draws)
        def paired(arm, other, metric, other_metric=None):
            pairs=[]
            for i in range(3):
                left={e['episode_id']:e[metric] for name,r in reports[cap,arm][i].items() if name in primary for e in r['episodes']}
                right={e['episode_id']:e[other_metric or metric] for name,r in reports[cap,other][i].items() if name in primary for e in r['episodes']}
                pairs.append((left,right))
            return paired_seed_episode_bootstrap(pairs,boot_seed,draws)
        checks={};comparisons[str(cap)]={}
        for control in ('E7-Random','E7-FIFO'):
            for metric in ('admission_coverage','hit_at_k','mrr','em'):
                value=paired('E7-Innovation',control,metric);comparisons[str(cap)][control+':'+metric]=value
                minimum=rules['coverage_min_difference'] if metric=='admission_coverage' else 0.
                checks[control+':'+metric]=value['paired_difference']>=minimum and value['ci95'][0]>0 and all(v>0 for v in value['seed_differences'])
        utility=paired('E7-Innovation','E7-Innovation','em','em_off');comparisons[str(cap)]['store_on_off']=utility
        checks['utilization']=utility['paired_difference']>=rules['utility_min_delta_em'] and utility['ci95'][0]>0
        vs_e6=paired('E7-Innovation','E6','em');comparisons[str(cap)]['E6:em']=vs_e6
        checks['final_vs_e6']=vs_e6['ci95'][0]>0 and all(v>0 for v in vs_e6['seed_differences'])
        resource_rows=[];score_valid=True;score_known=True
        for i,seed in enumerate(seeds):
            for name in primary:
                r=reports[cap,'E7-Innovation'][i][name];base=reports[cap,'E6'][i][name]
                auc=r['recoverability']['auroc'];score_known=score_known and auc is not None
                score_valid=score_valid and auc is not None and auc>.5
                latency=r['resources']['mean_query_latency_seconds']/base['resources']['mean_query_latency_seconds']
                extra=r['resources']['mean_dynamic_bytes']-base['resources']['mean_dynamic_bytes']
                resource_rows.append({'seed':seed['model_seed'],'condition':name,'query_latency_ratio':latency,'extra_dynamic_bytes':extra})
        checks['recoverability_predictive']=bool(score_valid)
        checks['resources']=all(r['query_latency_ratio']<=rules['max_query_latency_ratio_vs_e6'] and r['extra_dynamic_bytes']<=rules['max_added_dynamic_bytes_vs_e6'] for r in resource_rows)
        comparisons[str(cap)]['resource_ratios']=resource_rows
        comparisons[str(cap)]['checks']=checks;comparisons[str(cap)]['recoverability_estimable']=bool(score_known)
        all_checks.extend(checks.values())
    adequate=all(item['stable'] for item in adequacy.values())
    recoverable=all(v['recoverability_estimable'] for v in comparisons.values())
    if not adequate or not recoverable:status='INCONCLUSIVE';outcome='E: insufficient convergence or recoverability classes'
    elif all(all_checks):status='SUPPORTED';outcome='A: all frozen criteria satisfied at both capacities'
    else:
        status='NOT_SUPPORTED'
        adm=all(v['checks']['E7-Random:admission_coverage'] and v['checks']['E7-FIFO:admission_coverage'] for v in comparisons.values())
        oracle=np.mean([tables[c]['E7-Oracle']['em']['mean']-tables[c]['E6']['em']['mean'] for c in tables])
        outcome='B: admission without sufficient utilization' if adm else ('C: Oracle benefit without Innovation support' if oracle>0 else 'D: no Oracle answer benefit')
    return {'status':'SCIENTIFIC_EVALUATION_COMPLETE','H5_STATUS':status,'outcome':outcome,'adequacy':adequacy,
            'primary_long_conditions':primary,'seed_tables':tables,'condition_tables':condition_tables,'paired_comparisons':comparisons,
            'e3_reference':'Original M5 Smoke only; no SV Test re-evaluation without available weights',
            'm6_authorized':False,'scope':'M5-SV diagnostic evidence, not M6 formal confirmation'}
