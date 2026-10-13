"""Paired evidence and frozen criteria; scientific approval remains a human gate."""
from __future__ import annotations
import math
from .reuse import paired_bootstrap_delta,canonical_sha256

FLAGS={'scientific_decision':False,'path_mechanism_supported':False,
       'compute_adaptivity_supported':False,'system_superiority_supported':False}

def analyze(rows,c,p):
    if set(rows)!=set(('R0','R1','R2','M0')) or any(
        r.get('status')!='completed' or not r.get('decode_profile') for r in rows.values()
    ):return {**FLAGS,'status':'INCOMPLETE'}
    ids=[r['identity'] for r in rows.values()]
    common={k:v for k,v in ids[0].items() if k!='arm'}
    if any({k:v for k,v in i.items() if k!='arm'}!=common for i in ids):
        raise ValueError('source/config/protocol/data/seed/role identities do not match')
    if any(r['identity']['arm']!=a for a,r in rows.items()):raise ValueError('arm identity mismatch')
    if common['config_sha256']!=canonical_sha256(c) or common['protocol_sha256']!=canonical_sha256(p):
        raise ValueError('analysis does not match frozen configuration/protocol')
    role=common['role']
    expected=c['data'].get('confirmation_manifest_sha256') if role=='confirmation' else c['data']['manifest_sha256']
    if common['data_sha256']!=expected:raise ValueError('analysis data identity mismatch')
    evals={a:r['development_eval'] if role=='development' else r['confirmation_holdout'] for a,r in rows.items()}
    first=evals['R0']
    if any(e['window_starts']!=first['window_starts'] or e['story_groups']!=first['story_groups'] for e in evals.values()):
        raise ValueError('unpaired story/window identity')
    comparisons={}
    for a,b in [('R1','R0'),('R2','R1'),('R2','M0')]:
        e,f=evals[a],evals[b]
        ci=paired_bootstrap_delta(e['sample_nll'],f['sample_nll'],group_ids=e['story_groups'],
            seed=c['evaluation']['bootstrap_seed'],replicates=c['evaluation']['bootstrap_replicates'])
        comparisons[a+'_vs_'+b]={
            'ppl_ratio':e['perplexity']/f['perplexity'],
            'ppl_ratio_ci':[math.exp(v) for v in ci['paired_bootstrap_95pct_ci']],
            'cost_ratio':e['compute']['observed_gemm_flops_per_token']/f['compute']['observed_gemm_flops_per_token'],
            'decode_p50_ratio':rows[a]['decode_profile']['warm_p50_ms']/rows[b]['decode_profile']['warm_p50_ms'],
            'paired_nll':ci}
    candidate=comparisons['R2_vs_R1'];signal=candidate['ppl_ratio_ci'][1]<1 and candidate['cost_ratio']<=1
    frozen=all(v is not None for v in p['thresholds'].values()) and p['status']=='REGISTERED'
    criteria={};intervention_evidence={}
    if frozen:
        t=p['thresholds'];r=rows['R2'];route=r['routing_diagnostics']
        state=comparisons['R1_vs_R0'];moe=comparisons['R2_vs_M0']
        criteria['state_value']=state['ppl_ratio_ci'][1]<t['state_ppl_ratio_ci_upper'] and state['cost_ratio']<=t['state_cost_ratio_max']
        criteria['connection_value']=candidate['ppl_ratio_ci'][1]<t['connection_ppl_ratio_ci_upper'] and candidate['cost_ratio']<=t['connection_cost_ratio_max']
        criteria['noncollapsed_content_paths']=route['modal_path_fraction']<t['collapse_max_fraction'] and route['unused_cell_fraction']<=t['max_unused_cells_fraction']
        criteria['uses_context']=route['context_read_fraction']>=t['min_context_read_fraction']
        criteria['sampled_greedy_gap']=abs(r['sampled_greedy_nll_gap'])<=t['max_sampled_greedy_nll_gap']
        for label in ('connection_removed','connection_shuffled','same_multiset_order_permuted'):
            evidence=r.get('interventions',{}).get('interventions',{}).get(label)
            if evidence is None:
                criteria[label]=False;continue
            if evidence['window_starts']!=first['window_starts'] or evidence['story_groups']!=first['story_groups']:
                raise ValueError('intervention windows are not paired')
            ci=evidence['paired_effect']['paired_bootstrap_95pct_ci'];intervention_evidence[label]=ci
            criteria[label]=ci[0]>t['intervention_nll_delta_ci_lower']
            if label=='same_multiset_order_permuted':criteria[label]=criteria[label] and evidence['order_changed_fraction']>0
        criteria['moe_competitive']=moe['ppl_ratio_ci'][1]<=t['moe_quality_ratio_max'] and moe['cost_ratio']<1 and moe['decode_p50_ratio']<=t['moe_decode_p50_ratio_max']
    return {**FLAGS,'status':('DEVELOPMENT_SIGNAL' if signal else 'DEVELOPMENT_NO_PATH_SIGNAL') if role=='development' else 'AWAITING_CONFIRMATION_MATRIX_REVIEW',
            'identity':common,'comparisons':comparisons,'frozen_criteria':criteria,'criteria_frozen':frozen,
            'intervention_nll_ci':intervention_evidence,
            'boundary':'one seed never establishes path value or MoE superiority; R3 is not implemented'}

def matrix(reports,c,p):
    expected=set(c['seeds']['confirmation']);seen={r.get('seed') for r in reports}
    complete=(len(expected)>=3 and len(reports)==len(expected) and seen==expected and all(
        r.get('status')=='AWAITING_CONFIRMATION_MATRIX_REVIEW' and r.get('criteria_frozen') and
        r.get('identity',{}).get('role')=='confirmation' and r['identity']['seed']==r['seed'] for r in reports))
    common=[{k:v for k,v in r.get('identity',{}).items() if k!='seed'} for r in reports]
    if complete and any(i!=common[0] for i in common):raise ValueError('confirmation matrix identities differ')
    mechanism_keys=('state_value','connection_value','noncollapsed_content_paths','uses_context',
                    'sampled_greedy_gap','connection_removed','connection_shuffled','same_multiset_order_permuted')
    mechanism=complete and all(all(r['frozen_criteria'].get(k) is True for k in mechanism_keys) for r in reports)
    competitive=mechanism and all(r['frozen_criteria'].get('moe_competitive') is True for r in reports)
    verdict=('PATH_MECHANISM_AND_SYSTEM_CANDIDATE' if competitive else 'PATH_MECHANISM_SIGNAL_ONLY' if mechanism else 'NOT_SUPPORTED') if complete else 'INCOMPLETE'
    return {**FLAGS,'status':'AWAITING_RESEARCH_REVIEW' if complete else 'INCOMPLETE',
            'evidence_verdict':verdict,'all_seeds_direction_and_cost_consistent':mechanism,
            'seeds':sorted(x for x in seen if x is not None),'reports':reports,
            'boundary':'formal claims require reviewed confirmation evidence; R3 is a separately registered stage'}
