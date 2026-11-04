"""Stage decisions require measured development evidence, never module availability."""
from architecture import ITERATIONS


def choose_g1(models,profile,match,plan):
    if profile=='smoke' or match!='tokens':return {'status':'INSUFFICIENT_TRAINING_FOR_SELECTION','selected':None}
    baseline=models.get('A0',{});eligible=[];checks={}
    for name in ('A1','A2','A3'):
        entry=models.get(name,{})
        try:
            if entry['training_status']!='COMPLETE' or baseline['training_status']!='COMPLETE':raise ValueError('incomplete language training')
            tests={'ppl':entry['validation-results.json']['ppl']/baseline['validation-results.json']['ppl']<=plan['maximum_ppl_ratio'],
                   'flops':entry['flops-report.json']['forward_flops']<=baseline['flops-report.json']['forward_flops']*(1-plan['minimum_forward_flops_reduction']),
                   'decode':entry['inference-profile.json']['batch1_decode']['median_seconds']/baseline['inference-profile.json']['batch1_decode']['median_seconds']<=plan['maximum_decode_latency_ratio']}
            checks[name]=tests
            if all(tests.values()):eligible.append(name)
        except (KeyError,TypeError,ZeroDivisionError,ValueError) as error:checks[name]={'missing_evidence':str(error)}
    all_complete=all(models.get(n,{}).get('training_status')=='COMPLETE' and 'missing_evidence' not in checks.get(n,{}) for n in ('A0','A1','A2','A3'))
    selected=min(eligible,key=lambda n:models[n]['validation-results.json']['mean_nll']) if eligible and all_complete else None
    return {'status':'G1_BASE_SELECTED' if selected else ('DEPTH_REDUCTION_NOT_SUPPORTED' if all_complete else 'INCOMPLETE_G1_EVIDENCE'),
            'selected':selected,'base_iterations':list(ITERATIONS[selected]) if selected else None,'checks':checks,
            'selection_data':'validation only; holdout not used for choosing architecture','scientific_confirmation':False}


def choose_g2(models,plan):
    try:
        base=models['B0']['mechanisms'];flat=models['B1']['mechanisms'];cell=models['B2']['mechanisms']
        if cell['M5']['status']!='PASS':raise ValueError('state continuity failed')
        gains={task:cell['tasks'][task]['accuracy']-base['tasks'][task]['accuracy'] for task in ('M1','M2','M3','M4')}
        read={task:cell['tasks'][task]['accuracy']-cell['B3_read_disabled']['tasks'][task]['accuracy'] for task in gains}
        write={task:cell['tasks'][task]['accuracy']-cell['B4_write_disabled']['tasks'][task]['accuracy'] for task in gains}
        signal=any(gains[k]>=plan['minimum_candidate_accuracy_gain'] and read[k]>0 and write[k]>0 for k in gains)
        organization={k:cell['tasks'][k]['accuracy']-flat['tasks'][k]['accuracy'] for k in gains}
        return {'status':'G2_MECHANISM_CANDIDATE_SIGNAL' if signal else 'STATE_UTILITY_NOT_SUPPORTED_AT_THIS_BUDGET',
                'g3_candidate_allowed':signal,'B2_minus_B0':gains,'B2_minus_B1':organization,'read_contribution':read,'write_contribution':write,
                'organization':'NO_CLEAR_CELLULAR_ORGANIZATION_GAIN' if max(organization.values())<plan['minimum_candidate_accuracy_gain'] else 'CELLULAR_ORGANIZATION_CANDIDATE_SIGNAL',
                'scientific_confirmation':False,'stability':'NOT_ESTABLISHED_SINGLE_SEED'}
    except (KeyError,TypeError,ValueError) as error:return {'status':'INCOMPLETE_G2_EVIDENCE','g3_candidate_allowed':False,'error':str(error)}


def choose_g3(models):
    try:
        base=models['C0'];routed=models['C2'];stateless=models['C1'];control=models['C3']
        if any(models[n].get('training_status')!='COMPLETE' or models[n].get('evaluation_status')!='COMPLETE' for n in ('C0','C1','C2','C3')):raise ValueError('incomplete G3 training/evaluation')
        if routed['flops-report.json'].get('runtime_profile',{}).get('status')!='MEASURED_TRAINING_UPDATE':raise ValueError('routed Cell training kernel evidence unavailable')
        gain=routed['flops-report.json']['forward_flops']<base['flops-report.json']['forward_flops']
        latency=routed['inference-profile.json']['batch1_decode']['median_seconds']<base['inference-profile.json']['batch1_decode']['median_seconds']
        quality=routed['validation-results.json']['ppl']/base['validation-results.json']['ppl']<=1.02
        control_quality=routed['validation-results.json']['ppl']/control['validation-results.json']['ppl']<=1.02
        task=routed['mechanisms'];tasks=('M1','M2','M3','M4')
        utility=any(task['tasks'][k]['accuracy']>stateless['mechanisms']['tasks'][k]['accuracy'] and task['tasks'][k]['accuracy']>task['read_disabled_frozen']['tasks'][k]['accuracy'] and task['tasks'][k]['accuracy']>task['write_disabled_frozen']['tasks'][k]['accuracy'] for k in tasks)
        routing=routed['state-diagnostics.json']['routing'];executed=routing['executed_main_ffn_training_rows']>0
        if not executed:status='NO_CELL_COMPUTE_CONTRIBUTION'
        elif not quality or not control_quality:status='QUALITY_REGRESSION'
        elif gain and not latency:status='FLOPS_GAIN_ONLY'
        elif not utility:status='NO_CLEAR_PERSISTENT_CELL_COMPUTE_GAIN'
        elif gain and latency:status='CELL_COMPUTE_DEVELOPMENT_CANDIDATE'
        else:status='CELL_COMPUTE_NOT_EFFICIENT_AT_THIS_BUDGET'
        return {'status':status,'flops_lower':gain,'latency_lower':latency,'quality_within_2pct':quality,
                'parameter_control_quality_within_2pct':control_quality,'state_mechanism_signal':utility,
                'parameter_control_ratio':routed['parameters']/control['parameters'],
                'parameter_control_flops_ratio':routed['flops-report.json']['forward_flops']/control['flops-report.json']['forward_flops'],
                'scientific_confirmation':False,'caution':'single-seed development evidence; no multi-seed efficiency confirmation'}
    except (KeyError,TypeError,ValueError) as error:return {'status':'INCOMPLETE_G3_EVIDENCE','error':str(error)}
