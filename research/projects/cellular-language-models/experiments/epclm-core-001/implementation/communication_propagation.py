"""CP-001: one-shot communication pulses; frozen forward-only teacher forcing."""
from contextlib import contextmanager
import hashlib
import json
import math
import platform
import signal
import subprocess
import time
import numpy as np
import torch
from frozen_model import (AssetUnavailable, BaselineIntegrityError, DiagnosticCore,
                          check_deadline, load_frozen_model, recover_frozen_assets)
from mechanism_diagnostic import write, run_documents, checked_step, loss, sync
from runtime import weight_hash
from native_clm_runtime import sha256_file

EXPERIMENT='epclm-core-001-cp001'


class IntegrityError(ValueError):
    def __init__(self,message,status='FAILED_CAUSAL_CONE'):
        super().__init__(message);self.status=status


@contextmanager
def timed_budget(seconds):
    def expired(signum,frame): raise TimeoutError('CP-001 independent time budget exhausted')
    old=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def distances(cells,receiver):
    return [min((i-receiver)%cells,(receiver-i)%cells) for i in range(cells)]


def check_cone(delta,normal_norm,receiver,event_offset,internal_step,tolerances):
    ring=torch.tensor(distances(len(delta),receiver),device=delta.device)
    radius=4*event_offset+internal_step  # step 0 is the first completed update.
    measured=delta>tolerances['state_atol']+tolerances['state_rtol']*normal_norm
    if torch.any(measured & (ring>radius)):
        raise IntegrityError('state difference outside ring causal cone')
    return measured,ring<=radius


def distribution_metrics(a,b):
    # FP64 probability reductions avoid FP32 cancellation when pulses are tiny.
    p=a.double().log_softmax(-1);q=b.double().log_softmax(-1)
    midpoint=torch.logaddexp(p,q)-math.log(2)
    js=((p.exp()*(p-midpoint)).sum()+(q.exp()*(q-midpoint)).sum())/2
    return {'prediction_js_divergence':max(0.,float(js)),
            'prediction_total_variation':float((p.exp()-q.exp()).abs().sum()/2),
            'logit_max_abs_difference':float((a-b).abs().max())}


def replay(model,initial,inputs,targets,plan,deadline,mode='NORMAL',receiver=None,normal=None):
    """B and C share this trajectory; every h is measured for cumulative windows."""
    state=initial.clone();states=[];logits_list=[];points=[];spatial=[]
    tolerance=plan['tolerances'];epsilon=tolerance['epsilon'];first_arrival={}
    for h,(token,target) in enumerate(zip(inputs,targets)):
        check_deadline(deadline)
        injected=model.inject(state,[token])  # Exactly one event injection per branch/event.
        def observe(k,current):
            check_deadline(deadline)
            if not torch.isfinite(current).all(): raise FloatingPointError('non-finite CP-001 state')
            if normal is None:
                states.append(current.clone());return
            reference=normal['states'][h*4+k]
            delta=(current-reference).norm(dim=-1)[0];norm=reference.norm(dim=-1)[0]
            measurable=delta>tolerance['state_atol']+tolerance['state_rtol']*norm
            if mode=='SHAM' and measurable.any():
                raise IntegrityError('SHAM state differs from normal','FAILED_SHAM_INTEGRITY')
            if receiver is not None:
                measurable,within=check_cone(delta,norm,receiver,h,k,tolerance)
                if h==0 and k==0 and (measurable & (torch.arange(model.config.cells,device=state.device)!=receiver)).any():
                    raise IntegrityError('single-edge pulse changed another receiver on its first update')
                for cell in torch.nonzero(measurable).flatten().cpu().tolist():
                    first_arrival.setdefault(str(cell),h*4+k+1)
                spatial.append({'event_offset':h,'internal_step':k,'completed_update':h*4+k+1,
                                'absolute_state_delta':delta.cpu().tolist(),
                                'relative_state_delta':(delta/(norm+epsilon)).cpu().tolist(),
                                'measurable_cell_count':int(measurable.sum()),
                                'within_theoretical_causal_cone':within.cpu().tolist()})
        intervention=None
        if mode=='LOCAL' and h==0:
            intervention={'type':'EDGE_DROP_ONCE','receiver':receiver,
                          'source':(receiver-1)%model.config.cells,'internal_step':0}
        communication='CROSS_CELL_REMOVED' if mode=='GLOBAL' and h==0 else True
        state=model.evolve(injected,4,communication,intervention=intervention,on_update=observe)
        logits=model.read(state)
        if not torch.isfinite(logits).all(): raise FloatingPointError('non-finite CP-001 logits')
        nll=loss(logits,target)
        if normal is None:
            logits_list.append(logits.clone());points.append({'h':h,'nll':nll})
        else:
            reference=normal['states'][h*4+3];reference_logits=normal['logits'][h]
            delta=(state-reference).norm(dim=-1)[0]
            norm=reference.norm(dim=-1)[0]
            metrics=distribution_metrics(reference_logits,logits)
            difference=nll-normal['points'][h]['nll']
            if mode=='SHAM':
                bound=tolerance['logit_atol']+tolerance['logit_rtol']*float(reference_logits.abs().max())
                if (metrics['logit_max_abs_difference']>bound or abs(difference)>tolerance['nll_atol']
                        or metrics['prediction_js_divergence']>tolerance['js_atol']
                        or metrics['prediction_total_variation']>tolerance['tv_atol']):
                    raise IntegrityError('SHAM prediction differs from normal','FAILED_SHAM_INTEGRITY')
            measured=delta>tolerance['state_atol']+tolerance['state_rtol']*norm
            points.append({'h':h,'delta_nll':difference,**metrics,
                           'state_difference_norm':float((state-reference).norm()),
                           'relative_state_difference':float((state-reference).norm()/(reference.norm()+epsilon)),
                           'cell_state_difference':delta.cpu().tolist(),'measurable_cell_count':int(measured.sum()),
                           'theoretically_reachable_cells':sum(d<=4*h+3 for d in distances(model.config.cells,receiver)) if receiver is not None else model.config.cells,
                           'state_norm_normal':float(reference.norm()),'state_norm_intervention':float(state.norm())})
    if normal is None:
        expected,expected_logits=checked_step(model,initial.clone(),inputs[0])
        if not torch.equal(expected,states[3]) or not torch.equal(expected_logits,logits_list[0]):
            raise IntegrityError('instrumented normal path differs from original step','FAILED_BASELINE_INTEGRITY')
        return {'states':states,'logits':logits_list,'points':points}
    return {'mode':mode,'receiver':receiver,'source':(receiver-1)%model.config.cells if receiver is not None else None,
            'ring_distances':distances(model.config.cells,receiver) if receiver is not None else None,
            'available_horizons':[h for h in plan['horizons'] if h<len(points)],'max_available_horizon':len(points)-1,
            'points':points,'spatial_trace':spatial,'first_arrival_completed_update':first_arrival}


@torch.inference_mode()
def preflight():
    """Only the three prescribed tiny CPU functional checks; no training."""
    from model import Config,Core
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        model=DiagnosticCore(Config(vocab=16,cells=8,dim=8)).eval()
        for p in model.parameters():p.requires_grad_(False)
        core=Core(model.config).eval();core.load_state_dict(model.state_dict())
        state=torch.randn(1,8,8);injected=model.inject(state,[2]);reference=core.evolve(injected)
        if not torch.equal(reference,model.evolve(injected,intervention=None)):
            raise IntegrityError('no-op forward mismatch','FAILED_BASELINE_INTEGRITY')
        normal=[];model.evolve(injected,on_update=lambda k,s:normal.append(s.clone()))
        altered=[];pulse={'type':'EDGE_DROP_ONCE','receiver':1,'source':0,'internal_step':0}
        model.evolve(injected,intervention=pulse,on_update=lambda k,s:altered.append(s.clone()))
        first=(normal[0]-altered[0]).norm(dim=-1)[0]
        if not (first[1]>0 and torch.count_nonzero(first)==1): raise IntegrityError('single-edge isolation gate failed')
        for k,(a,b) in enumerate(zip(normal,altered)):
            check_cone((a-b).norm(dim=-1)[0],a.norm(dim=-1)[0],1,0,k,
                       {'state_atol':1e-6,'state_rtol':1e-6})
    return {'no_op':'PASS','single_edge_isolation':'PASS','causal_cone':'PASS'}


def cluster_summary(rows,value_key,plan):
    if not rows:return {'observations':0,'stories':0,'mean':None,'story_bootstrap_95':None}
    groups={}
    for row in rows:groups.setdefault(row['story_id'],[]).append(row[value_key])
    means=np.array([np.mean(groups[k]) for k in sorted(groups)])
    draws=np.random.default_rng(plan['bootstrap_seed']).choice(means,size=(plan['bootstrap_draws'],len(means))).mean(1)
    return {'observations':len(rows),'stories':len(means),'mean':float(means.mean()),
            'mean_abs':float(np.mean([np.mean(np.abs(groups[k])) for k in sorted(groups)])),
            'positive_fraction':float(np.mean([r[value_key]>0 for r in rows])),
            'positive_story_fraction':float(np.mean(means>0)),
            'story_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),
            'estimand':'equal-weight story means; resample stories, retaining all within-story probes/edges'}


def summarize(records,plan):
    groups={'GLOBAL':[r for r in records if r['mode']=='GLOBAL'],
            'LOCAL_POOLED':[r for r in records if r['mode']=='LOCAL']}
    groups.update({'LOCAL_'+str(i):[r for r in records if r['mode']=='LOCAL' and r['receiver']==i] for i in plan['receivers']})
    temporal={};cumulative={};spatial={}
    metrics=['delta_nll','prediction_js_divergence','prediction_total_variation','state_difference_norm','relative_state_difference']
    for name,branches in groups.items():
        temporal[name]={};cumulative[name]={}
        for h in plan['horizons']:
            rows=[{'story_id':r['story_id'],**p} for r in branches for p in r['points'] if p['h']==h]
            temporal[name][str(h)]={k:cluster_summary(rows,k,plan) for k in metrics}
            temporal[name][str(h)]['available_probes']=len({(r['story_id'],r['position']) for r in branches if h<=r['max_available_horizon']})
        for window in plan['cumulative_windows']:
            values=[];future=[]
            for branch in branches:
                # Fixed-window estimates use only complete windows. Truncated availability
                # is reported at each horizon rather than silently changing H.
                if branch['max_available_horizon']<window:continue
                points=branch['points'][:window+1]
                values.append({'story_id':branch['story_id'],'delta_nll':float(np.mean([p['delta_nll'] for p in points])),
                               'abs_delta_nll':float(np.mean([abs(p['delta_nll']) for p in points]))})
                future.append({'story_id':branch['story_id'],'delta_nll':float(np.mean([p['delta_nll'] for p in points[1:]]))})
            cumulative[name][str(window)]={'inclusive_h0':cluster_summary(values,'delta_nll',plan),
                'mean_absolute_delta':cluster_summary(values,'abs_delta_nll',plan),
                'future_only':cluster_summary(future,'delta_nll',plan)}
    for receiver in plan['receivers']:
        branches=groups['LOCAL_'+str(receiver)];arrivals={}
        for cell in range(16):
            values=[r['first_arrival_completed_update'][str(cell)] for r in branches if str(cell) in r['first_arrival_completed_update']]
            arrivals[str(cell)]={'ring_distance':distances(16,receiver)[cell],'observed_probes':len(values),
                                 'earliest_completed_update':min(values) if values else None,
                                 'median_completed_update':float(np.median(values)) if values else None}
        per_update={}
        for update in range((max(plan['horizons'])+1)*4):
            traces=[r['spatial_trace'][update] for r in branches if update<len(r['spatial_trace'])]
            if traces:
                per_update[str(update+1)]={'event_offset':update//4,'internal_step':update%4,'probes':len(traces),
                    'mean_absolute_state_delta':np.mean([p['absolute_state_delta'] for p in traces],axis=0).tolist(),
                    'mean_relative_state_delta':np.mean([p['relative_state_delta'] for p in traces],axis=0).tolist(),
                    'mean_measurable_cell_count':float(np.mean([p['measurable_cell_count'] for p in traces]))}
        spatial[str(receiver)]={'source':(receiver-1)%16,'receiver':receiver,'first_arrivals':arrivals,'per_update':per_update,
                                'causal_cone':'PASS' if branches else 'NOT_EVALUATED','branches':len(branches)}
    return {'temporal':temporal,'cumulative':cumulative,'spatial':spatial}


def findings(records,stats,plan,sham):
    tolerance=plan['tolerances'];labels=[]
    local=[r for r in records if r['mode']=='LOCAL']
    if any(r['first_arrival_completed_update'].get(str(r['receiver']))==1 and any(int(cell)!=r['receiver'] for cell in r['first_arrival_completed_update']) for r in local):
        labels.append('SPATIAL_TRANSMISSION_OBSERVED')
    if any(p['h'] in plan['horizons'] and p['h']>0 and p['measurable_cell_count']>0 and p['state_difference_norm']>sham.get('max_state_difference',0.) for r in records for p in r['points']):
        labels.append('TEMPORAL_PERSISTENCE_OBSERVED')
    noise=max(tolerance['js_atol'],sham.get('max_js_divergence',0))
    affected_stories={r['story_id'] for r in records for p in r['points'] if p['h'] in plan['horizons'] and p['h']>0 and p['prediction_js_divergence']>noise}
    if len(affected_stories)>=plan['signal_rules']['future_prediction_min_stories']:
        labels.append('FUTURE_PREDICTION_EFFECT_OBSERVED')
    rule=plan['useful_delayed_signal_rule'];windows=stats['cumulative'][rule['primary_family']]
    primary=windows[str(rule['primary_window'])]['future_only']
    if (primary['stories']>=rule['minimum_stories'] and primary['mean']>=rule['minimum_mean_delta_nll']
            and primary['positive_story_fraction']>=rule['minimum_positive_story_fraction']
            and primary['story_bootstrap_95'][0]>0
            and all(windows[str(h)]['future_only']['mean'] is not None and windows[str(h)]['future_only']['mean']>0 for h in rule['require_positive_cumulative_means_for_windows'])):
        labels.append('USEFUL_DELAYED_COMMUNICATION_SIGNAL')
    return labels or ['NO_CLEAR_SIGNAL']


def plot(report,selected,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    figure,axes=plt.subplots(1,4,figsize=(15,3.6))
    for group,label in [('GLOBAL','Global pulse'),('LOCAL_POOLED','Local edges pooled')]:
        temporal=report.get('statistics',{}).get('temporal',{}).get(group,{})
        for axis,key,title in zip(axes,['state_difference_norm','prediction_js_divergence','delta_nll','delta_nll'],['State difference','Prediction JS','Signed ΔNLL','Absolute ΔNLL']):
            xs=[h for h in report['identity']['protocol']['horizons'] if temporal.get(str(h),{}).get(key,{}).get('mean') is not None]
            axis.plot(xs,[temporal[str(h)][key]['mean_abs' if title=='Absolute ΔNLL' else 'mean'] for h in xs],marker='o',label=label)
            axis.set(title=title,xlabel='Future event offset')
    axes[2].axhline(0,color='gray',linewidth=.5);axes[0].legend(fontsize=8)
    figure.suptitle(report['status']);figure.tight_layout();figure.savefig(directory/'temporal-response.png');plt.close(figure)
    figure,axes=plt.subplots(1,4,figsize=(14,4),squeeze=False)
    for axis,receiver in zip(axes[0],report['identity']['protocol']['receivers']):
        record=next((r for r in selected if r['receiver']==receiver),None)
        if record:
            image=np.array([r['relative_state_delta'] for r in record['spatial_trace']])
            mesh=axis.imshow(image,aspect='auto',origin='upper',cmap='magma');figure.colorbar(mesh,ax=axis,fraction=.045)
            for boundary in range(3,len(image),4):axis.axhline(boundary+.5,color='white',alpha=.3,linewidth=.5)
            ticks=list(range(0,len(image),8));axis.set_yticks(ticks);axis.set_yticklabels([i+1 for i in ticks])
        else:axis.text(.5,.5,'NOT COMPLETED',ha='center',transform=axis.transAxes)
        axis.set(title=f'{(receiver-1)%16} → {receiver}',xlabel='Cell ID',ylabel='Completed internal update')
    figure.suptitle('Fixed first-story 25% probe; relative state difference (not semantic attribution)')
    figure.tight_layout();figure.savefig(directory/'propagation-heatmap.png');plt.close(figure)


def publish(root,output,report,selected):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    public={k:v for k,v in report.items() if k!='completed'}
    publication=root/'results'/EXPERIMENT/'publish';publication.mkdir(parents=True,exist_ok=True)
    write(publication/'decision.json',public)
    stats=report.get('statistics',{})
    for name,value in [('global-pulse',stats.get('temporal',{}).get('GLOBAL',{})),
                       ('local-edge-pulse',{k:v for k,v in stats.get('temporal',{}).items() if k.startswith('LOCAL')}),
                       ('temporal-impact',stats.get('cumulative',{})),('spatial-propagation',{'by_receiver':stats.get('spatial',{}),'fixed_heatmap_data':selected})]:
        write(publication/(name+'.json'),value)
    (publication/'summary.md').write_text('# Communication Propagation-001\n\n'+report['status']+'\n\n'+
        ', '.join(report.get('findings',[]))+'\n\nFrozen C1; new training tokens 0; optimizer updates 0. H1–H5 NOT_EVALUATED.\n\n'+
        'Exploratory single-seed, story-clustered analysis; no multiple-testing confirmation. Spatial propagation, future prediction sensitivity and useful delayed prediction effects are separate claims. Readout aggregates all cells, so early output changes do not establish spatial spread.\n'+
        ('\nStop reason: '+report['error']+'\n' if report.get('error') else ''))
    plot(report,selected,publication)
    reports=output/'reports';reports.mkdir(exist_ok=True)
    import shutil
    for name in EXPERIMENTS[EXPERIMENT].files:shutil.copyfile(publication/name,reports/name)
    destination=prepare_artifacts(root,EXPERIMENT,EXPERIMENTS[EXPERIMENT],None)
    push_results(root,destination,EXPERIMENT,'kaggle/epclm-core-001-cp001-'+report['identity']['diagnostic_source'][:12],'GITHUB_TOKEN','origin/main')
    print(json.dumps({'mode':'FROZEN_COMMUNICATION_PROPAGATION','status':report['status'],'report':str(output/'decision.json')}))


def read_records(output,report):
    records=[]
    from frozen_model import safe_path
    for item in report.get('completed',{}).values():
        path=output/safe_path(item['path'])
        if sha256_file(path)!=item['sha256']:raise IntegrityError('saved branch hash mismatch','FAILED_BASELINE_INTEGRITY')
        record=json.loads(path.read_text())
        if record['identity']!=report['identity']:raise IntegrityError('saved branch identity mismatch','FAILED_BASELINE_INTEGRITY')
        records.append(record)
    return records


def selected_records(records,plan):
    select=plan['heatmap_selection']
    # Fixed at design time, including on partial runs: never substitute a striking probe.
    return [{k:v for k,v in r.items() if k in ('story_id','position','receiver','source','ring_distances','spatial_trace')}
            for r in records if r['mode']=='LOCAL' and r['story_index']==select['story_index']
            and r['probe_fraction']==select['probe_fraction'] and r['receiver'] in select['receivers']]


@torch.inference_mode()
@torch.autocast(device_type='cuda',enabled=False)
def launch(root,base,output,device='cuda'):
    from scripts.research.publish_experiment_results import validate_origin,load_github_token
    from storage import bundle
    if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('enable Kaggle GPU')
    validate_origin(root);load_github_token('GITHUB_TOKEN')
    output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((base/'communication-propagation-protocol.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    protocol_sha=sha256_file(base/'communication-propagation-protocol.json')
    identity={'experiment':EXPERIMENT,'mode':'FROZEN_COMMUNICATION_PROPAGATION','diagnostic_source':source,
              'protocol_sha256':protocol_sha,'protocol':plan,'training_source':plan['training_identity']['source'],
              'training_weight_sha256':plan['checkpoint_identity']['weights_sha256'],
              'data_manifest_sha256':plan['training_identity']['data'],'training_tokens':250000,
              'new_training_tokens':0,'optimizer_updates':0,'formal_confirmation':False}
    report={'format':'minicells.epclm-core-001.cp001.v1','status':'INCOMPLETE','identity':identity,'completed':{},
            'hypotheses':dict.fromkeys(['H1','H2','H3','H4','H5'],'NOT_EVALUATED'),
            'exploratory_hypotheses':dict.fromkeys(['H-CP1','H-CP2','H-CP3','H-CP4'],'NOT_EVALUATED'),
            'findings':[],'timings':{},'sham':{'probes':0,'replays':0,'checked_probe_indices':[],'max_js_divergence':0.,'max_state_difference':0.,'max_abs_delta_nll':0.,'max_total_variation':0.,'max_logit_difference':0.},
            'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
                           'precision':'fp32','autocast_enabled':False,'grad_scaler_created':False,'inference_mode':True},
            'limitations':['single frozen training seed; same 16 development stories; exploratory multiple horizons/edges',
                           'readout pools all cells; output sensitivity alone does not prove spatial propagation',
                           'useful delayed signal rule chosen after prior diagnostic, before CP-001 results; no confirmation']}
    path=output/'decision.json';recovery_identity={'source':source,'protocol':protocol_sha,'data':identity['data_manifest_sha256']}
    def load_progress():
        nonlocal report
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['identity']!=identity:raise BaselineIntegrityError('CP-001 progress identity mismatch')
            report=saved;read_records(output,report)
    load_progress()
    if report['status']!='INCOMPLETE':
        publish(root,output,report,selected_records(read_records(output,report),plan));return
    model=None;before=None;stage='recovery';started=time.monotonic();execution_deadline=started+plan['budgets']['execution_seconds']
    def persist():
        began_save=time.monotonic();write(path,report);bundle(output,recovery_identity)
        report['timings']['progress_save_seconds']=report['timings'].get('progress_save_seconds',0.)+time.monotonic()-began_save
    try:
        report['functional_preflight']=preflight()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assets=output.with_name(output.name+'-frozen-assets');began=time.monotonic()
        try:
            with timed_budget(plan['budgets']['recovery_seconds']):
                restored=recover_frozen_assets(root,assets,plan['training_identity'],plan['checkpoint_identity'],
                    min(execution_deadline,time.monotonic()+plan['budgets']['recovery_seconds']),plan['fallback_saved_versions'],
                    progress_output=output,progress_identity=recovery_identity,preferred_versions=plan['preferred_asset_versions'])
                load_progress()
                if restored!='DIAGNOSTIC_TERMINAL':
                    model,_,documents=load_frozen_model(plan['training_identity'],plan['checkpoint_identity'],assets,base/'protocol.json',device)
        finally:report['timings']['asset_recovery_seconds']=time.monotonic()-began
        if restored!='DIAGNOSTIC_TERMINAL':
            before=weight_hash(model);report['weights_before']=before
            report['asset_provenance']=json.loads((assets/'asset-provenance.json').read_text())
            documents=documents[:plan['documents']]
            if len(documents)!=plan['documents'] or sum(len(d['tokens'])-1 for d in documents)!=plan['expected_prediction_tokens']:
                raise BaselineIntegrityError('original 16-document/3322-target identity differs')
            report['data_assets']={n:sha256_file(assets/'assets'/n) for n in ('manifest.json','tokenizer.json','development.json')}
            stage='diagnostic';began=time.monotonic();deadline=min(execution_deadline,began+plan['budgets']['diagnostic_seconds'])
            try:
                with timed_budget(min(plan['budgets']['diagnostic_seconds'],execution_deadline-time.monotonic())):
                    probes=[];sync(model);baseline_start=time.monotonic()
                    baseline=run_documents(model,documents,deadline,probes=probes,fractions=plan['probe_fractions'])
                    report['timings']['baseline_forward_seconds']=time.monotonic()-baseline_start
                    if abs(baseline['nll']-plan['expected_baseline_nll'])>plan['baseline_atol']:
                        raise BaselineIntegrityError('frozen baseline NLL mismatch')
                    report['baseline']={k:v for k,v in baseline.items() if k!='stories'}
                    persist()
                    by_id={d['id']:(i,d) for i,d in enumerate(documents)}
                    for index,probe in enumerate(probes):
                        check_deadline(deadline)
                        modes=[('GLOBAL',None)]+[('LOCAL',i) for i in plan['receivers']]
                        pending=[(mode,receiver) for mode,receiver in modes if f'{index:03d}-{mode}-{receiver}' not in report['completed']]
                        if not pending:continue
                        story_index,doc=by_id[probe['story_id']];position=probe['position']
                        length=min(max(plan['horizons'])+1,len(doc['tokens'])-1-position)
                        inputs=doc['tokens'][position:position+length];targets=doc['tokens'][position+1:position+length+1]
                        fraction=next(f for f in plan['probe_fractions'] if int((len(doc['tokens'])-1)*f)==position)
                        sync(model);began_probe=time.monotonic()
                        normal=replay(model,probe['state'],inputs,targets,plan,deadline)
                        sham=replay(model,probe['state'].clone(),inputs,targets,plan,deadline,'SHAM',normal=normal)
                        report['sham']['replays']+=1
                        report['sham']['checked_probe_indices']=sorted(set(report['sham']['checked_probe_indices']+[index]))
                        report['sham']['probes']=len(report['sham']['checked_probe_indices'])
                        report['sham']['max_js_divergence']=max(report['sham']['max_js_divergence'],max(p['prediction_js_divergence'] for p in sham['points']))
                        report['sham']['max_state_difference']=max(report['sham']['max_state_difference'],max(p['state_difference_norm'] for p in sham['points']))
                        report['sham']['max_abs_delta_nll']=max(report['sham']['max_abs_delta_nll'],max(abs(p['delta_nll']) for p in sham['points']))
                        report['sham']['max_total_variation']=max(report['sham']['max_total_variation'],max(p['prediction_total_variation'] for p in sham['points']))
                        report['sham']['max_logit_difference']=max(report['sham']['max_logit_difference'],max(p['logit_max_abs_difference'] for p in sham['points']))
                        report['sham']['status']='PASS'
                        sync(model)
                        report['timings']['normal_sham_forward_seconds']=report['timings'].get('normal_sham_forward_seconds',0.)+time.monotonic()-began_probe
                        for mode,receiver in pending:
                            check_deadline(deadline);sync(model);began_branch=time.monotonic()
                            result=replay(model,probe['state'].clone(),inputs,targets,plan,deadline,mode,receiver,normal)
                            sync(model);elapsed=time.monotonic()-began_branch
                            report['timings']['branch_forward_seconds']=report['timings'].get('branch_forward_seconds',0.)+elapsed
                            result.update(identity=identity,story_id=doc['id'],story_index=story_index,position=position,
                                          probe_fraction=fraction,inference_seconds=elapsed,
                                          teacher_forced_input_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest())
                            key=f'{index:03d}-{mode}-{receiver}';relative='private/'+key+'.json'
                            write(output/relative,result)
                            report['completed'][key]={'path':relative,'sha256':sha256_file(output/relative)}
                            persist()
                        sync(model)
                        report['timings'].setdefault('probe_elapsed_seconds_including_checkpoint_io',0.)
                        report['timings']['probe_elapsed_seconds_including_checkpoint_io']+=time.monotonic()-began_probe
                    records=read_records(output,report)
                    if len(records)!=len(probes)*5:raise IntegrityError('missing pulse branches','INCOMPLETE')
                    report['statistics']=summarize(records,plan)
                    report['findings']=findings(records,report['statistics'],plan,report['sham'])
                    report['completed_counts']={'probes':len(probes),'global':sum(r['mode']=='GLOBAL' for r in records),
                                                'local':sum(r['mode']=='LOCAL' for r in records),'new_training_tokens':0}
                    report['status']='COMPLETE';report.pop('error',None)
                    report['exploratory_hypotheses']={h:('EXPLORATORY_SIGNAL_OBSERVED' if label in report['findings'] else 'NO_CLEAR_SIGNAL') for h,label in zip(['H-CP1','H-CP2','H-CP3','H-CP4'],['SPATIAL_TRANSMISSION_OBSERVED','TEMPORAL_PERSISTENCE_OBSERVED','FUTURE_PREDICTION_EFFECT_OBSERVED','USEFUL_DELAYED_COMMUNICATION_SIGNAL'])}
            finally:report['timings']['diagnostic_seconds']=time.monotonic()-began
    except AssetUnavailable as error:report.update(status='BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error))
    except BaselineIntegrityError as error:report.update(status='FAILED_BASELINE_INTEGRITY',error=str(error))
    except IntegrityError as error:report.update(status=error.status,error=str(error))
    except TimeoutError as error:report.update(status='BLOCKED_MODEL_ASSET_UNAVAILABLE' if stage=='recovery' else 'INCOMPLETE',error=str(error))
    except FloatingPointError as error:report.update(status='FAILED_NUMERICAL_CHECK',error=str(error))
    except (RuntimeError,OSError,ValueError,KeyError) as error:
        report.update(status='BLOCKED_MODEL_ASSET_UNAVAILABLE' if stage=='recovery' else 'FAILED_NUMERICAL_CHECK',error=str(error))
    finally:
        # If recovery failed after copying progress, retain its verified completed branches.
        if model is None and path.exists():
            saved=json.loads(path.read_text())
            if saved.get('identity')==identity:report['completed']=saved.get('completed',report['completed'])
        if model is not None and before is not None:
            try:
                report['weights_after']=weight_hash(model)
                if report['weights_after']!=before:report.update(status='FAILED_BASELINE_INTEGRITY',error='frozen weights changed')
            except RuntimeError as error:report.update(status='FAILED_NUMERICAL_CHECK',error=str(error))
        report['timings']['execution_seconds']=time.monotonic()-started
        persist()
    records=read_records(output,report)
    if report['status']!='COMPLETE':
        report['exploratory_hypotheses']=dict.fromkeys(['H-CP1','H-CP2','H-CP3','H-CP4'],'NOT_EVALUATED')
        report['findings']=[]
    if records and report['status']!='COMPLETE':
        report['statistics']=summarize(records,plan)
        report['findings']=[]
        report['completed_counts']={'global':sum(r['mode']=='GLOBAL' for r in records),'local':sum(r['mode']=='LOCAL' for r in records),'new_training_tokens':0}
        persist()
    publish(root,output,report,selected_records(records,plan))
