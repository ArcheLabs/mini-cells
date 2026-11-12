"""Single P3 entry with a common measured budget and all-arm curriculum gate."""
import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from p3_data import BASE,prepare,write,digest,file_hash,construct,ROOT,StageSampler
from p3_metrics import stage1_gate
from b0_audit import verify_dependencies
from b0_models import state_hash,parameter_report
from b0_integrity import environment,publisher
from e0 import load_checkpoint
sys.path.insert(0,str(BASE.parent/'b0-p2'))
from p2_integrity import download,freeze_bundle


def charged(out,action,arm,protocol,data,folder):
    #Reuse the frozen P2 clock/kill implementation in a private module instance.
    spec=importlib.util.spec_from_file_location('_p3_private_gpu_budget',BASE.parent/'b0-p2/p2_run.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.BASE=BASE;return module.charged_phase(out,action,arm,protocol,data,folder)


def plan(bench,spent,p):
    forecasts={};details={};factor=p['benchmark']['safety_factor']
    for updates in p['stage2_choices']:
        total=spent;costs={}
        for arm in p['models_order']:
            cost=60.
            for stage,u in [(1,128),(2,updates)]:
                b=bench[arm][str(stage)];step=max(x['total'] for x in b['steps']);checks=5
                #Scheduled full Validation plus independent final checkpoint replay.
                cost+=factor*(u*step+(checks+1)*b['validation_seconds']+(u//64+checks+2)*b['checkpoint_seconds'])
            costs[arm]=cost;total+=cost
        forecasts[updates]=total;details[updates]=costs
    chosen=next((u for u in p['stage2_choices'] if forecasts[u]<=6480),None)
    return {'status':'PASS' if chosen else 'BLOCKED_RESOURCE_LIMIT','stage2_updates':chosen,'forecasts_seconds':forecasts,
        'per_arm_forecast':details,'charged_benchmark_seconds':spent,'planning_limit':6480,'hard_limit':7200,
        'query_isolated':'optional after mandatory training;all-arm groups only;reserve independent verification'}


def publish(folder):
    import requests
    spec=importlib.util.spec_from_file_location('_p3_private_publisher',BASE.parent/'b0/src/b0_integrity.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);module.freeze_bundle=freeze_bundle
    m=json.loads((folder/'run-manifest.json').read_text());api='https://api.github.com/repos/ArcheLabs/mini-cells/releases'
    headers={'Authorization':'Bearer '+publisher().load_github_token('GITHUB_TOKEN')}
    response=requests.get(api+'/tags/'+m['run_id'],headers=headers,timeout=(15,60))
    if response.status_code==404:
        response=requests.post(api,headers=headers,json={'tag_name':m['run_id'],'target_commitish':m['source_commit'],
            'name':'B0-P3 '+m['arm'],'prerelease':True,'body':'Episode-local associative recall calibration. Final independent verification is in the companion result index. No Test access,12M or M6.'},timeout=(15,60))
    response.raise_for_status();return module.publish_bundle(folder)


def metadata(manifest,artifact,proof,report):
    pub=publisher();schema='minicells.b0-p3.index.v1'
    with tempfile.TemporaryDirectory(prefix='b0-p3-index-') as temp:
        root=Path(temp)/'repo';subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',manifest['source_commit']],cwd=root,check=True)
        stage=root/'results/b0-p3';stage.mkdir(parents=True)
        write(stage/'decision.json',{'format':schema,'manifest':manifest,'artifact':artifact,'verification':proof,'test_status':'SEALED_NOT_ACCESSED','hypotheses':'NOT_EVALUATED'})
        (stage/'REPORT.md').write_text(report);branch='kaggle/'+manifest['run_id']
        spec=pub.ExperimentSpec(str(stage.relative_to(root)),'artifacts/experiments/epclm004-m5-sv-b0-p3/'+manifest['run_id'],('decision.json','REPORT.md'),branch,schema)
        result=pub.prepare_artifacts(root,'epclm004-m5-sv-b0-p3',spec,None)
        pub.push_results(root,result,'epclm004-m5-sv-b0-p3',branch,'GITHUB_TOKEN','origin/main')
        return {'status':'REMOTE_SHA_VERIFIED','branch':branch,'sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}


def verify_payload(folder,banks,p,identity,stages):
    from p3_data import validate
    folder=Path(folder);last=None
    if json.loads((folder/'dataset-manifest.json').read_text())['dataset_sha256']!=identity['dataset_sha256']:raise ValueError('Dataset manifest differs')
    for stage in stages:
        level=str(p['stages'][str(stage)]['level']);rows=banks[level]['train'];u=p['stages'][str(stage)]['updates']
        saved=load_checkpoint(folder/('checkpoints/stage'+str(stage)+'-final.pt'))
        if saved['identity']!=identity or saved['stage_step']!=u or saved['stage']!=stage:raise ValueError('Checkpoint identity/stage/step differs')
        sampler=StageSampler(rows,p['master_seed'],int(level));expected=[]
        for _ in range(u):
            selected=sampler.propose();expected.append([r['id'] for r in selected]);sampler.resolve(True,{'committed_updates':0},{})
        history=[r for r in saved['history'] if r['stage']==stage]
        if [r['batch_episode_ids'] for r in history]!=expected:raise ValueError('Common data order/budget differs')
        if len(saved['history'])!=saved['global_step'] or saved['global_step']!=(p['stages']['1']['updates']+u if stage==2 else u):raise ValueError('Global step discontinuity')
        if any(not np.isfinite(r['active_nll']) or not np.isfinite(r['gradient_norm']) or r['parameter_change_norm']<=0 for r in history):raise ValueError('Invalid real update history')
        #Check actual archived data, not just the generator's recorded status.
        for name in ('train','validation'):
            archived=json.loads((folder/'datasets'/(level+'-'+name+'.json')).read_text())
            if digest(archived)!=digest(banks[level][name]):raise ValueError('Archived bank differs')
            for row in archived:validate(row)
        result=json.loads((folder/('stage'+str(stage)+'-result.json')).read_text());local=result['stage_ledger']
        for key,value in {'episodes':u*8,'input_tokens':u*8*p['stages'][str(stage)]['length'],
            'valid_input_tokens':u*8*p['stages'][str(stage)]['valid_length'],'supervised_target_tokens':u*8*int(level),'query_target_tokens':u*8*int(level),'committed_updates':u}.items():
            if local[key]!=value:raise ValueError('Stage token/update ledger differs:'+key)
        for probability in (folder/'training'/('stage'+str(stage))/'evaluation').glob('*/probabilities.npz'):
            metrics=json.loads((probability.parent/'metrics.json').read_text());rows=banks[level]['validation']
            with np.load(probability) as a:
                expected_targets=np.array([t for row in rows for t in [row['labels'][pos] for pos in row['query_positions']]])
                if not np.array_equal(expected_targets,a['targets']):raise ValueError('Saved Query labels differ')
                expected_maps=np.array([row['mapping_hash'] for row in rows for _ in row['query_positions']])
                if not np.array_equal(expected_maps,a['mapping_hashes']):raise ValueError('Saved Mapping order differs')
                c=a['candidate_ids'];lp=a['candidate_log_probabilities'];valid=c>=0
                expected_candidates=np.array([row['candidates']+[-1]*(c.shape[1]-len(row['candidates'])) for row in rows for _ in row['query_positions']])
                if not np.array_equal(c,expected_candidates):raise ValueError('Saved candidate set differs from current facts')
                value=a['value_log_probabilities']
                if not np.isfinite(value).all() or not np.allclose(np.exp(value).sum(-1),1,atol=2e-6):raise ValueError('Invalid value probabilities')
                random_accuracy=float(np.mean([1/len(row['candidates']) for row in rows]));random_nll=float(np.mean([np.log(len(row['candidates'])) for row in rows]))
                if abs(metrics['random_active_accuracy']-random_accuracy)>1e-12 or abs(metrics['random_active_nll']-random_nll)>1e-12:raise ValueError('Reported random baseline differs')
                if not np.isfinite(lp[valid]).all() or not np.allclose(np.where(valid,np.exp(lp),0).sum(-1),1,atol=2e-6):raise ValueError('Invalid candidate probabilities')
                gold=np.array([np.flatnonzero(c[i]==t)[0] for i,t in enumerate(a['targets'])]);nll=float(-lp[np.arange(len(gold)),gold].mean(dtype=float))
                if abs(nll-metrics['active_nll'])>2e-5:raise ValueError('Saved candidate NLL differs')
                accuracy=float((a['active_predictions']==a['targets']).mean())
                if accuracy!=metrics['active_exact_match']:raise ValueError('Saved accuracy differs')
                if 'full_log_probabilities' in a:
                    if not np.isfinite(a['full_log_probabilities']).all() or not np.allclose(np.exp(a['full_log_probabilities']).sum(-1),1,atol=2e-6):raise ValueError('Invalid full probabilities')
                    full=float(-a['full_log_probabilities'][np.arange(len(gold)),a['targets']].mean(dtype=float))
                    if abs(full-metrics['full_nll'])>2e-5:raise ValueError('Saved full NLL differs')
        last=saved
    return {'status':'PASS','stages':stages,'global_steps':last['global_step'],'test_comparison':'NOT_CHECKED_TEST_SEALED'}


def model_report(arm,results,gate,isolated,proof):
    lines=['# B0-P3 '+arm,'','Stage1 gate: '+gate['status']+'. Stage2 started: '+str(2 in results)+'. Single-seed development;no H1–H6 confirmation.','',
        '| Stage | Active EM | Active NLL | Actual random EM | Actual random NLL | Value EM | Full EM |','|---|---:|---:|---:|---:|---:|---:|']
    for stage,r in results.items():
        v=r['validation'];lines.append(f"| {stage} | {v['active_exact_match']:.4f} | {v['active_nll']:.6f} | {v['random_active_accuracy']:.4f} | {v['random_active_nll']:.6f} | {v['value_exact_match']:.4f} | {v['full_exact_match']:.4f} |")
    lines+=['','## Recorded training calls','',
        '| Stage | Updates | Logical slots | Valid processed tokens | Targets | Fit wall this call, s | Peak allocated MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stage,r in results.items():
        ledger=r['stage_ledger'];resource=r['resources'];peak=resource['training_peak_allocated_bytes']
        lines.append(f"| {stage} | {ledger['committed_updates']} | {ledger['input_tokens']} | {ledger['valid_input_tokens']} | {ledger['supervised_target_tokens']} | {resource['fit_wall_seconds_this_call']:.2f} | {peak/2**20 if peak is not None else 'N/A'} |")
    lines+=['','Fit time/peak cover the completing call; cumulative subprocess charges include interrupted calls and are the time-budget authority.',
        'An individual passing arm does not enter Stage2 when another arm blocks the cohort.','', 'Gate checks: '+json.dumps(gate),'','Query-isolated diagnostic: '+json.dumps(isolated),'',
        'Primary evaluation teacher-forces earlier gold answers; current targets are strictly causal next tokens. Query-isolated evaluation changes answer history, position and chunk/Store availability, so its difference does not isolate a causal answer-history effect.',
        'Only active-candidate loss is optimized. Full/value output collapse is diagnostic, not a Stage1 veto. Actual per-Episode candidate random baselines are used. Empirical cluster intervals can collapse at floor/ceiling and do not include across-seed variance.',
        'All valid59/115 tokens are retained; only trailing5/13 PAD slots are omitted. Logical slot counts64/128 are distinct from processed token counts. E7 stores raw token representations; selected fact-position co-hits do not prove semantic binding retrieval or causal benefit.',
        'Test is SEALED_NOT_ACCESSED;Train/Test mapping intersection is not inspected. No12M,32–64-pair scaling,main multi-seed training or M6.','',
        'Independent verification: '+json.dumps(proof)]
    return '\n'.join(lines)+'\n'


def launch(args):
    started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=True);verify_dependencies()
    p=json.loads((BASE/'configs/protocol.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    namespace={'experiment':'B0-P3','source_commit':source,'template_sha256':digest(p)};stamp=out/'namespace.json'
    if stamp.exists() and json.loads(stamp.read_text())!=namespace:raise ValueError('Output namespace differs')
    if not stamp.exists() and (out/'decision.json').exists():raise ValueError('Unrecognized old experiment Output')
    write(stamp,namespace);banks,data=prepare(out/'data',p);write(out/'protocol-template.json',p)
    initial={};params={}
    for arm in p['models_order']:
        model=construct(arm,p);initial[arm]=state_hash(model);params[arm]=parameter_report(model)
        expected=10130688 if arm=='E0' else 10021120
        if params[arm]['static_parameters']!=expected:raise ValueError('Frozen parameter count differs')
        del model
    if initial['E6']!=initial['E7-Innovation']:raise ValueError('Shared initial weights differ')
    write(out/'initialization-preflight.json',{'status':'PASS','hashes':initial,'parameters':params,'historical_weights_loaded':False})
    if not args.execute:write(out/'decision.json',{'status':'CPU_PREFLIGHT_PASS_GPU_NOT_STARTED','test_status':p['test_status']});return
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('Commit modified source before execution')
    tracked=subprocess.run(['git','cat-file','-e','HEAD:'+str((BASE/'p3_run.py').relative_to(ROOT))],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if tracked.returncode:raise ValueError('Commit P3 source before execution')
    env=environment()
    if not env['gpu'] or 'T4' not in env['gpu']:raise RuntimeError('BLOCKED_HARDWARE: actual Tesla T4 required')
    experiment_identity={**namespace,'dataset_sha256':data['dataset_sha256'],'environment_sha256':digest({k:env[k] for k in ('python','torch','cuda','gpu','driver','precision','tf32','amp')})}
    header=out/'execution-identity.json'
    if not header.exists() and Path('/kaggle/input').exists():
        options=[]
        for f in Path('/kaggle/input').rglob('execution-identity.json'):
            if json.loads(f.read_text())==experiment_identity:
                budget=f.parent/'gpu-budget.json';options.append((json.loads(budget.read_text())['charged_seconds'] if budget.exists() else 0,str(f.parent)))
        if options:shutil.copytree(Path(max(options)[1]),out,dirs_exist_ok=True)
    if header.exists() and json.loads(header.read_text())!=experiment_identity:raise ValueError('P3 resume identity differs')
    write(header,experiment_identity);bench={};template=out/'protocol-template.json'
    for arm in p['models_order']:
        bench[arm]={}
        for stage in (1,2):
            folder=out/'benchmark'/arm/str(stage)
            if not (folder/'benchmark.json').exists():charged(out,'benchmark'+str(stage),arm,template,out/'data',folder)
            bench[arm][str(stage)]=json.loads((folder/'benchmark.json').read_text())
    final_protocol=out/'protocol.json'
    if final_protocol.exists():
        p=json.loads(final_protocol.read_text())
        if not (out/'budget-plan.json').exists():raise ValueError('Missing frozen budget plan')
        forecast=json.loads((out/'budget-plan.json').read_text())
    else:
        budget=json.loads((out/'gpu-budget.json').read_text());forecast=plan(bench,budget['charged_seconds'],p);write(out/'budget-plan.json',forecast)
        if forecast['status']!='PASS':raise RuntimeError('BLOCKED_RESOURCE_LIMIT: minimum Stage2 budget cannot fit; no formal training')
        u=forecast['stage2_updates'];p['stages']['2'].update(updates=u,epochs=u//64,validation_updates=[0,u//4,u//2,3*u//4,u]);p['training']['updates']=128+u
        write(final_protocol,p)
    if p['stages']['1']['updates']!=128 or p['stages']['2']['updates'] not in p['stage2_choices']:raise ValueError('Invalid frozen budgets')
    configuration={'models':p['models'],'memory':p['memory'],'predictor':p['predictor'],'training':p['training'],'stages':p['stages']}
    identities={};results={a:{} for a in p['models_order']}
    for arm in p['models_order']:
        folder=out/arm;folder.mkdir(exist_ok=True);identity={**experiment_identity,'phase':'b0-p3','arm':arm,'model_seed':p['seeds']['model'],
            'protocol_sha256':digest(p),'config_sha256':digest(configuration),'initialization_sha256':initial[arm],'logical_batch_size':8,'historical_weights_loaded':False,
            'planned_stage_updates':{stage:cfg['updates'] for stage,cfg in p['stages'].items()},
            'planned_logical_input_slots_per_arm':sum(cfg['updates']*8*cfg['length'] for cfg in p['stages'].values()),
            'planned_supervised_targets_per_arm':sum(cfg['updates']*8*cfg['level'] for cfg in p['stages'].values())}
        identity['run_id']='b0-p3-'+arm.lower()+'-'+digest(identity)[:20];identities[arm]=identity
        m=folder/'run-manifest.json'
        if m.exists() and json.loads(m.read_text())!=identity:raise ValueError('Arm identity differs')
        write(m,identity)
        if not (folder/'stage1-result.json').exists():
            import requests
            response=requests.get('https://api.github.com/repos/ArcheLabs/mini-cells/releases/tags/'+identity['run_id'],timeout=(15,60))
            if response.status_code==200:raise RuntimeError('MISSING_COMPLETED_OUTPUT: attach published P3 Output;no retraining')
            if response.status_code!=404:response.raise_for_status()
            write(folder/'protocol.json',p);write(folder/'config.json',configuration);write(folder/'dataset-manifest.json',data)
            folder_data=folder/'datasets';folder_data.mkdir(exist_ok=True)
            for f in (out/'data').glob('*.json'):shutil.copy2(f,folder_data/f.name)
            charged(out,'stage1',arm,final_protocol,out/'data',folder)
        results[arm][1]=json.loads((folder/'stage1-result.json').read_text())
    gates={}
    for arm,r in results.items():
        verify_payload(out/arm,banks,p,identities[arm],[1])
        gates[arm]=stage1_gate(r[1]['validation'],r[1]['online'],p)
        gates[arm]['checkpoint_sha256']=file_hash(out/arm/'checkpoints/stage1-final.pt')
    gate={'status':'PASS' if all(g['status']=='PASS' for g in gates.values()) else 'BLOCKED','arms':gates,'protocol_sha256':digest(p)}
    write(out/'stage1-gate.json',gate);stages=[1]
    if gate['status']=='PASS':
        for arm in p['models_order']:
            if not (out/arm/'stage2-result.json').exists():charged(out,'stage2',arm,final_protocol,out/'data',out/arm)
            results[arm][2]=json.loads((out/arm/'stage2-result.json').read_text())
        stages.append(2)
    #Optional diagnostics run for all three arms per stage, after reserving mandatory restore time.
    isolation={a:{} for a in p['models_order']}
    reserve=sum(p['benchmark']['safety_factor']*bench[a][str(s)]['validation_seconds']+15 for a in p['models_order'] for s in stages)
    optional_path=out/'query-isolated-plan.json'
    optional=json.loads(optional_path.read_text()) if optional_path.exists() else {}
    for stage in stages:
        spent=json.loads((out/'gpu-budget.json').read_text())['charged_seconds'];cost=sum(p['benchmark']['safety_factor']*bench[a][str(stage)]['validation_seconds']+15 for a in p['models_order'])
        if str(stage) not in optional:
            optional[str(stage)]=spent+reserve+cost<=6480;write(optional_path,optional)
        permitted=optional[str(stage)]
        for arm in p['models_order']:
            target=out/arm/'query-isolated'/str(stage)/'metrics.json'
            if target.exists():isolation[arm][str(stage)]={'status':'COMPLETED'};continue
            if permitted:
                charged(out,'isolated'+str(stage),arm,final_protocol,out/'data',out/arm);isolation[arm][str(stage)]={'status':'COMPLETED'}
            else:isolation[arm][str(stage)]={'status':'NOT_RUN_BUDGET'}
        for arm in p['models_order']:write(out/arm/'query-isolated-status.json',isolation[arm])
    returned={}
    for arm in p['models_order']:
        folder=out/arm;verify_payload(folder,banks,p,identities[arm],stages)
        if not (folder/'publication.json').exists():
            checkpoint=load_checkpoint(folder/('checkpoints/stage'+str(stages[-1])+'-final.pt'))
            write(folder/'metrics/training.json',checkpoint['history'])
            validation={};distribution={};decomposition={};states={};store={}
            for stage in stages:
                validation[str(stage)]={}
                for path in sorted((folder/'training'/('stage'+str(stage))/'evaluation').iterdir(),key=lambda f:int(f.name)):
                    value=json.loads((path/'metrics.json').read_text());metric_key=str(stage)+':'+path.name
                    validation[str(stage)][path.name]=value
                    distribution[metric_key]={name:value[name] for name in ('prediction_distribution','dominant_token_rate','unique_predicted_tokens','collapse_episode_fraction')}
                    decomposition[metric_key]={name:value[name] for name in ('full_nll','type_nll','candidate_nll','active_nll','value_nll','random_active_nll')}
                    memory=json.loads((path/'memory-diagnostics.json').read_text())
                    state_norms=[v for row in memory for v in row.get('state_norms',[])]
                    states[metric_key]={'raw_records':str((path/'memory-diagnostics.json').relative_to(folder)),
                        'episode_stream_resets':sum(row.get('episode_reset_count',0) for row in memory),
                        'state_norm_mean':float(np.mean(state_norms)) if state_norms else None,
                        'state_update_norm':'NOT_AVAILABLE','saturation':'NOT_AVAILABLE: no calibrated threshold'}
                    actual=[row['store'] for row in memory if 'write_attempts' in row.get('store',{})]
                    reads=[read for row in actual for read in row['retrieval_records']]
                    store[metric_key]={'raw_records':str((path/'memory-diagnostics.json').relative_to(folder)),
                        'status':'OBSERVED' if actual else 'NOT_APPLICABLE',
                        'write_attempts':sum(row['write_attempts'] for row in actual),'accepted_writes':sum(row['accepted_writes'] for row in actual),
                        'rejected_writes':sum(row['rejected_writes'] for row in actual),'evictions':sum(row['evictions'] for row in actual),
                        'query_layer_reads':len(reads),'key_hits':sum(row['correct_key_hit'] for row in reads),
                        'fact_value_hits':sum(row['correct_fact_value_hit'] for row in reads),'joint_hits':sum(row['key_value_joint_hit'] for row in reads),
                        'semantic_binding_retrieval':'NOT_AVAILABLE','causal_contribution':'NOT_AVAILABLE'}
            for name,value in [('validation',validation),('prediction-distribution',distribution),('loss-decomposition',decomposition),('state-diagnostics',states),('innovation-store',store)]:write(folder/'metrics'/(name+'.json'),value)
            convergence={}
            for stage,points in validation.items():
                ordered=sorted(((int(step),value) for step,value in points.items()),key=lambda item:item[0])
                first=next((step for step,value in ordered if value['active_exact_match']>value['random_active_accuracy']),None)
                confident=next((step for step,value in ordered if value['active_bootstrap_95ci'][0]>value['random_active_accuracy']),None)
                stable=None;previous=False
                for step,value in ordered:
                    current=(value['active_exact_match']>=p['gate']['validation_accuracy_min'] and value['active_bootstrap_95ci'][0]>value['random_active_accuracy'] and value['random_active_nll']-value['active_nll']>=p['gate']['nll_improvement_min'])
                    if current and previous and stable is None:stable=step
                    previous=current
                convergence[stage]={'first_observed_above_random_stage_step':first,'first_ci_above_random_stage_step':confident,
                    'first_two_consecutive_validation_stage_step':stable,
                    'stability_definition':'Two consecutive scheduled points meeting the three Stage1 Validation conditions;descriptive only,no extra Stage2 gate'}
            write(folder/'metrics/convergence.json',convergence)

            import csv
            with (folder/'metrics/learning-curves.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['stage','stage_step','global_step','source','active_accuracy','active_nll','random_accuracy','random_nll'])
                for row in checkpoint['history']:writer.writerow([row['stage'],row['stage_step'],row['global_step'],'online_pre_update',row['active_exact_match'],row['active_nll'],row['random_accuracy'],row['random_nll']])
                for stage,points in validation.items():
                    for step,value in points.items():writer.writerow([stage,step,int(step)+(128 if stage=='2' else 0),'validation',value['active_exact_match'],value['active_nll'],value['random_active_accuracy'],value['random_active_nll']])
            logs=folder/'logs';logs.mkdir(exist_ok=True)
            (logs/'training.log').write_text(''.join((folder/'training'/('stage'+str(stage))/'training.log').read_text() for stage in stages))
            (logs/'validation.log').write_text(json.dumps(validation))
            for raw in (out/'logs').glob('*-'+arm+'.log'):shutil.copy2(raw,logs/('gpu-'+raw.name))
            for name in ('p3-cpu-preflight.json','cpu-tests.log'):
                if (out/name).exists():shutil.copy2(out/name,folder/name)
            sources=json.loads((BASE.parent/'b0/configs/dependency-lock.json').read_text())['files']
            for f in [*BASE.glob('*.py'),*(BASE.parent/'b0/src').glob('*.py'),*(BASE.parent/'b0-p2').glob('*.py')]:sources[str(f.relative_to(ROOT))]=file_hash(f)
            write(folder/'source-files.json',sources)
            (folder/'REPORT.md').write_text(model_report(arm,results[arm],gates[arm],isolation[arm],'PENDING_INDEPENDENT_DOWNLOAD_RESTORE'))
            write(folder/'stage1-gate.json',gate)
        artifact=publish(folder);extracted,download_proof=download(artifact,out/'independent'/arm)
        payload=verify_payload(extracted,banks,p,identities[arm],stages)
        proof_file=extracted/'independent-restore/verification.json'
        if not proof_file.exists():charged(out,'verify',arm,final_protocol,extracted/'datasets',extracted)
        proof={'download':download_proof,'payload':payload,'restore':json.loads(proof_file.read_text()),'stage1_gate':gate['status'],'stage2_started':2 in stages,
            'charged_worker_phases':[event for event in json.loads((out/'gpu-budget.json').read_text())['phases'] if event['arm']==arm],
            'independent_worker_log':(out/'logs'/('verify-'+arm+'.log')).read_text() if (out/'logs'/('verify-'+arm+'.log')).exists() else ''}
        index=metadata(identities[arm],artifact,proof,model_report(arm,results[arm],gates[arm],isolation[arm],proof))
        returned[arm]={'artifact':artifact,'verification':proof,'index':index}
        write(out/'decision.json',{'status':'P3_RETURN_IN_PROGRESS','stage1_gate':gate['status'],'stage2_started':2 in stages,'arms':returned})
    paired={};rng=np.random.default_rng(p['seeds']['statistics'])
    for stage in stages:
        paired[str(stage)]={}
        for a,b in [('E6','E0'),('E7-Innovation','E6'),('E7-Innovation','E0')]:
            x=results[a][stage]['validation']['clusters'];y=results[b][stage]['validation']['clusters']
            if [r['mapping_hash'] for r in x]!=[r['mapping_hash'] for r in y]:raise ValueError('Paired Mapping IDs differ')
            differences={}
            for metric in ('active_accuracy','active_nll'):
                delta=np.array([r[metric]-s[metric] for r,s in zip(x,y)]);boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
                differences[metric]={'mean':float(delta.mean()),'episode_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
            paired[str(stage)][a+' minus '+b]=differences
    decision={'execution_status':'COMPLETE','stage1_gate':gate['status'],'stage2_started':2 in stages,'stages':stages,'arms':returned,
        'paired_analysis':paired,'gpu_budget':json.loads((out/'gpu-budget.json').read_text()),'test_status':p['test_status'],
        'controller_wall_before_summary_seconds':time.monotonic()-started,'hypotheses':'NOT_EVALUATED'}
    report='# B0-P3 Summary\n\nStage1 gate: '+gate['status']+'; Stage2 started: '+str(2 in stages)+'.\n\n'
    for arm in p['models_order']:report+=model_report(arm,results[arm],gates[arm],isolation[arm],returned[arm]['verification'])+'\n'
    if gate['status']=='BLOCKED':report+='\nAll-arm basic recall was not established under the predefined gate. Stage2 and32–64-pair scaling are not justified by this run; inspect failed checks and curves. This is not an architecture-impossibility claim.\n'
    else:report+='\nAll arms passed the predefined8-pair candidate-conditioned gate. Stage2 measures16-pair adaptation; this is not a full-vocabulary generation or causal memory-mechanism claim. Multi-seed confirmation and Store ablation remain future decisions.\n'
    report+='\n## Learning speed and implementation cost\n\n'
    costs={arm:sum(event.get('elapsed_seconds',0) for event in decision['gpu_budget']['phases'] if event['arm']==arm and event['action'].startswith('stage')) for arm in p['models_order']}
    report+='Charged formal-training worker wall by arm (includes Validation/checkpoint/CPU observations): '+json.dumps(costs)+'. Lowest observed: '+min(costs,key=costs.get)+'. This compares this reference implementation, not universal architectural complexity.\n\n'
    for arm in p['models_order']:
        report+=arm+' convergence: '+(out/arm/'metrics/convergence.json').read_text()+'\n'
        if 2 in stages:
            before=results[arm][1]['validation'];initial16=json.loads((out/arm/'training/stage2/evaluation/0/metrics.json').read_text());after=results[arm][2]['validation']
            skill=lambda value:(value['active_exact_match']-value['random_active_accuracy'])/(1-value['random_active_accuracy'])
            report+=f"{arm}:8-pair final EM {before['active_exact_match']:.4f};16-pair start EM {initial16['active_exact_match']:.4f};16-pair final EM {after['active_exact_match']:.4f}. Chance-normalized start shift {skill(initial16)-skill(before):.4f}. The start shift uses the same trained weights but different independent mappings; continued16-pair training is a separate effect.\n"
    report+='\nE6 success establishes the combined Mamba/SWA/Summary path, not a causal Summary-only contribution. E7 versus E6 point estimates/intervals and position traces are preliminary; Store ablation and independent seeds would be needed for a mechanism claim. Further scaling or formal replication are research-owner decisions, not automatic actions.\n'
    report+='\n## Paired Mapping-cluster analysis\n\n'+json.dumps(paired,indent=2)+'\n\nGPU phase account:\n\n'+json.dumps(decision['gpu_budget'],indent=2)
    if not isinstance(experiment_identity,dict):raise TypeError('Experiment identity must be a mapping')
    summary={**experiment_identity,'phase':'b0-p3-summary','run_id':'b0-p3-summary-'+digest(experiment_identity)[:20]}
    decision['summary_index']=metadata(summary,{a:r['artifact'] for a,r in returned.items()},decision,report)
    decision['end_to_end_wall_seconds']=time.monotonic()-started;write(out/'decision.json',decision);(out/'REPORT.md').write_text(report)
    print({'execution_status':'COMPLETE','stage1_gate':gate['status'],'stage2_started':2 in stages,'gpu_charged_seconds':decision['gpu_budget']['charged_seconds']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Controller interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:launch(args)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        budget=json.loads((args.output/'gpu-budget.json').read_text()) if (args.output/'gpu-budget.json').exists() else {}
        status='TIME_BUDGET_EXCEEDED' if isinstance(error,TimeoutError) or budget.get('charged_seconds',0)>=7200 else 'STOPPED'
        import traceback
        failed_frame=traceback.extract_tb(error.__traceback__)[-1]
        failure={'failed_phase':failed_frame.name,'failed_line':failed_frame.lineno,'status':status,'error':type(error).__name__+': '+str(error),'gpu_budget':budget,'test_status':'SEALED_NOT_ACCESSED'}
        write(args.output/'execution-error.json',failure)
        (args.output/'STOPPED.md').write_text('# B0-P3 stopped\n\n'+json.dumps(failure,indent=2)+'\n\nNo incomplete run is a PASS. Retained checkpoints and logs are authoritative; no fallback or budget expansion.\n')
        raise
