"""Single-entry P2: audited data, measured budget, fresh training, independent return."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from p2_dataset import BASE,build
from b0_audit import verify_dependencies
from b0_integrity import environment,publish_bundle,publisher
from b0_models import state_hash
from e0 import ROOT,write,load_checkpoint
from data import digest,file_hash
from p2_integrity import download,verify_payload,publish


def checked_data(folder,p):
    banks,audit=build(folder,p['split_seed'])
    limits=p['p2'];js=audit['marginal_jsd_bits']
    if max(js['key_unique'],js['value_unique'])>limits['unique_marginal_jsd_bits_max'] or max(js['key_observed'],js['value_observed'])>limits['observed_marginal_jsd_bits_max']:
        raise ValueError('STOP: approved marginal limits not met')
    audit.update(acceptance_status='PASS',approved_limits={'unique_jsd_bits':limits['unique_marginal_jsd_bits_max'],
        'observed_jsd_bits':limits['observed_marginal_jsd_bits_max']})
    write(Path(folder)/'SPLIT-AUDIT.json',audit)
    manifest=json.loads((Path(folder)/'DATASET-MANIFEST.json').read_text());manifest['split_audit_sha256']=file_hash(Path(folder)/'SPLIT-AUDIT.json')
    write(Path(folder)/'DATASET-MANIFEST.json',manifest)
    frozen=BASE/'data/DATASET-MANIFEST.json'
    if frozen.exists():
        expected=json.loads(frozen.read_text())
        if manifest['dataset_sha256']!=expected['dataset_sha256'] or manifest['files']!=expected['files']:
            raise ValueError('STOP: frozen P2 Episode bank hashes differ')
    return banks,audit,manifest


def charged_phase(out,action,arm,pfile,data,folder):
    path=out/'gpu-budget.json'
    budget=json.loads(path.read_text()) if path.exists() else {'charged_seconds':0.,'phases':[]}
    if budget.get('active'):
        budget['charged_seconds']+=min(time.time()-budget['active']['started_epoch'],7200-budget['charged_seconds'])
        budget['phases'].append({**budget.pop('active'),'status':'INTERRUPTED_PREVIOUS_CONTROLLER'})
    remaining=7200-budget['charged_seconds']
    if remaining<=0:
        budget['status']='TIME_BUDGET_EXCEEDED';write(path,budget)
        raise TimeoutError('TIME_BUDGET_EXCEEDED')
    print({'stage':action,'arm':arm,'gpu_seconds_remaining':remaining},flush=True)
    began=time.monotonic();budget['active']={'action':action,'arm':arm,'started_epoch':time.time()};write(path,budget)
    child=None
    try:
        log=out/'logs';log.mkdir(exist_ok=True)
        with (log/(action+'-'+arm+'.log')).open('a') as stream:
            child=subprocess.Popen([sys.executable,str(BASE/'worker.py'),'--action',action,'--arm',arm,
                '--protocol',str(pfile),'--data',str(data),'--output',str(folder),'--remaining',str(remaining)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=max(0.,remaining-(time.monotonic()-began)))
            except subprocess.TimeoutExpired as error:raise TimeoutError('TIME_BUDGET_EXCEEDED') from error
        if code:raise RuntimeError('GPU phase failed; retained Output: '+action+' '+arm)
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        elapsed=time.monotonic()-began;budget['charged_seconds']+=elapsed
        budget['phases'].append({**budget.pop('active'),'elapsed_seconds':elapsed,
            'status':'COMPLETE' if child is not None and child.returncode==0 else 'STOPPED'})
        budget['status']='TIME_BUDGET_EXCEEDED' if budget['charged_seconds']>=7200 else 'WITHIN_LIMIT'
        write(path,budget)
    if budget['charged_seconds']>=7200:raise TimeoutError('TIME_BUDGET_EXCEEDED')


def plan_budget(benchmarks,prior,spent,p):
    forecasts={};factor=p['p2']['timing_safety_factor']
    for updates in p['p2']['budget_choices']:
        total=spent
        for arm,b in benchmarks.items():
            #Use the slower observed new step or prior actual mean, then25% margin.
            history=prior[arm]['history'];previous=sum(row['compute_seconds'] for row in history)/len(history)
            step=max(max(t['total'] for t in b['steps']),previous)
            checks=len([n for n in (0,256,512,768,1024) if n<=updates])+int(updates not in (256,512,768,1024))
            #Initial+scheduled evals, plus independent downloaded-checkpoint eval.
            evaluation=(checks+1)*(b['validation_seconds']+b['diagnostic_seconds'])
            saves=(updates//128+2)*b['checkpoint_seconds']
            total+=factor*(updates*step+evaluation+saves)+15
        forecasts[updates]=total
    chosen=next((u for u in p['p2']['budget_choices'] if forecasts[u]<=6480),None)
    return {'status':'PASS' if chosen else 'BLOCKED_RESOURCE_LIMIT','updates_per_arm':chosen,
        'forecast_total_seconds':forecasts,'charged_benchmark_seconds':spent,'planning_limit_seconds':6480,
        'hard_gpu_phase_seconds':7200,'safety_factor':factor,
        'timing_scope':'Conservative charged GPU subprocess wall, including imports, diagnostics, checkpoint and restore; CPU downloads/uploads excluded'}


def metadata_return(manifest,proof,verification,report):
    pub=publisher();schema='minicells.b0-p2.index.v1'
    with tempfile.TemporaryDirectory(prefix='b0-p2-index-') as temp:
        root=Path(temp)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',manifest['source_commit']],cwd=root,check=True)
        stage=root/'results/b0-p2-index';stage.mkdir(parents=True)
        write(stage/'decision.json',{'format':schema,'status':'DEVELOPMENT_PILOT_COMPLETE','manifest':manifest,
            'artifact':proof,'independent_verification':verification,'hypotheses':'NOT_EVALUATED','test_accessed':False})
        (stage/'REPORT.md').write_text(report)
        branch='kaggle/'+manifest['run_id']
        spec=pub.ExperimentSpec(str(stage.relative_to(root)),'artifacts/experiments/epclm004-b0-p2/'+manifest['run_id'],('decision.json','REPORT.md'),branch,schema)
        artifacts=pub.prepare_artifacts(root,'epclm004-b0-p2',spec,None)
        pub.push_results(root,artifacts,'epclm004-b0-p2',branch,'GITHUB_TOKEN','origin/main')
        return {'status':'REMOTE_SHA_VERIFIED','branch':branch,'sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}


def report_text(arm,result,verified):
    final=result['curve'][-1]
    return '# B0-P2 '+arm+'\n\n'+json.dumps({'status':result['status'],'ledger':result['ledger'],
        'training':{k:final['training'][k] for k in ('correct','query_targets','nll','type_nll','value_conditional_nll')},
        'validation':{k:final['validation'][k] for k in ('correct','query_targets','nll','type_nll','value_conditional_nll')},
        'independent_verification':verified},indent=2)+'\n\nSingle-seed development evidence only. Split and exposure changed jointly; no isolated attribution or H5 confirmation. Test unaccessed; no12M, main three-seed experiment or M6. Store position hits do not establish causal utility.\n'


def summary_report(out,decision):
    results=decision['arms'];budget=decision['gpu_budget']
    lines=['# B0-P2 Controlled Learnability Pilot','',
        'Status: '+decision['status']+'. Single-seed development; H1–H6 NOT_EVALUATED. Test unaccessed.','',
        f"Cumulative charged GPU-phase wall: {budget['charged_seconds']:.2f}/7200 seconds. Controller wall through analysis: {decision['analysis_wall_seconds']:.2f} seconds (before final summary return).",'',
        '## Quality','',
        '| Arm | Training Diagnostic | Validation | Full NLL | Type NLL | Conditional NLL | Value mass | Dominant rate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in results.items():
        point=r['result']['curve'][-1];t=point['training'];v=point['validation']
        lines.append(f"| {arm} | {t['correct']}/{t['query_targets']} | {v['correct']}/{v['query_targets']} | {v['nll']:.6f} | {v['type_nll']:.6f} | {v['value_conditional_nll']:.6f} | {v['valid_value_probability_mass_mean']:.6f} | {v['dominant_token_rate']:.4f} |")
    lines+=['','Training Diagnostic uses32 Episodes already in Training. Validation uses64 Episodes/512 clustered Queries. Uniform valid-value conditional NLL is ln256=5.545177; it is not the full-vocabulary random baseline. Complete initial/scheduled curves and prediction distributions are in each verified bundle.','',
        '## Resources','',
        '| Arm | Static parameters | Updates | Fit wall seconds | Compute input tokens/s | Peak allocated MiB | Checkpoint MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in results.items():
        resource=json.loads((out/arm/'training/resources.json').read_text())
        lines.append(f"| {arm} | {resource['static_parameters']} | {r['result']['ledger']['committed_updates']} | {resource['fit_wall_seconds']:.2f} | {resource['compute_tokens_per_second']:.2f} | {resource['training_peak_allocated_bytes']/2**20:.2f} | {resource['checkpoint']['bytes']/2**20:.2f} |")
    lines+=['','Timing includes parameter observers. Compute throughput excludes evaluation/checkpoint/publication; fit wall includes completed training/evaluation/checkpoint calls. The cumulative worker clock also charges restore and imports conservatively. These are implementation-level pilot costs, not universal architecture complexity claims.','',
        '## Paired uncertainty','',
        '| Paired comparison | Metric | Mean difference | Episode-bootstrap95% CI |','|---|---|---:|---|']
    for pair,r in decision['paired_analysis'].items():
        for metric,values in r['metrics'].items():
            lo,hi=values['episode_bootstrap_95ci'];lines.append(f"| {pair} | {metric} | {values['mean_difference']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines+=['','Empirical bootstrap intervals can collapse at a floor/ceiling; they do not quantify across-seed variation.','',
        '## Previous Pilot reference','',
        '| Arm | B0-2 Validation | B0-2 NLL | P2 minus B0-2 EM |','|---|---:|---:|---:|']
    for arm,r in results.items():
        old=json.loads((out/'prior-pilot'/arm/'extracted/training/metrics.json').read_text())['validation']
        current=r['result']['curve'][-1]['validation']
        lines.append(f"| {arm} | {old['correct']}/{old['query_targets']} | {old['nll']:.6f} | {current['full_vocab_exact_match']-old['full_vocab_exact_match']:.6f} |")
    step=next(iter(results.values()))['result']['ledger']['committed_updates']
    records=json.loads((out/'E7-Innovation/training/evaluation'/str(step)/'validation/memory-diagnostics.json').read_text())
    writes=sum(row['store']['accepted_writes'] for row in records);attempts=sum(row['store']['write_attempts'] for row in records)
    reads=[read for row in records for read in row['store']['retrieval_records']]
    key_hits=sum(row['correct_key_retrieved'] for row in reads);value_hits=sum(row['correct_value_retrieved'] for row in reads)
    lines+=['','## Memory observations','',
        f"E7 final Validation: {writes}/{attempts} accepted write attempts; {key_hits}/{len(reads)} Query-layer reads select the corresponding factual Key position and {value_hits}/{len(reads)} select its Value position.",
        'These are actual selected-position overlaps across all SWA layers, not causal attribution. Retrieval scores, calibrated state saturation and a causal Store-effect intervention are NOT_AVAILABLE. Per-Episode Summary/state norms, activations, reset evidence, EMA statistics, admissions and evictions are archived.','',
        '## Interpretation and execution boundary','',
        'The quality and conditional-NLL curves describe whether E0 learns the task and how E6/E7 compare under the same input/data/update conditions. No post-hoc learnability threshold is invented. Training-versus-Validation gaps and concentration/type diagnostics distinguish memorization, answer-type learning and possible retrieval progress.',
        'Differences from B0-2 cannot isolate the cause: pair membership, supplemental observation allocation and exposure changed jointly. A single seed and small differences cannot establish architecture superiority or a general Store benefit. Full-episode runtime, parameter counts and initialization were kept fixed.',
        'Convergence comparisons use common update boundaries and the resource table. Additional controlled ablations and independent replication would be required before a scientific mechanism claim. Main12M training, the main three-seed experiment and M6 remain unauthorized.','',
        '## Verified artifacts','']
    for arm,r in results.items():lines.append(f"- {arm}: [complete Release bundle]({r['artifact']['release_url']}), SHA256 {r['artifact']['sha256']}. Independent full download, all member hashes and downloaded-checkpoint re-evaluation passed.")
    if 'end_to_end_wall_seconds' in decision:lines+=['',f"Controller end-to-end wall including verified summary return: {decision['end_to_end_wall_seconds']:.2f} seconds."]
    return '\n'.join(lines)+'\n'


def launch(args):
    started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=True)
    p=json.loads((BASE/'configs/protocol.json').read_text());verify_dependencies()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    namespace={'experiment':'B0-P2','source_commit':source,'protocol_template_sha256':digest(p)}
    stamp=out/'p2-namespace.json'
    if stamp.exists() and json.loads(stamp.read_text())!=namespace:raise ValueError('STOP: output namespace differs')
    if not stamp.exists() and (out/'decision.json').exists():raise ValueError('STOP: unrecognized existing experiment Output')
    write(stamp,namespace)
    if args.execute and subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('STOP: commit/source does not describe the modified working tree')
    banks,audit,data=checked_data(out/'data',p);write(out/'protocol-template.json',p)
    if not args.execute:
        write(out/'decision.json',{'status':'CPU_DATA_PREFLIGHT_PASS_GPU_NOT_STARTED','test_accessed':False});return
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();env=environment()
    if not env['gpu'] or 'T4' not in env['gpu']:raise RuntimeError('BLOCKED_HARDWARE: actual Tesla T4 required')
    key={'source_commit':source,'protocol_template_sha256':digest(p),'dataset_sha256':data['dataset_sha256'],
        'environment_sha256':digest(env)}
    #Only stable software fields define resume; utilization/disk/CPU RAM are observations.
    key['environment_sha256']=digest({k:env[k] for k in ('python','torch','cuda','gpu','driver','precision','tf32','amp')})
    identity_path=out/'execution-identity.json'
    if not identity_path.exists() and Path('/kaggle/input').exists():
        matches=[]
        for f in Path('/kaggle/input').rglob('execution-identity.json'):
            if json.loads(f.read_text())==key:
                budget=f.parent/'gpu-budget.json';spent=json.loads(budget.read_text())['charged_seconds'] if budget.exists() else 0
                matches.append((spent,str(f.parent)))
        if matches:shutil.copytree(Path(max(matches)[1]),out,dirs_exist_ok=True)
    if identity_path.exists() and json.loads(identity_path.read_text())!=key:raise ValueError('P2 resume identity differs')
    write(identity_path,key)
    prior={};evidence=json.loads((BASE/'configs/prior-pilot-evidence.json').read_text())
    #CPU-only independent downloads supply real historical timings, not estimates from log tails.
    for arm,record in evidence.items():
        extracted,proof=download(record['artifact'],out/'prior-pilot'/arm)
        prior[arm]=json.loads((extracted/'training/metrics.json').read_text())
        write(out/'prior-pilot'/arm/'timing.json',json.loads((extracted/'training/resources.json').read_text()))
    template=out/'protocol-template.json';benchmarks={}
    for arm in p['arms']:
        folder=out/'benchmark'/arm
        if not (folder/'benchmark.json').exists():charged_phase(out,'benchmark',arm,template,out/'data',folder)
        benchmarks[arm]=json.loads((folder/'benchmark.json').read_text())
    budget=json.loads((out/'gpu-budget.json').read_text())
    planning=plan_budget(benchmarks,prior,budget['charged_seconds'],p);write(out/'budget-plan.json',planning)
    print({'stage':'BUDGET_PLAN','plan':planning},flush=True)
    frozen=out/'protocol.json'
    if frozen.exists():
        approved=json.loads(frozen.read_text())
        if approved['training']['updates'] not in p['p2']['budget_choices']:raise ValueError('Invalid frozen update budget')
        p=approved
    else:
        if planning['status']!='PASS':raise RuntimeError('BLOCKED_RESOURCE_LIMIT:640 updates do not fit; no fallback')
        updates=planning['updates_per_arm'];p['training']['updates']=updates
        p['p2']['frozen_budget']={'updates_per_arm':updates,'epoch_equivalent':updates/512,
            'input_tokens_per_arm':updates*512,'supervised_targets_per_arm':updates*8,
            'query_targets_per_arm':updates*8,'total_input_tokens':updates*512*3,'total_supervised_targets':updates*8*3}
        write(frozen,p)
    results={}
    for arm in p['arms']:
        folder=out/arm;folder.mkdir(exist_ok=True)
        prior_manifest=evidence[arm]['manifest'];manifest={**key,'phase':'b0-p2','arm':arm,'seed':p['seeds']['model'],
            'protocol_sha256':digest(p),'config_sha256':digest(p['training']),
            'update_budget':p['training']['updates'],'initialization_sha256':prior_manifest['initialization_sha256'],
            'old_weights_loaded':False,'environment_identity_fields':['python','torch','cuda','gpu','driver','precision','tf32','amp']}
        manifest['run_id']='b0-p2-'+arm.lower()+'-'+digest(manifest)[:20]
        mf=folder/'run-manifest.json'
        if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Arm identity differs')
        write(mf,manifest)
        if not (folder/'result.json').exists():
            #A published run without its completed Output must not be silently retrained.
            import requests
            response=requests.get('https://api.github.com/repos/ArcheLabs/mini-cells/releases/tags/'+manifest['run_id'],timeout=(15,60))
            if response.status_code==200:
                checkpoint=folder/'training/checkpoint.pt'
                if not checkpoint.exists() or load_checkpoint(checkpoint)['ledger']['committed_updates']!=p['training']['updates']:
                    raise RuntimeError('MISSING_COMPLETED_OUTPUT: attach completed P2 Output; published training will not be repeated')
            if response.status_code not in (200,404):response.raise_for_status()
            write(folder/'protocol.json',p);write(folder/'config.json',p['training']);write(folder/'dataset-manifest.json',data)
            write(folder/'datasets/training.json',banks['training']);write(folder/'datasets/validation.json',banks['validation'])
            write(folder/'datasets/diagnostic.json',banks['training'][:32])
            sources=json.loads((BASE.parent/'b0/configs/dependency-lock.json').read_text())['files']
            for f in [*BASE.glob('*.py'),*(BASE.parent/'b0/src').glob('*.py')]:sources[str(f.relative_to(ROOT))]=file_hash(f)
            write(folder/'source-files.json',sources)
            for name in ('p2-cpu-preflight.json','cpu-tests.log'):
                if (out/name).exists():shutil.copy2(out/name,folder/name)
            charged_phase(out,'train',arm,frozen,out/'data',folder)
        result=json.loads((folder/'result.json').read_text())
        verify_payload(folder,p,banks['training'],banks['validation'],manifest)
        if not (folder/'publication.json').exists():
            (folder/'REPORT.md').write_text(report_text(arm,result,'PENDING_INDEPENDENT_DOWNLOAD_RESTORE'))
        artifact=publish(folder)
        extracted,download_proof=download(artifact,out/'independent'/arm)
        payload_proof=verify_payload(extracted,p,banks['training'],banks['validation'],manifest)
        proof_file=extracted/'independent-restored/verification.json'
        if not proof_file.exists():charged_phase(out,'verify',arm,frozen,extracted/'datasets',extracted)
        restore_proof=json.loads(proof_file.read_text())
        verification={'download':download_proof,'payload':payload_proof,'restore':restore_proof}
        final_report=report_text(arm,result,verification)
        index=metadata_return(manifest,artifact,verification,final_report)
        results[arm]={'result':result,'artifact':artifact,'independent_verification':verification,'index':index}
        write(out/'decision.json',{'status':'P2_IN_PROGRESS','arms':results,'test_accessed':False})
    #Same Query IDs and Episode clusters; paired comparisons never treat8 Queries as independent.
    import numpy as np
    paired={};rng=np.random.default_rng(p['seeds']['statistics'])
    for a,b in [('E6','E0'),('E7-Innovation','E6'),('E7-Innovation','E0')]:
        x=results[a]['result']['curve'][-1]['validation']['episode_metrics'];y=results[b]['result']['curve'][-1]['validation']['episode_metrics']
        if [r['episode_id'] for r in x]!=[r['episode_id'] for r in y]:raise ValueError('Paired evaluation IDs differ')
        metrics={}
        for kind in ('exact_match','nll','type_nll','conditional_nll'):
            difference=np.array([r[kind]-s[kind] for r,s in zip(x,y)])
            samples=difference[rng.integers(len(x),size=(2000,len(x)))].mean(1)
            metrics[kind]={'mean_difference':float(difference.mean()),'episode_bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist()}
        paired[a+' minus '+b]={'metrics':metrics,'episodes':len(x),'seed_count':1}
    write(out/'paired-analysis.json',paired)
    decision={'status':'P2_ANALYSIS_COMPLETE_PUBLICATION_PENDING','arms':results,'paired_analysis':paired,
        'gpu_budget':json.loads((out/'gpu-budget.json').read_text()),'analysis_wall_seconds':time.monotonic()-started,
        'test_accessed':False,'hypotheses':'NOT_EVALUATED'}
    write(out/'decision.json',decision)
    combined_report=summary_report(out,{**decision,'status':'B0_P2_COMPLETE_REQUIRES_RESEARCH_REVIEW'})
    (out/'REPORT.md').write_text(summary_report(out,decision))
    summary_manifest={**key,'run_id':'b0-p2-summary-'+digest(key)[:20],'phase':'b0-p2-summary'}
    summary_proof={'arms':{a:r['artifact'] for a,r in results.items()}}
    decision['summary_index']=metadata_return(summary_manifest,summary_proof,{'paired_analysis':paired,'gpu_budget':decision['gpu_budget']},combined_report)
    decision.update(status='B0_P2_COMPLETE_REQUIRES_RESEARCH_REVIEW',end_to_end_wall_seconds=time.monotonic()-started)
    write(out/'decision.json',decision);(out/'REPORT.md').write_text(summary_report(out,decision))
    print({'status':decision['status'],'gpu_charged_seconds':decision['gpu_budget']['charged_seconds']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Controller interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:launch(args)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'execution-error.json',{'status':'TIME_BUDGET_EXCEEDED' if isinstance(error,TimeoutError) else 'STOPPED',
            'error':type(error).__name__+': '+str(error),'test_accessed':False})
        raise
