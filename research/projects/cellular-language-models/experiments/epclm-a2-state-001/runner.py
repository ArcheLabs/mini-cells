"""Fresh B1/B2/B3, finalized evidence, independent publication receipts."""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
import torch
from bootstrap import BASE,ROOT,shared,old_evaluate,old_train
from .architecture import InteractionModel,ARMS
from .audit import load,inspect,verify_inputs
from .training import run as train_arm
from .evaluate import run as evaluate_arm
from .flops import report as flops
from .report import render

EXPERIMENT='epclm-a2-state-001'
FILES=('config.json','training-metrics.json','validation-results.json','inference-profile.json','flops-report.json','state-diagnostics.json','decision.json')


def publish(out,branch):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results,remote_branch_sha
    spec=EXPERIMENTS[EXPERIMENT];destination=ROOT/spec.source_dir
    if destination.exists():shutil.rmtree(destination)
    destination.mkdir(parents=True)
    names=[]
    for source in out.rglob('*'):
        if not source.is_file() or source.suffix not in ('.json','.md','.pt','.py','.ipynb') or source.name.startswith('publication-'):continue
        name=source.relative_to(out).as_posix();target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);names.append(name)
    artifact=prepare_artifacts(ROOT,EXPERIMENT,replace(spec,files=tuple(sorted(names))),None)
    before=shared.digest(out/'decision.json');push_results(ROOT,artifact,EXPERIMENT,branch,'GITHUB_TOKEN','origin/main')
    first=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt={'format':'minicells.a2-state.publication-receipt.v1','status':'ARTIFACT_COMMIT_REMOTE_SHA_VERIFIED',
             'artifact_commit':first,'branch':branch,'decision_sha256':before,'decision_immutable':True,
             'note':'this receipt is outside the original SHA manifest; its own commit is verified separately in stdout/local verification'}
    shared.write(artifact/'publication-receipt.json',receipt);shared.write(out/'publication-receipt.json',receipt)
    # Append receipt to the verified artifact commit; preserve the original SHA manifest.
    push_results(ROOT,artifact,EXPERIMENT,branch,'GITHUB_TOKEN',None)
    final=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if remote_branch_sha(ROOT,branch)!=final or shared.digest(out/'decision.json')!=before:raise RuntimeError('receipt verification or immutable decision check failed')
    shared.write(out/'publication-verification.json',{'status':'REMOTE_SHA_VERIFIED','artifact_commit':first,'receipt_commit':final,'branch':branch,'decision_sha256':before})
    return final


def run(args):
    p=load();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run_id=EXPERIMENT+'-'+source[:12]+'-'+uuid.uuid4().hex[:8];out=args.output/run_id;out.mkdir(parents=True)
    deadline=time.monotonic()+min(args.execution_seconds,p['budget']['execution_seconds'])
    identity={'source_commit':source,'protocol_sha256':shared.digest(BASE/'protocol.json'),'seed':91001,'state_seed':p['state_seed'],
              'run_id':run_id,'profile':'dev','match':'tokens','independent_test_accessed':False}
    decision={'format':'minicells.epclm-a2-state-001.v1',**identity,'status':'INCOMPLETE','training_status':'NOT_STARTED','evaluation_status':'NOT_STARTED',
              'publication_status':'SEE_INDEPENDENT_RECEIPT','scientific_confirmation':False,'hypotheses':dict.fromkeys(('H1','H2','H3'),'NOT_RUN')}
    models={};device=torch.device('cuda:0');shared.write(out/'protocol.json',p)
    for source_file in BASE.iterdir():
        if source_file.is_file() and source_file.suffix in ('.py','.json','.md','.ipynb'):
            target=out/'source'/source_file.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source_file,target)
    identity['implementation_sha256']=shared.sha({f.name:shared.digest(f) for f in BASE.glob('*.py')});decision.update(identity)
    try:
        audit=inspect();shared.write(out/'baseline-audit.json',audit)
        if not torch.cuda.is_available() or torch.cuda.get_device_name()!='Tesla T4':
            decision['status']=decision['training_status']='NOT_RUN_RESOURCE_UNAVAILABLE';return decision
        from scripts.research.publish_experiment_results import validate_origin,load_github_token
        validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
        check=subprocess.run([sys.executable,'-m','pytest','-q',str(BASE/'tests')],cwd=ROOT,env=env,text=True,capture_output=True,timeout=min(120,max(1,deadline-time.monotonic())))
        shared.write(out/'correctness.json',{'returncode':check.returncode,'stdout':check.stdout[-16000:],'stderr':check.stderr[-2000:]})
        if check.returncode:raise RuntimeError('A2 State correctness tests failed; training not started')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from epclm001.retrain_data import prepare
        frozen=json.loads((shared.INCREMENTAL.parent/'retrain_config.json').read_text())
        with shared.bounded(max(1,deadline-time.monotonic())):corpus=prepare(args.cache,frozen)
        inputs=verify_inputs(corpus);identity.update(inputs);decision.update(inputs)
        historical=json.loads((BASE/'reports/history/manifest.json').read_text());starts=historical['evaluation_offsets']
        tokenizer=shared.native.load_tokenizer(corpus.tokenizer_path)
        environment={'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'precision':'FP16 autocast, FP32 parameters, FP32 GRU update'}
        decision['environment']=environment
        historical_environment=historical['environment'];decision['historical_environment_match']=all(environment[k]==str(historical_environment[k]) for k in ('gpu','torch','cuda'))
        shared.write(out/'manifest.json',{**identity,'environment':environment,'data':corpus.manifest,'evaluation_offsets':starts,'baseline_audit':audit})
        for arm in ARMS:
            folder=out/'models'/arm;folder.mkdir(parents=True);model=None
            entry={**identity,'arm':arm,'training_status':'NOT_STARTED','evaluation_status':'NOT_STARTED','strict_historical_initialization_match':False}
            models[arm]=entry
            try:
                if time.monotonic()>=deadline:raise TimeoutError('job budget exhausted before arm')
                model=InteractionModel(arm,state_seed=p['state_seed'])
                if model.base_initial_sha256!=audit['initialization_reconstruction_sha256']:raise ValueError('base initialization differs')
                entry.update(base_initial_sha256=model.base_initial_sha256,state_initial_sha256=model.state_initial_sha256,
                    configuration=model.configuration,parameters=sum(v.numel() for v in model.parameters()))
                shared.write(folder/'config.json',entry)
                model=model.to(device=device,dtype=torch.float32)
                with shared.bounded(max(1,deadline-time.monotonic())):train_arm(model,corpus,starts[0],p['training'],device,folder,deadline,entry)
                model.zero_grad(set_to_none=True)
                checkpoint=folder/'final-model.pt';shared.native._atomic_save({'model':model.state_dict(),'configuration':model.configuration,'identity':identity},checkpoint)
                entry.update(model_weight_identity=shared.model_hash(model),checkpoint_sha256=shared.digest(checkpoint),checkpoint_bytes=checkpoint.stat().st_size)
                shared.write(folder/'flops-report.json',flops(model));entry['flops-report.json']=flops(model)
                entry['training-metrics.json']=json.loads((folder/'training-metrics.json').read_text())
                with shared.bounded(max(1,deadline-time.monotonic())):results=evaluate_arm(model.eval(),corpus,tokenizer,starts,p['evaluation'],device,folder,deadline)
                entry.update(results)
                entry['evaluation_status']='COMPLETE' if all(v.get('status') in ('COMPLETE','MEASURED') for v in results.values()) else 'INCOMPLETE'
            except Exception as error:
                entry['error']=type(error).__name__+': '+str(error)
                if entry['training_status']=='IN_PROGRESS':entry['training_status']='INCOMPLETE'
                if model is not None and entry['training_status']=='COMPLETE' and not (folder/'final-model.pt').exists():
                    try:shared.native._atomic_save({'model':model.state_dict(),'configuration':model.configuration,'identity':identity},folder/'final-model.pt')
                    except Exception as save_error:entry['checkpoint_error']=str(save_error)
            finally:
                shared.write(folder/'decision.json',entry)
                for filename in FILES:
                    if not (folder/filename).exists():shared.write(folder/filename,{'status':'NOT_RUN','reason':entry.get('error','not reached')})
                if model is not None:del model
                torch.cuda.empty_cache();shared.write(out/'stage-results.json',models)
        decision['training_status']='COMPLETE' if all(e['training_status']=='COMPLETE' for e in models.values()) else 'INCOMPLETE'
        decision['evaluation_status']='COMPLETE' if all(e['evaluation_status']=='COMPLETE' for e in models.values()) else 'INCOMPLETE'
        decision['status']='DEVELOPMENT_PILOT_COMPLETE' if decision['training_status']==decision['evaluation_status']=='COMPLETE' else 'INCOMPLETE'
        decision['hypotheses']=dict.fromkeys(('H1','H2','H3'),'INCONCLUSIVE' if any(e['training_status']=='COMPLETE' for e in models.values()) else 'NOT_RUN')
    except Exception as error:decision['error']=type(error).__name__+': '+str(error)
    finally:
        shared.write(out/'stage-results.json',models)
        # Decision is finalized once, before the publisher constructs its SHA manifest.
        shared.write(out/'decision.json',decision);(out/'RESEARCH-REPORT.zh-CN.md').write_text(render(decision,models))
        branch='kaggle/'+run_id
        try:
            with shared.bounded(p['budget']['publication_seconds']):final=publish(out,branch)
            print(json.dumps({'status':decision['status'],'publication_status':'REMOTE_SHA_VERIFIED','remote_sha':final,'branch':branch,'report':str(out/'decision.json')}),flush=True)
        except Exception as error:
            shared.write(out/'publication-failure.json',{'status':'FAILED','error':type(error).__name__+': '+str(error),'output_preserved':True})
            print(json.dumps({'status':decision['status'],'publication_status':'FAILED','report':str(out/'decision.json')}),flush=True)
    return decision
