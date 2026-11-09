"""Bounded M1 E0 smoke; no Screening/Confirmation or other architectures."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import platform
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from data import MQARConfig, episode, dataset_manifest, prepare_wikitext, lm_rows, digest, file_hash
from e0 import E0, Config, fit, parameters, evaluate, write, BASE, ROOT, load_checkpoint


def gpu_environment():
    info={'python':platform.python_version(),'torch':torch.__version__,'cuda_build':torch.version.cuda,
          'cuda_available':torch.cuda.is_available(),'platform':platform.platform(),
          'dependencies':{p:importlib.metadata.version(p) for p in ('numpy','tokenizers','datasets')}}
    if info['cuda_available']:
        info.update(gpu=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
                    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    return info


def publish(out, decision):
    """Existing publisher handles isolated main-based artifact commits and SHA check."""
    import sys
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import ExperimentSpec, prepare_artifacts, push_results
    names=('decision.json','manifest.json','metrics.json','metrics.jsonl','parameter-count.json','dataset-manifest.json')
    destination=ROOT/'artifacts/experiments/epclm-004-m1-e0'/decision['run_id']
    staging=ROOT/'results/epclm-004-m1-e0/publish';staging.mkdir(parents=True,exist_ok=True)
    for n in names:shutil.copy2(out/n,staging/n)
    spec=ExperimentSpec(str(staging.relative_to(ROOT)),str(destination.relative_to(ROOT)),names,
                        decision['result_branch'],'minicells.epclm004.m1.v1')
    # Call the shared API directly; no duplicate publisher or registry is needed.
    folder=prepare_artifacts(ROOT,'epclm-004-m1-e0',spec,None)
    push_results(ROOT,folder,'epclm-004-m1-e0',decision['result_branch'],'GITHUB_TOKEN','origin/main')
    receipt={'status':'REMOTE_SHA_VERIFIED','branch':decision['result_branch']}
    write(out/'publication.json',receipt);print(receipt,flush=True)


def launch(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.publish_only:
        publish(out,json.loads((out/'decision.json').read_text()));return
    if args.profile=='gpu-smoke' and args.device!='cuda':raise ValueError('GPU profile requires CUDA')
    started=time.monotonic();deadline=started+1800
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code_hash=digest({p:file_hash(BASE/p) for p in ('data.py','e0.py','run.py','configs/model-e0.json')})
    c=MQARConfig(bindings=2 if args.profile=='cpu-tiny' else 8,length=32 if args.profile=='cpu-tiny' else 128,
                 queries=1,seed=41001)
    model_values=json.loads((BASE/'configs/model-e0.json').read_text())
    if args.profile=='cpu-tiny':model_values.update(dim=32,layers=1,heads=4,kv_heads=4,ffn=64)
    model_config=Config(**model_values)
    training={'seed':41017,'batch':4,'lr':.001 if args.profile=='cpu-tiny' else .0003,
              'updates':12 if args.profile=='cpu-tiny' else (32 if args.task=='mqar' else 4),
              'precision':'fp32','scheduler':'constant-smoke-only','attention':'full-causal-supplied-sequence-no-training-cache'}
    config={'profile':args.profile,'task':args.task,'mqar':asdict(c),'model':asdict(model_config),'training':training}
    if (out/'decision.json').exists():
        previous=json.loads((out/'decision.json').read_text())
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['source_commit']!=source or manifest['config']!=config or manifest['implementation_sha256']!=code_hash:
            raise ValueError('existing output belongs to different source/config; use a new output')
        if previous['status']=='COMPLETE':
            checkpoint=out/'checkpoint.pt'
            metrics=json.loads((out/'metrics.json').read_text())
            if file_hash(checkpoint)!=metrics['checkpoint']['sha256']:raise ValueError('completed checkpoint hash differs')
            print({'status':'ALREADY_COMPLETE_NO_UPDATES','ledger':metrics['ledger']},flush=True)
            if args.publish:publish(out,previous)
            return
        run_id=previous['run_id']
    else:run_id=f'epclm004-m1-{args.profile}-{args.task}-{source[:12]}-{uuid.uuid4().hex[:8]}'
    decision={'format':'minicells.epclm004.m1.v1','run_id':run_id,'source_commit':source,
              'status':'INCOMPLETE','stage':'PRECHECK','scientific_hypotheses':'NOT_EVALUATED',
              'independent_test_accessed':False,'result_branch':'kaggle/'+run_id,'publication':'PENDING' if args.publish else 'NOT_REQUESTED'}
    manifest={'format':'minicells.epclm004.run.v1','source_commit':source,'implementation_sha256':code_hash,
              'config':config,'config_sha256':digest(config),'environment':gpu_environment(),
              'seed':41017,'test_status':'SEALED_NOT_ACCESSED','evidence_level':'ENGINEERING_DEVELOPMENT',
              'formal_training_tokens':0,'kaggle_script_version_id':None,'kaggle_identity_status':'ID_PENDING_IF_NOT_EXPOSED',
              'budget':{'max_updates':training['updates'],'execution_seconds':1800,'scope':'M1 smoke only'}}
    write(out/'manifest.json',manifest)
    for n,v in [('metrics.json',{}),('parameter-count.json',{}),('dataset-manifest.json',{})]:
        if not (out/n).exists():write(out/n,v)
    if not (out/'metrics.jsonl').exists():(out/'metrics.jsonl').write_text('')
    write(out/'decision.json',decision)
    try:
        if args.publish:
            import sys
            sys.path.insert(0,str(ROOT))
            from scripts.research.publish_experiment_results import load_github_token,validate_origin
            validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        if args.device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('GPU_UNAVAILABLE')
            if torch.cuda.get_device_capability(0)<(7,5):raise RuntimeError('GPU_BELOW_TARGET_T4_SM75; no GPU training attempted')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        decision['stage']='ASSETS'
        assets=prepare_wikitext(args.assets)
        manifest['tokenizer_sha256']=assets['files']['tokenizer.json']
        manifest['wikitext_assets']=assets
        if args.task=='mqar':
            rows={s:[episode(c,s,i) for i in range(n)] for s,n in [('train',128),('validation',32)]}
        else:rows={s:lm_rows(args.assets,s,c.length,n) for s,n in [('train',128),('validation',32)]}
        data_manifest=dataset_manifest(config['mqar'] if args.task=='mqar' else assets['request'],rows,manifest['tokenizer_sha256'])
        write(out/'dataset-manifest.json',data_manifest);manifest['dataset_sha256']=digest(data_manifest)
        random.seed(training['seed']);np.random.seed(training['seed']);torch.manual_seed(training['seed'])
        model=E0(model_config).to(args.device)
        count=parameters(model);write(out/'parameter-count.json',count);manifest['parameters']=count
        if args.profile=='gpu-smoke' and count['static_parameters']!=10130688:raise ValueError('M0 parameter baseline differs')
        write(out/'manifest.json',manifest)
        identity={k:manifest[k] for k in ('source_commit','implementation_sha256','config_sha256','dataset_sha256','tokenizer_sha256')}
        decision['stage']='TRAINING'
        result=fit(model,rows['train'],rows['validation'],training,out,identity,deadline)
        # Exercise restoration in a fresh instance; optimizer/RNG/sampler restore
        # equivalence is covered by the tiny integration test rather than GPU repeats.
        state=load_checkpoint(out/'checkpoint.pt');fresh=E0(model_config).to(args.device);fresh.load_state_dict(state['model'])
        restored=evaluate(fresh,rows['validation'],training['batch'])
        if (not math.isclose(restored['nll'],result['validation']['nll'],rel_tol=1e-6,abs_tol=1e-6)
                or restored['query_exact_match']!=result['validation']['query_exact_match']):
            raise ValueError('checkpoint restored evaluation differs')
        result['checkpoint']['restored_evaluation']='PASS'
        if args.task=='mqar':
            removed=[]
            for row in rows['validation']:
                r={**row,'input_ids':row['input_ids'].copy()}
                r['input_ids'][1:1+3*c.bindings]=[7]*(3*c.bindings);removed.append(r)
            result['no_binding_control']=evaluate(fresh,removed,training['batch'])
        else:result['validation']['ppl']=math.exp(min(result['validation']['nll'],20))
        write(out/'metrics.json',result)
        manifest['ledger']=result['ledger'];manifest['checkpoint']=result['checkpoint'];write(out/'manifest.json',manifest)
        decision.update(status=result['status'],stage='FINISHED',ledger=result['ledger'],checkpoint_restore='PASS')
    except Exception as error:
        decision.update(status='INCOMPLETE',error=f'{type(error).__name__}: {error}')
        if (out/'checkpoint.pt').exists():
            decision['committed_ledger']=load_checkpoint(out/'checkpoint.pt')['ledger']
        print(decision,flush=True)
    finally:
        decision['wall_seconds']=time.monotonic()-started;write(out/'decision.json',decision)
    if args.publish:publish(out,decision)
    print({'status':decision['status'],'report':str(out/'decision.json')},flush=True)
    if decision['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=['cpu-tiny','gpu-smoke'],default='cpu-tiny')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--task',choices=['mqar','lm'],default='mqar')
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publish',action='store_true');parser.add_argument('--publish-only',action='store_true')
    launch(parser.parse_args())
