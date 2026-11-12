"""B0-2 only: audited frozen data, fresh models, exact resume and verified return."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from preflight import BASE,run as data_preflight
import torch
from b0_models import construct,state_hash,parameter_report
from b0_training import fit_stage
from b0_metrics import evaluate
from b0_integrity import environment,verify_checkpoint,publish_bundle,publisher
from data import digest,file_hash
from e0 import ROOT,write
HERE=Path(__file__).resolve().parent


def verify_overfit(protocol,data):
    evidence=json.loads((HERE/'overfit-evidence.json').read_text());records={}
    for arm,item in evidence.items():
        exists=subprocess.run(['git','cat-file','-e',item['index_commit']+'^{commit}'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if not exists:subprocess.run(['git','fetch','--quiet','--depth=1','origin',item['index_commit']],cwd=ROOT,check=True,timeout=180)
        path='artifacts/experiments/epclm004-m5-sv-b0/'+item['run_id']+'/decision.json'
        r=json.loads(subprocess.check_output(['git','show',item['index_commit']+':'+path],cwd=ROOT))
        m=r['manifest']
        if (r['scientific_status']!='OVERFIT_ESTABLISHED' or r['publication_status']!='VERIFIED'
            or m['arm']!=arm or m['phase']!='overfit' or r['artifact']['sha256']!=item['bundle_sha256']
            or m['protocol_sha256']!=digest(protocol) or m['dataset_sha256']!=digest(data)
            or m['initialization_sha256']!=item['initialization_sha256']):
            raise ValueError('STOP: Overfit prerequisite identity differs')
        records[arm]=item
    return records


def verify_result(folder,banks,p):
    saved=verify_checkpoint(folder/'training');record=json.loads((folder/'training/metrics.json').read_text())
    ledger=saved['ledger']
    for key,value in {'committed_updates':512,'input_tokens':262144,'processed_tensor_tokens':262144,
                      'padding_tokens':0,'supervised_target_tokens':4096,'query_target_tokens':4096}.items():
        if ledger[key]!=value:raise ValueError('STOP: Pilot ledger differs: '+key)
    if [r['update'] for r in saved['history']]!=list(range(1,513)):
        raise ValueError('STOP: nonsequential update history')
    expected=sorted(digest([r['input_ids']]) for r in banks['pilot_train'])
    if sorted(r['input_sha256'] for r in saved['history'])!=expected:
        raise ValueError('STOP: Pilot did not consume each approved Episode exactly once')
    if any(r['query_targets']!=8 or r['supervised_targets']!=8 for r in saved['history']):
        raise ValueError('STOP: target count differs')
    curve=json.loads((folder/'training/curve.json').read_text())
    if [r['update'] for r in curve]!=p['pilot']['validation_updates']:
        raise ValueError('STOP: validation curve incomplete')
    import numpy as np
    labels=[y for row in banks['validation'] for y in row['labels'] if y!=-100]
    for point in curve:
        prob=folder/'training/probabilities'/('update-'+str(point['update'])+'.npz')
        if file_hash(prob)!=point['evaluation']['full_query_probability_artifact']['sha256']:
            raise ValueError('STOP: probability SHA256 differs')
        with np.load(prob) as values:
            probabilities=values['probabilities'];targets=values['targets']
            if probabilities.shape!=(512,8192) or not np.isfinite(probabilities).all() or targets.tolist()!=labels:
                raise ValueError('STOP: invalid probability artifact')
            if not np.allclose(probabilities.sum(-1),1.,atol=2e-6) or (probabilities<0).any():
                raise ValueError('STOP: probabilities are not normalized')
            expected_ids=[r['id'] for r in banks['validation'] for _ in range(8)]
            if values['episode_ids'].tolist()!=expected_ids:raise ValueError('STOP: probability Episode order differs')
            correct=int((probabilities.argmax(-1)==targets).sum())
            nll=float(-np.log(probabilities[np.arange(512),targets].astype(np.float64)).mean())
            if correct!=point['evaluation']['correct'] or abs(nll-point['evaluation']['nll'])>2e-5:
                raise ValueError('STOP: curve/probability disagreement')
    return saved,record


def publish(folder):
    manifest=json.loads((folder/'run-manifest.json').read_text())
    result=json.loads((folder/'result.json').read_text());verify_checkpoint(folder/'training')
    proof=publish_bundle(folder);proof.update(scientific_status=result['status'],training_ledger=result['ledger'])
    pub=publisher();schema='minicells.b0.pilot-index.v1'
    with tempfile.TemporaryDirectory(prefix='b0-pilot-index-') as temp:
        root=Path(temp)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',manifest['source_commit']],cwd=root,check=True)
        stage=root/'results/b0-pilot-index';stage.mkdir(parents=True)
        write(stage/'decision.json',{'format':schema,'training_status':'COMPLETED','publication_status':'VERIFIED',
            'manifest':manifest,'artifact':proof,'hypotheses':'NOT_EVALUATED','test_accessed':False})
        branch='kaggle/'+manifest['run_id']
        spec=pub.ExperimentSpec(str(stage.relative_to(root)),'artifacts/experiments/epclm004-m5-sv-b0/'+manifest['run_id'],('decision.json',),branch,schema)
        prepared=pub.prepare_artifacts(root,'epclm004-m5-sv-b0',spec,None)
        pub.push_results(root,prepared,'epclm004-m5-sv-b0',branch,'GITHUB_TOKEN','origin/main')
        proof['index']={'branch':branch,'status':'REMOTE_SHA_VERIFIED',
            'remote_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
    return proof


def write_report(folder,result):
    curve=result['curve']
    with (folder/'validation-curve.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['update','correct','query_targets','exact_match','query_nll'])
        for r in curve:
            e=r['evaluation'];writer.writerow([r['update'],e['correct'],e['query_targets'],e['correct']/e['query_targets'],e['nll']])
    (folder/'REPORT.md').write_text('# B0-2 Learnability Pilot\n\n'+json.dumps(result,indent=2)+'\n\nFresh initialization; full512-token forward;8 answer-only targets. Validation only. H5 NOT_EVALUATED.\n')


def launch(args):
    started=time.monotonic();p=json.loads((BASE/'configs/protocol.json').read_text())
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    banks,data,audit=data_preflight(out,p);evidence=verify_overfit(p,data)
    initial={};parameters={}
    for arm in p['arms']:
        model=construct(arm,p);initial[arm]=state_hash(model);parameters[arm]=parameter_report(model)
        del model
    if initial['E6']!=initial['E7-Innovation']:raise ValueError('STOP: paired initialization differs')
    mismatches=[arm for arm in p['arms'] if initial[arm]!=evidence[arm]['initialization_sha256']]
    write(out/'initialization-preflight.json',{'status':'STOPPED_INITIALIZATION_MISMATCH' if mismatches else 'PASS',
        'initialization_sha256':initial,'expected_initialization_sha256':{a:evidence[a]['initialization_sha256'] for a in p['arms']},
        'mismatched_arms':mismatches,'torch':torch.__version__,'overfit_weights_loaded':False,'parameters':parameters})
    if mismatches:raise ValueError('STOP: fresh initialization differs; require the validated T4 software environment; no fallback')
    if not args.execute:return
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('BLOCKED_HARDWARE: actual Tesla T4 required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    env=environment();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    identity={'source_commit':source,'phase':'pilot','protocol_sha256':digest(p),'dataset_sha256':digest(data),
        'software':{k:env[k] for k in ('python','torch','cuda','gpu','driver','precision','tf32','amp')}}
    header=out/'execution-manifest.json'
    if not header.exists() and Path('/kaggle/input').exists():
        candidates=[]
        for path in Path('/kaggle/input').rglob('execution-manifest.json'):
            if json.loads(path.read_text())==identity:
                score=sum(json.loads(m.read_text())['ledger']['committed_updates'] for m in path.parent.glob('*/training/metrics.json'))
                candidates.append((score,str(path.parent)))
        if candidates:shutil.copytree(Path(max(candidates)[1]),out,dirs_exist_ok=True)
    if header.exists() and json.loads(header.read_text())!=identity:raise ValueError('STOP: Pilot resume identity differs')
    write(header,identity);deadline=started+min(9000,p['execution']['execution_seconds']);results={}
    decision={'status':'PILOT_IN_PROGRESS','test_accessed':False,'overfit_weights_loaded':False,'hypotheses':'NOT_EVALUATED','arms':results}
    try:
        for arm in p['arms']:
            folder=out/arm;folder.mkdir(exist_ok=True)
            model=construct(arm,p).cuda()
            if state_hash(model)!=initial[arm]:raise ValueError('STOP: initial model changed')
            manifest={**identity,'arm':arm,'seed':p['seeds']['model'],'initialization_sha256':initial[arm],
                'overfit_index_commit':evidence[arm]['index_commit'],'overfit_weights_loaded':False}
            manifest['run_id']='b0-pilot-'+arm.lower()+'-'+digest(manifest)[:20]
            path=folder/'run-manifest.json'
            if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('STOP: Pilot arm identity differs')
            write(path,manifest)
            if not (folder/'result.json').exists():
                if not (folder/'training/checkpoint.pt').exists():
                    import requests
                    response=requests.get('https://api.github.com/repos/ArcheLabs/mini-cells/releases/tags/'+manifest['run_id'],
                        headers={'Authorization':'Bearer '+publisher().load_github_token('GITHUB_TOKEN')},timeout=(15,60))
                    if response.status_code==200:raise RuntimeError('MISSING_OUTPUT: attach completed Pilot Output; no retraining')
                    if response.status_code!=404:response.raise_for_status()
                write(folder/'environment.json',env);write(folder/'protocol.json',p);write(folder/'dataset-manifest.json',data)
                write(folder/'data-preflight.json',audit);write(folder/'training-episodes.json',banks['pilot_train']);write(folder/'validation-episodes.json',banks['validation'])
                write(folder/'parameters.json',parameter_report(model))
                result=fit_stage(model,banks['pilot_train'],banks['validation'],p,folder/'training',manifest,'pilot',deadline)
                if result['status']=='INCOMPLETE_RESOURCE_LIMIT':raise TimeoutError('Pilot prefix saved; attach Output and resume')
                saved,record=verify_result(folder,banks,p)
                restored=construct(arm,p).cuda();restored.load_state_dict(saved['model'])
                if state_hash(restored)!=state_hash(model):raise ValueError('STOP: restored model differs')
                recalled=evaluate(restored,banks['validation'],1,bootstrap_draws=p['metrics']['bootstrap_draws'])
                delta=abs(recalled['nll']-record['validation']['nll'])
                if recalled['correct']!=record['validation']['correct'] or delta>2e-5:
                    raise ValueError('STOP: restored Validation differs')
                result['checkpoint_restore']={'status':'PASS','validation_nll_abs_error':delta,'tolerance':2e-5,
                    'correct':recalled['correct'],'query_targets':recalled['query_targets'],'new_optimizer_updates':0}
                del restored,saved
                write(folder/'result.json',result);write_report(folder,result)
            else:
                result=json.loads((folder/'result.json').read_text());verify_result(folder,banks,p)
            del model;torch.cuda.empty_cache()
            sequence=[json.loads(line)['input_sha256'] for line in (folder/'training/updates.jsonl').read_text().splitlines()]
            if results:
                first=next(iter(results))
                reference=[json.loads(line)['input_sha256'] for line in (out/first/'training/updates.jsonl').read_text().splitlines()]
                if sequence!=reference:raise ValueError('STOP: paired training order differs')
            results[arm]={**result,'publication':publish(folder)}
            write(out/'decision.json',decision)
        sequences=[[json.loads(line)['input_sha256'] for line in (out/a/'training/updates.jsonl').read_text().splitlines()] for a in p['arms']]
        if any(s!=sequences[0] for s in sequences[1:]):raise ValueError('STOP: paired training order differs')
        decision.update(status='PILOT_COMPLETE_REQUIRES_RESEARCH_REVIEW',publication_status='VERIFIED')
    except Exception as error:
        decision.update(status='STOPPED',error=type(error).__name__+': '+str(error));raise
    finally:
        write(out/'decision.json',decision);print(json.dumps(decision),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    try:launch(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        failure={'status':'STOPPED','error':type(error).__name__+': '+str(error),'test_accessed':False}
        write(args.output/'execution-error.json',failure)
        if not (args.output/'decision.json').exists():write(args.output/'decision.json',failure)
        raise
