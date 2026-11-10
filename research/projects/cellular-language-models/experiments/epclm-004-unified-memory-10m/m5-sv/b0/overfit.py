"""Fixed-sample Overfit only; verified prior Gate A, no automatic Pilot."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'src'))
import torch
from b0_models import construct,state_hash
from b0_data import prepare
from b0_audit import verify_dependencies
from b0_training import fit_stage
from b0_integrity import environment,verify_checkpoint,publish_bundle,publish_index,publisher
from data import digest
from e0 import ROOT,write


def verify_gate_a(protocol,data):
    expected=json.loads((BASE/'configs/gate-a-evidence.json').read_text())['arms'];proof={}
    for arm,item in expected.items():
        subprocess.run(['git','fetch','--quiet','--depth=1','origin',item['index_commit']],cwd=ROOT,check=True,timeout=180)
        fetched=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=ROOT,text=True).strip()
        if fetched!=item['index_commit']:raise ValueError('Gate A index commit differs')
        path='artifacts/experiments/epclm004-m5-sv-b0/'+item['run_id']+'/decision.json'
        record=json.loads(subprocess.check_output(['git','show',item['index_commit']+':'+path],cwd=ROOT))
        m=record['manifest'];artifact=record['artifact']
        if m['arm']!=arm or m['phase']!='gpu-smoke' or m['run_id']!=item['run_id']:raise ValueError('Gate A arm identity differs')
        if record['publication_status']!='VERIFIED' or record['training_status']!='COMPLETED' or artifact['sha256']!=item['bundle_sha256']:
            raise ValueError('Gate A verification failed: '+arm)
        if m['protocol_sha256']!=digest(protocol) or m['dataset_sha256']!=digest(data):raise ValueError('Gate A protocol/data differs')
        proof[arm]=record
    return proof


def overfit_gate(results,arms):
    return all(results.get(arm,{}).get('status')=='OVERFIT_ESTABLISHED' and results[arm].get('publication_status')=='VERIFIED' for arm in arms)


def publish_arm(folder):
    manifest=json.loads((folder/'run-manifest.json').read_text())
    result=json.loads((folder/'result.json').read_text());verify_checkpoint(folder/'training')
    if result['ledger']['committed_updates']!=512:raise ValueError('Cannot publish an incomplete Overfit as completed')
    proof=publish_bundle(folder);proof.update(scientific_status=result['status'],training_ledger=result['ledger'])
    index=publish_index(manifest,proof);return {**proof,'index':index}


def launch(args):
    started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.publish_only:
        for manifest in out.glob('*/run-manifest.json'):publish_arm(manifest.parent)
        return
    p=json.loads((BASE/'configs/protocol.json').read_text());verify_dependencies();banks,data=prepare(out/'datasets',p)
    gate_a=verify_gate_a(p,data);write(out/'gate-a-verification.json',{'status':'PASS','evidence':gate_a,'smoke_reexecuted':False})
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Actual Tesla T4 required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    env=environment();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    identity={'source_commit':source,'protocol_sha256':digest(p),'dataset_sha256':digest(data),
        'software':{k:env[k] for k in ('python','torch','cuda','gpu','driver','precision','tf32','amp')}}
    header=out/'execution-manifest.json'
    if header.exists() and json.loads(header.read_text())!=identity:raise ValueError('RESUME_REJECTED: B0 Overfit identity differs')
    if not header.exists() and Path('/kaggle/input').exists():
        candidates=[]
        for candidate in Path('/kaggle/input').rglob('execution-manifest.json'):
            if json.loads(candidate.read_text())==identity:
                progress=sum(json.loads(m.read_text())['ledger']['committed_updates'] for m in candidate.parent.glob('*/training/metrics.json'))
                candidates.append((progress,str(candidate.parent)))
        if candidates:
            shutil.copytree(Path(max(candidates)[1]),out,dirs_exist_ok=True);prepare(out/'datasets',p)
    write(header,identity);deadline=started+p['execution']['execution_seconds'];results={}
    decision={'training_status':'RUNNING','publication_status':'PENDING','overfit_gate':'NOT_PASSED','pilot_started':False,'test_accessed':False,'arms':results}
    publishing=False
    try:
        for arm in p['arms']:
            folder=out/arm;folder.mkdir(exist_ok=True)
            if (folder/'publication.json').exists():
                results[arm]={**json.loads((folder/'result.json').read_text()),'training_status':'COMPLETED','publication_status':'PENDING'}
                publishing=True
                publishing=True
            proof=publish_arm(folder);results[arm].update(proof);publishing=False;publishing=False;continue
            if (folder/'result.json').exists():
                result=json.loads((folder/'result.json').read_text());verify_checkpoint(folder/'training')
            else:
                if time.monotonic()+300>=deadline:raise TimeoutError('Session budget: resume retained Output')
                model=construct(arm,p).cuda();initial=state_hash(model)
                if initial!=gate_a[arm]['manifest']['initialization_sha256']:raise ValueError('Gate A initialization differs')
                manifest={**identity,'phase':'overfit','arm':arm,'seed':p['seeds']['model'],'initialization_sha256':initial,
                    'gate_a_index_commit':json.loads((BASE/'configs/gate-a-evidence.json').read_text())['arms'][arm]['index_commit']}
                manifest['run_id']='b0-overfit-'+arm.lower()+'-'+digest(manifest)[:20]
                path=folder/'run-manifest.json'
                if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('RESUME_REJECTED: arm identity differs')
                if not (folder/'training/checkpoint.pt').exists():
                    import requests
                    response=requests.get('https://api.github.com/repos/ArcheLabs/mini-cells/releases/tags/'+manifest['run_id'],
                        headers={'Authorization':'Bearer '+publisher().load_github_token('GITHUB_TOKEN')},timeout=(15,60))
                    if response.status_code==200:raise RuntimeError('MISSING_CHECKPOINT_OUTPUT: attach the saved B0 Output; completed training will not be repeated')
                    if response.status_code!=404:response.raise_for_status()
                write(path,manifest)
                if not (folder/'environment.json').exists():write(folder/'environment.json',env)
                write(folder/'protocol.json',p);write(folder/'dataset-manifest.json',data)
                write(folder/'fixed-training-episodes.json',banks['overfit'])
                result=fit_stage(model,banks['overfit'],banks['overfit'],p,folder/'training',manifest,'overfit',deadline)
                del model;torch.cuda.empty_cache()
                if result['status']=='INCOMPLETE_RESOURCE_LIMIT':raise TimeoutError('Partial training saved; no publication freeze yet')
                saved=verify_checkpoint(folder/'training')
                if saved['ledger']['committed_updates']!=512 or saved['ledger']['input_tokens']!=262144 or saved['ledger']['supervised_target_tokens']!=4096:
                    raise ValueError('Overfit budget mismatch')
                write(folder/'result.json',result)
                (folder/'REPORT.md').write_text('# B0 Fixed-Sample Overfit\n\nStatus: '+result['status']+'\n\n512 updates;16 fixed Episodes repeated32 times. No generalization or H5 claim.\n')
            results[arm]={**result,'training_status':'COMPLETED','publication_status':'PENDING'}
            publishing=True
            proof=publish_arm(folder);results[arm].update(proof);publishing=False
            write(out/'decision.json',decision)
        decision.update(training_status='COMPLETED',publication_status='VERIFIED',overfit_gate='PASS' if overfit_gate(results,p['arms']) else 'NOT_ESTABLISHED')
    except Exception as error:
        completed=len(results)==len(p['arms']) and all(r.get('training_status')=='COMPLETED' for r in results.values())
        decision.update(error=f'{type(error).__name__}: {error}',training_status='COMPLETED' if completed else 'INTERRUPTED_OR_PARTIAL',publication_status='FAILED' if publishing else 'PENDING')
        raise
    finally:
        write(out/'decision.json',decision)
        (out/'REPORT.md').write_text('# B0 Overfit execution\n\n'+json.dumps(decision,indent=2)+'\n\nPilot, Test,12M and M6 were not started.\n')
        print({'training_status':decision['training_status'],'publication_status':decision['publication_status'],'overfit_gate':decision['overfit_gate'],'pilot_started':False,'error':decision.get('error')},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--publish-only',action='store_true');launch(parser.parse_args())
