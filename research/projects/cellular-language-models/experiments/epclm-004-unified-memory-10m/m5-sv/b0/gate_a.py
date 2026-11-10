"""GPU Gate A only: independent Smoke, recovery fixtures and verified binary return."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'src'))
import torch
from b0_models import construct,state_hash
from b0_data import prepare
from b0_audit import verify_dependencies
from b0_training import fit_stage
from b0_integrity import environment,verify_checkpoint,publish_bundle,publish_index
from data import digest,file_hash
from e0 import ROOT,write,load_checkpoint


def equal_tree(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    if hasattr(a,'shape'):return bool((a==b).all())
    return a==b


def recovery_check(folder,p,arm,rows):
    """Actual32+16 update continuation, using tiny architecture, never scientific data budget."""
    fixture=copy.deepcopy(p)
    for config in fixture['models'].values():config.update(vocab=536,dim=16,layers=4,heads=2,kv_heads=2,ffn=32)
    fixture['memory'].update(state_size=4,summary_tokens=2)
    fixture['training']['updates']=48;fixture['overfit']['evaluation_every']=16
    identity={'scope':'independent recovery fixture','arm':arm};device='cuda' if torch.cuda.is_available() else 'cpu'
    a=construct(arm,fixture).to(device);fit_stage(a,rows,rows,fixture,folder/'continuous',identity,'overfit',stop_after=32)
    fit_stage(a,rows,rows,fixture,folder/'continuous',identity,'overfit')
    b=construct(arm,fixture).to(device);fit_stage(b,rows,rows,fixture,folder/'restart',identity,'overfit',stop_after=32)
    del b
    b=construct(arm,fixture).to(device);fit_stage(b,rows,rows,fixture,folder/'restart',identity,'overfit')
    x=verify_checkpoint(folder/'continuous');y=verify_checkpoint(folder/'restart')
    checks={key:equal_tree(x[key],y[key]) for key in ('model','optimizer','sampler','rng','ledger')}
    checks['data_order']=[h['input_sha256'] for h in x['history']]==[h['input_sha256'] for h in y['history']]
    checks['losses']=[h['loss'] for h in x['history']]==[h['loss'] for h in y['history']]
    if not all(checks.values()):raise ValueError('RECOVERY_NOT_BITWISE: no equivalence claim or automatic tolerance relaxation')
    return {'status':'PASS','comparison':'BITWISE_IDENTICAL','checks':checks,'updates':48,'architecture':'tiny engineering fixture'}


def launch(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.publish_only:
        for manifest in out.glob('*/run-manifest.json'):
            proof=publish_bundle(manifest.parent);publish_index(json.loads(manifest.read_text()),proof)
        return
    p=json.loads((BASE/'configs/protocol.json').read_text());verify_dependencies();banks,data=prepare(out/'development-data',p)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('BLOCKED_HARDWARE: actual Tesla T4 required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    env=environment();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    identity={'source_commit':source,'protocol_sha256':digest(p),'dataset_sha256':digest(data),'environment_sha256':digest({k:env[k] for k in ('python','torch','cuda','gpu','driver','precision','tf32','amp','deterministic_algorithms')}),
        'dependency_lock_sha256':file_hash(BASE/'configs/dependency-lock.json')}
    decision={'training_status':'RUNNING','publication_status':'PENDING','gate_a':'NOT_PASSED','overfit_started':False,'pilot_started':False,'test_accessed':False,'arms':{}}
    try:
        deadline=time.monotonic()+p['execution']['execution_seconds']
        for arm in p['arms']:
            if time.monotonic()+300>=deadline:raise RuntimeError('SESSION_LIMIT: resume existing Gate A Output')
            folder=out/arm;folder.mkdir(exist_ok=True)
            if (folder/'publication.json').exists():
                proof=publish_bundle(folder);index=publish_index(json.loads((folder/'run-manifest.json').read_text()),proof);decision['arms'][arm]={**proof,'index':index};continue
            model=construct(arm,p).cuda();manifest={**identity,'phase':'gpu-smoke','arm':arm,'seed':p['seeds']['model'],
                'initialization_sha256':state_hash(model),'runtime':'new Episode per update; no persisted cross-Episode Store',
                'scheduler_state':'constant LR, no mutable scheduler','gradient_accumulation':'none'}
            manifest['run_id']='b0-gpu-smoke-'+arm.lower()+'-'+digest(manifest)[:20]
            path=folder/'run-manifest.json'
            if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('RESUME_REJECTED: Smoke identity differs')
            write(path,manifest)
            if not (folder/'environment.json').exists():write(folder/'environment.json',env)
            write(folder/'protocol.json',p);write(folder/'dataset-manifest.json',data)
            smoke=copy.deepcopy(p);smoke['training']['updates']=1;smoke['overfit']['evaluation_every']=1
            fit_stage(model,banks['overfit'][:1],banks['overfit'][:1],smoke,folder/'smoke',manifest,'overfit',deadline)
            saved=verify_checkpoint(folder/'smoke');restored=construct(arm,p).cuda();restored.load_state_dict(saved['model'])
            if state_hash(model)!=state_hash(restored):raise ValueError('Smoke model restore differs')
            del model,restored;torch.cuda.empty_cache()
            recovery_path=folder/'recovery.json'
            recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else recovery_check(folder/'recovery',p,arm,banks['overfit'][:1])
            write(recovery_path,recovery)
            write(folder/'training-status.json',{'training_status':'COMPLETED','smoke_updates':1,'smoke_targets':8,'recovery_fixture':recovery,
                'not_applicable':{'scientific_validation':'Smoke only','test':'not generated or accessed','scheduler':'constant, no state'}})
            proof=publish_bundle(folder);index=publish_index(manifest,proof);decision['arms'][arm]={**proof,'index':index}
        decision.update(training_status='COMPLETED',publication_status='VERIFIED',gate_a='PASS')
    except Exception as error:
        decision.update(error=f'{type(error).__name__}: {error}',publication_status='FAILED_OR_PENDING',gate_a='NOT_PASSED')
        raise
    finally:write(out/'gate-a.json',decision);print(decision,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--publish-only',action='store_true');launch(parser.parse_args())
