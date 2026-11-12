"""Independent B0 review entry; real experiments require explicit --execute."""
import argparse,json,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'src'))
from b0_audit import verify_dependencies,parameter_audit
from b0_data import prepare
from b0_models import construct
from b0_training import fit_stage
from data import digest
from e0 import write,ROOT


def launch(args):
    p=json.loads((BASE/'configs/protocol.json').read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    verify_dependencies();banks,data=prepare(out/'datasets',p)
    identity={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config_sha256':digest(p),'dataset_sha256':digest(data)}
    run_id='epclm004-m5-sv-b0-'+identity['source_commit'][:12]+'-'+identity['config_sha256'][:8]+'-'+identity['dataset_sha256'][:8]
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('B0 run identity differs')
    write(manifest,{'run_id':run_id,'identity':identity,'test_accessed':False});write(out/'protocol.json',p);write(out/'parameters.json',parameter_audit(p))
    if not args.execute:
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(BASE/'tests')],cwd=ROOT,text=True,capture_output=True)
        (out/'tests.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('B0 engineering tests failed')
        write(out/'decision.json',{'status':'ENGINEERING_REVIEW_COMPLETE_EXPERIMENTS_NOT_STARTED','identity':identity,'test_accessed':False,'hypotheses':'NOT_EVALUATED'})
        return
    raise RuntimeError('LEGACY_COMBINED_ENTRY_DISABLED: use overfit.py with verified prior Gate A evidence; Pilot remains separately gated')
    import torch
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Actual Tesla T4 required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    deadline=time.monotonic()+p['execution']['execution_seconds'];results={}
    for stage in ('overfit','pilot'):
        if stage=='pilot' and not all(results.get('overfit:'+arm,{}).get('status')=='OVERFIT_ESTABLISHED' for arm in p['arms']):
            write(out/'decision.json',{'status':'OVERFIT_NOT_ESTABLISHED','results':results,'pilot_started':False});return
        for arm in p['arms']:
            model=construct(arm,p).cuda()
            train=banks['overfit'] if stage=='overfit' else banks['pilot_train'];val=train if stage=='overfit' else banks['validation']
            result=fit_stage(model,train,val,p,out/stage/arm,{**identity,'arm':arm},stage,deadline)
            results[stage+':'+arm]=result;del model;torch.cuda.empty_cache()
            write(out/'decision.json',{'status':'B0_IN_PROGRESS_REQUIRES_RESEARCH_REVIEW','identity':identity,'results':results,'test_accessed':False})
            if result['status']=='INCOMPLETE_RESOURCE_LIMIT':return
    write(out/'decision.json',{'status':'B0_COMPLETE_REQUIRES_RESEARCH_REVIEW','identity':identity,'results':results,'test_accessed':False,'m6_authorized':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');launch(parser.parse_args())
