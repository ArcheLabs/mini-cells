"""Registered manual execution entrypoint. Gate checks precede all external reads."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import torch
from .config import load,authorize,blockers,ROOT,BASE
from .checkpoint import identity,file_sha256
from .assets import prepare,prepare_confirmation,load_assets
from .preflight import run
from .training import train
from .baseline import make_model
from .profiling import benchmark,parity
from .reuse import make_batch,write_json

def verify_source(source):
    subprocess.run(['git','merge-base','--is-ancestor',source,'HEAD'],cwd=ROOT,check=True)
    paths=[str(BASE.relative_to(ROOT)/p) for p in ('implementation','config','protocol.json','registration')]+['experiments/common','research/08-hybrid-clm/experiments/_shared/common']
    subprocess.run(['git','diff','--exit-code',source,'HEAD','--',*paths],cwd=ROOT,check=True,stdout=subprocess.PIPE)
    subprocess.run(['git','diff','--exit-code','HEAD','--',*paths],cwd=ROOT,check=True,stdout=subprocess.PIPE)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['preflight','dry-run','prepare-assets','train','profile','gpu-micro-preflight']);ap.add_argument('--config');ap.add_argument('--arm',choices=['R0','R1','R2','M0']);ap.add_argument('--seed',type=int);ap.add_argument('--role',default='development',choices=['development','confirmation']);ap.add_argument('--source-commit');ap.add_argument('--assets');ap.add_argument('--output');ap.add_argument('--source-parquet');ap.add_argument('--source-sha256');ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    c,p,_=load(args.config)
    if args.mode=='preflight':print(json.dumps({**run(c),'registration_blockers':blockers(c,p)},indent=2));return
    if args.mode=='dry-run':print(json.dumps({'experiment':'006-path-first-clm','status':p['status'],'blockers':blockers(c,p),'gpu_started':False,'confirmation_read':False},indent=2));return
    if args.mode=='gpu-micro-preflight':
        from .gpu_micro import run as gpu_micro
        if args.assets or args.seed is not None or args.role!='development':raise RuntimeError('GPU micro mode accepts no corpus, experiment seed or confirmation role')
        if args.source_commit:verify_source(args.source_commit)
        result=gpu_micro(c,args.output,args.device,args.source_commit)
        print(json.dumps({'status':result['status'],'scientific_decision':False,'optimizer_steps':0},indent=2));return
    if args.mode=='prepare-assets':
        if args.role=='confirmation':
            authorize(c,p,args.seed,'confirmation',require_confirmation_manifest=False)
            if not all([args.output,args.source_parquet,args.assets]):ap.error('confirmation preparation requires output, frozen development assets and pinned source')
            result=prepare_confirmation(c,p,args.seed,args.output,args.assets,args.source_parquet)
        else:
            if not all([args.output,args.source_parquet,args.source_sha256]):ap.error('prepare-assets requires output and pinned source file/hash')
            result=prepare(c,args.output,args.source_parquet,args.source_sha256)
        print(json.dumps(result,indent=2));return
    authorize(c,p,args.seed,args.role)
    if not all([args.arm,args.source_commit,args.assets,args.output]):ap.error('train/profile require arm, source, assets and output')
    verify_source(args.source_commit)
    if args.mode=='profile':
        record=json.loads((Path(args.output)/'metrics.json').read_text())
        if record['status']!='completed' or record['steps']!=c['training']['steps']:raise RuntimeError('profiling requires the frozen final checkpoint before evaluation assets are opened')
    assets=load_assets(args.assets,c,p,args.seed,args.role,include_evaluation=args.role=='development' or args.mode=='profile')
    run_id=identity(args.source_commit,c,p,assets['manifest_sha256'],args.arm,args.seed,args.role)
    if args.mode=='train':
        deferred=(lambda:load_assets(args.assets,c,p,args.seed,args.role,include_evaluation=True)) if args.role=='confirmation' else None
        train(c,run_id,assets,args.output,args.device,final_assets_loader=deferred);return
    output=Path(args.output)
    try:
        output=Path(args.output);m=json.loads((output/'metrics.json').read_text());checkpoint=output/'latest.pt'
        if m['identity']!=run_id or m['status']!='completed' or file_sha256(checkpoint)!=m['checkpoint_sha']:raise ValueError('profile checkpoint identity mismatch')
        if (output/'decode-profile.json').exists():raise FileExistsError('decode evidence is immutable')
        model=make_model(c['model'],args.arm,args.seed).to(args.device);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['identity']!=run_id:raise ValueError('profile identity mismatch')
        model.load_state_dict(payload['model']);model.eval();x,_=make_batch(assets[args.role],assets['eval_starts'][:1],c['training']['sequence_length'],torch.device(args.device))
        check=parity(model,x,c['evaluation']['parity_atol'],c['evaluation']['parity_rtol'])
        if not check['passed']:raise RuntimeError('decode parity failed')
        profile=benchmark(model,x,c);write_json(output/'decode-profile.json',{'identity':run_id,'checkpoint_sha':m['checkpoint_sha'],'parity':check,'profile':profile})
    except BaseException as exc:
        write_json(output/'failure.json',{'identity':run_id,'status':'INCOMPLETE','scientific_decision':False,
            'phase':'isolated_profile','error_type':type(exc).__name__,'reason':str(exc),
            'completed_checks':['training checkpoint available']})
        raise
if __name__=='__main__':main()
