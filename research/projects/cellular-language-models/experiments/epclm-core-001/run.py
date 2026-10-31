"""Single development entrypoint; no real training in the CPU preflight."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
sys.path[:0]=[str(BASE/'implementation'),str(BASE/'tests'),str(ROOT),str(ROOT/'shared/python'),str(BASE.parent/'native-clm-v0/implementation')]
from model import Core,Config
from runtime import Runtime,weight_hash
from trainer import train,evaluate,paired,atomic_save
from data import prepare
from diagnostics import pilot,inspector,history_test
from storage import bundle,recover
from native_clm_runtime import sha256_file,load_tokenizer

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def launch(output,device,preflight_only=False):
    from test_core import test_contracts
    test_contracts()
    output.mkdir(parents=True,exist_ok=True)
    if preflight_only:
        write(output/'preflight.json',{'engineering':'CPU_CONTRACTS_PASS','data_assets':'NOT_PREPARED','gpu':'NOT_EVALUATED','protocol_sha256':sha256_file(BASE/'protocol.json'),'scientific_gates':'NOT_EVALUATED'})
        return
    if device!='cuda' or not torch.cuda.is_available(): raise RuntimeError('enable Kaggle GPU')
    from scripts.research.publish_experiment_results import load_github_token,validate_origin
    validate_origin(ROOT); load_github_token('GITHUB_TOKEN')
    config=json.loads((BASE/'protocol.json').read_text())
    deadline=time.monotonic()+5400
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    recover(ROOT,output,source,sha256_file(BASE/'protocol.json'))
    manifest=prepare(output/'assets',config['development_stories'])
    identity={'source':source,'protocol':sha256_file(BASE/'protocol.json'),
              'data':sha256_file(output/'assets/manifest.json'),'seed':config['seed']}
    docs={split:json.loads((output/'assets'/f'{split}.json').read_text()) for split in ('train','development')}
    report={'format':'minicells.epclm-core-001.v1','status':'INCOMPLETE','identity':identity,
            'formal_confirmation':False,'scientific_decision':False,'hypotheses':dict.fromkeys(['H1','H2','H3','H4','H5'],'NOT_EVALUATED'),'arms':{}}
    previous={}
    saved_report=output/'report.json'
    if saved_report.exists():
        previous=json.loads(saved_report.read_text())
        if previous['identity']!=identity: raise ValueError('saved report identity mismatch')
        if previous['status']=='DEVELOPMENT_COMPLETE': report=previous
    arms=() if report['status']=='DEVELOPMENT_COMPLETE' else ('C1','F0','C0','L0')
    for arm in arms:
        if time.monotonic()>deadline-60: break
        torch.manual_seed(config['seed']); c=Config(arm=arm); model=Core(c).to(device)
        pilot_result=pilot(model) if not (output/f'{arm}.pt').exists() else previous.get('arms',{}).get(arm,{}).get('gpu_pilot',{'status':'REUSED_CHECKPOINT'})
        arm_identity={**identity,'model':asdict(c),'target_tokens':config['pilot_tokens']}
        try:
            result=train(model,docs['train'],arm_identity,output/f'{arm}.pt',config['pilot_tokens'],deadline,config['seed'],on_checkpoint=lambda:bundle(output,identity))
        except (FloatingPointError,RuntimeError) as error:
            report['arms'][arm]={'status':'STOPPED_NUMERICAL_FAILURE','error':str(error)}
            break
        result.update({'gpu_pilot':pilot_result,'parameters':sum(p.numel() for p in model.parameters()),'state_bytes':model.init_state(1).numel()*4,
                       'weights_sha256':weight_hash(model),'model':asdict(c),'kernel_launch_count':pilot_result.get('kernel_launch_count','NOT_MEASURED'),
                       'model_asset':{'storage':'PRIVATE_KAGGLE_OUTPUT','path':f'epclm-core-001/{arm}.pt','sha256':sha256_file(output/f'{arm}.pt'),'saved_version':'PENDING_SAVE'}})
        try:
            if result['status']=='COMPLETE' and time.monotonic()<deadline-60:
                result['validation']=evaluate(model,docs['development'][:16],deadline=deadline)
                if arm=='C1':
                    result['ablations']={name:evaluate(model,docs['development'][:4],ablation=name,deadline=deadline) for name in ('reset','swap','communication','entry')}
                    result['k_sensitivity']={str(k):evaluate(model,docs['development'][:4],steps=k,deadline=deadline) for k in (2,4,8)}
                    write(output/'inspector.json',inspector(model,output))
                    result['history_probe']=history_test(model,load_tokenizer(output/'assets/tokenizer.json'))
                    write(output/'generation.json',{'token_ids':Runtime(model).generate([2],32)})
                    atomic_save(Runtime(model).snapshot(model.init_state(1)),output/'inference-snapshot.pt')
        except (TimeoutError,FloatingPointError) as error:
            result['status']='INCOMPLETE'
            result['error']=str(error)
        result['training_status']=result['status']
        if result['status']=='COMPLETE' and 'validation' not in result: result['status']='INCOMPLETE_VALIDATION'
        report['arms'][arm]=result
        write(output/'report.json',report)
    if len(report['arms'])==4 and all(v['status']=='COMPLETE' and 'validation' in v for v in report['arms'].values()):
        report['status']='DEVELOPMENT_COMPLETE'
        report['paired']={arm:paired(report['arms']['C1']['validation'],report['arms'][arm]['validation']) for arm in ('C0','F0','L0')}
    write(output/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axis=plt.subplots()
    for arm,result in report['arms'].items():
        curve=result.get('curve',[])
        if curve: axis.plot([v['tokens'] for v in curve],[v['nll'] for v in curve],label=arm)
    axis.set(xlabel='Training tokens',ylabel='Training NLL'); axis.legend()
    figure.savefig(output/'learning-curves.png'); plt.close(figure)
    bundle(output,identity)
    # Explicit metadata-only projection; raw stories and weights stay in private Kaggle Output.
    publication=ROOT/'results/epclm-core-001/publish'
    def public(value):
        if isinstance(value,dict): return {k:public(v) for k,v in value.items() if k not in ('stories','trace','history_probe')}
        if isinstance(value,list): return [public(v) for v in value]
        return value
    write(publication/'decision.json',public(report))
    (publication/'summary.md').write_text('# EPCLM Core-001 development\n\n'+report['status']+'\n\nAll scientific hypotheses remain NOT EVALUATED pending independent confirmation.\n')
    from scripts.research.publish_experiment_results import prepare_artifacts,push_results,EXPERIMENTS
    destination=prepare_artifacts(ROOT,'epclm-core-001',EXPERIMENTS['epclm-core-001'],None)
    push_results(ROOT,destination,'epclm-core-001','kaggle/epclm-core-001-'+source[:12],'GITHUB_TOKEN','origin/main')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path('/kaggle/working/epclm-core-001'))
    parser.add_argument('--device',default='cuda'); parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(); launch(args.output,args.device,args.preflight_only)
