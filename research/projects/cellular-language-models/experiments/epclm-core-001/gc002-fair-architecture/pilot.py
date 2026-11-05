"""The approved GPU training entrypoint, separate from prepare.ipynb."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
BASE=Path(__file__).resolve().parent;CORE=BASE.parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
sys.path[:0]=[str(BASE),str(CORE/'gc001-baseline-repair'),str(CORE/'implementation'),str(CORE.parent/'native-clm-v0/implementation'),str(ROOT),str(ROOT/'shared/python')]
import torch
from preparation import write
from cached_transformer import Config
from pilot_training import PilotEngine,verify_resume
from readiness import previous_evidence
from repair_io import recover_inputs,timed_budget
from generation_evaluation import audit_data
from native_clm_runtime import load_tokenizer,sha256_file


def publish(out,report):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    experiment='epclm-gc002-t1-pilot';dest=ROOT/'results'/experiment/'publish';dest.mkdir(parents=True,exist_ok=True)
    write(dest/'decision.json',report);write(dest/'pilot-report.json',report);write(dest/'metrics.json',report.get('metrics',[]));write(dest/'execution-identity.json',report['identity'])
    (dest/'summary.md').write_text('# GC002 T1-PM-KV approved GPU Pilot\n\n'+report['status']+'\n\n'+json.dumps({k:report.get(k) for k in ('training_status','committed_updates','committed_tokens','resume','arithmetic_status','error')},indent=2)+'\n\n50 updates / 12,800 tokens only. No automatic formal training or budget extension. H1–H5 NOT_EVALUATED.\n')
    artifacts=prepare_artifacts(ROOT,experiment,EXPERIMENTS[experiment],None)
    branch='kaggle/'+experiment+'-'+report['identity']['source'][:12]
    push_results(ROOT,artifacts,experiment,branch,'GITHUB_TOKEN','origin/main')
    print(json.dumps({'mode':'APPROVED_T1_PM_KV_GPU_PILOT','status':report['status'],'updates':report.get('committed_updates',0),'tokens':report.get('committed_tokens',0),'publication':'REMOTE_SHA_VERIFIED','branch':branch}),flush=True)


def launch(out):
    from scripts.research.publish_experiment_results import validate_origin,load_github_token
    validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
    if not torch.cuda.is_available():raise RuntimeError('Enable Kaggle GPU: approved Pilot has no CPU training fallback')
    plan=json.loads((BASE/'pilot-protocol.json').read_text())
    if (plan['updates'],plan['training_tokens'],plan['batch'],plan['tbptt'],plan['precision'],plan['resume_check_update'],plan['checkpoint_interval'])!=(50,12800,4,64,'fp32',25,5):raise ValueError('approved Pilot scope changed')
    if plan['optimizer']!={'name':'AdamW','lr':0.0003,'betas':[0.9,0.95],'weight_decay':0.1,'foreach':False} or plan['clip_norm']!=1. or plan['warmup_updates']!=5:raise ValueError('Pilot effective optimizer/scheduler differs from protocol')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old=json.loads((CORE/'gc001-baseline-repair/protocol.json').read_text())['gc_protocol']
    identity={'source':source,'protocol':sha256_file(BASE/'pilot-protocol.json'),'data':old['training_identity']['data'],'seed':plan['seed'],'mode':'GC002_APPROVED_T1_PM_KV_PILOT'}
    out.mkdir(parents=True,exist_ok=True);path=out/'pilot-report.json';checkpoint=out/'private/pilot.pt'
    prior=None
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['identity']!=identity:raise ValueError('local Pilot source differs; prior output retained, no cross-source optimizer reuse')
        if prior.get('training_status')=='COMPLETE':publish(out,prior);return
    report={'format':'minicells.gc002.pilot.v1','status':'INCOMPLETE','training_status':'NOT_STARTED','identity':identity,'protocol':plan,
            'formal_hypotheses':dict.fromkeys(['H1','H2','H3','H4','H5'],'NOT_EVALUATED'),'human_review':'PENDING','automatic_formal_training':False,
            'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'precision':'fp32','autocast':False,'grad_scaler':False}}
    engine=None;started=time.monotonic();stage='VERIFIED_ASSET_RECOVERY'
    try:
        with timed_budget(plan['budget']['execution']):
            print('[GC002 Pilot] Training authorization: T1-PM-KV, FP32, 50 updates, 12,800 tokens only.',flush=True)
            evidence,proof=previous_evidence(ROOT,BASE);report['reused_preparation']=proof
            assets=out.with_name('epclm-gc002-preparation')/'private-input-assets'
            with timed_budget(120):report['input_recovery']=recover_inputs(ROOT,assets,old,[142],time.monotonic()+120,include_weights=False)
            tokenizer=load_tokenizer(assets/'assets/tokenizer.json');groups,audit=audit_data(assets,tokenizer,old)
            if audit!=evidence['baseline-audit.json']['result']['data_audit']:raise ValueError('Pilot data audit differs')
            report['data_audit']=audit;report['tokenizer_sha256']=sha256_file(assets/'assets/tokenizer.json')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if plan['model']!=evidence['parameter-match-report.json']['result']['config']:raise ValueError('Pilot model differs from verified 967808-parameter candidate')
            config=Config(**plan['model']);args=(groups['train'],config,identity,'cuda',plan['seed'],plan['batch'],plan['tbptt'],plan['updates'])
            if prior and prior.get('committed_tokens',0)>0 and not checkpoint.exists():raise ValueError('prior committed Pilot progress has no local checkpoint; refusing fresh retraining')
            engine=PilotEngine.load(checkpoint,*args) if checkpoint.exists() else PilotEngine(*args)
            if engine.ledger.updates>50:raise ValueError('Checkpoint exceeds approved update limit')
            torch.cuda.reset_peak_memory_stats();stage='TRAINING'
            while engine.ledger.updates<50:
                if time.monotonic()-started>plan['budget']['execution']-30:
                    engine.save(checkpoint);raise TimeoutError('Pilot execution budget exhausted; local same-source checkpoint preserved')
                if engine.ledger.updates==25 and engine.resume_result['status']!='PASS':engine=verify_resume(engine,checkpoint)
                row=engine.update();report['training_status']='INCOMPLETE'
                report.update(committed_updates=engine.ledger.updates,committed_tokens=engine.ledger.consumed,metrics=engine.curve,resume=engine.resume_result)
                if row['update'] in (1,2,10,25,50):print(json.dumps({'stage':'REAL_TRAINING_UPDATE',**row}),flush=True)
                if engine.ledger.updates%plan['checkpoint_interval']==0:engine.save(checkpoint)
                write(path,report)
            if engine.resume_result['status']!='PASS' or engine.ledger.consumed!=12800:raise ValueError('Pilot completion conditions not met')
            engine.save(checkpoint)
            unsupported={}
            for row in engine.curve:
                for key,value in row['unsupported_operators'].items():unsupported[key]=unsupported.get(key,0)+value
            report.update(status='COMPLETE' if not unsupported else 'COMPLETE_WITH_ARITHMETIC_LIMITS',training_status='COMPLETE',
                          committed_updates=50,committed_tokens=12800,resume=engine.resume_result,exposure=engine.ledger.report(),
                          arithmetic_status='COVERED_UNDER_DECLARED_OPERATOR_CONVENTION' if not unsupported else 'PARTIAL_NOT_COMPUTE_MATCHED',unsupported_operators=unsupported,
                          registered_arithmetic=sum(sum(r['registered_arithmetic'].values()) for r in engine.curve),formal_compute_matching_ready=False,
                          model_sha256=engine.model.weight_identity(),checkpoint_sha256=sha256_file(checkpoint),
                          peak_vram_bytes=torch.cuda.max_memory_allocated(),training_seconds=engine.training_seconds,checkpoint_seconds=engine.checkpoint_seconds,
                          tokens_per_second=12800/engine.training_seconds,estimated_250k_training_seconds=250000/(12800/engine.training_seconds),
                          estimate_scope='instrumented Pilot throughput only; excludes assets/full evaluation/publication; no budget extension authorized')
    except Exception as error:
        report.update(status='INCOMPLETE',error=type(error).__name__+': '+str(error),failed_stage=stage)
    finally:
        report['execution_seconds']=time.monotonic()-started
        if engine is not None:report.update(committed_updates=engine.ledger.updates,committed_tokens=engine.ledger.consumed,metrics=engine.curve,resume=engine.resume_result)
        write(path,report)
    with timed_budget(plan['budget']['publication']):publish(out,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='APPROVED: real T1-PM-KV GPU Pilot only, 50 updates / 12800 tokens')
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/epclm-gc002-t1-pilot'))
    launch(parser.parse_args().output)
