"""Bounded R-line development calibration, never formal confirmation."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'scripts/research/publish_experiment_results.py').exists())


def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def publish(out,decision):
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results,remote_branch_sha
    key='m3c-002b-cal-001';spec=EXPERIMENTS[key];staging=ROOT/spec.source_dir
    if staging.exists():shutil.rmtree(staging)
    staging.mkdir(parents=True);names=[]
    for f in out.rglob('*'):
        if '_recovery_objects' in f.parts or not f.is_file() or f.suffix not in ('.json','.csv','.md'):continue
        name=f.relative_to(out).as_posix();target=staging/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target);names.append(name)
    script_id=decision.get('execution',{}).get('script_version_id')
    dest=prepare_artifacts(ROOT,key,replace(spec,files=tuple(sorted(names))),script_id)
    branch='kaggle/'+decision['run_id'];push_results(ROOT,dest,key,branch,'GITHUB_TOKEN','origin/main')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert remote_branch_sha(ROOT,branch)==commit,'publication remote SHA mismatch'
    from train import write
    write(out/'publication-verification.json',{'status':'REMOTE_SHA_VERIFIED','branch':branch,'sha':commit})
    print({'status':decision['status'],'publication':'REMOTE_SHA_VERIFIED','branch':branch},flush=True)


def launch(args):
    from train import write,file_sha
    started=args.started or time.monotonic();p=json.loads((BASE/'protocol.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    out=args.output/('m3c-002b-cal-001-'+source[:12]);out.mkdir(parents=True,exist_ok=True)
    if args.publish_only:
        publish(out,json.loads((out/'decision.json').read_text()));return
    decision={'format':'minicells.m3c-002b-cal-001.v1','run_id':out.name+'-'+uuid.uuid4().hex[:8],
         'source_commit':source,'status':'INCOMPLETE','type':'DEVELOPMENT_CALIBRATION','formal_training_targets':0,
         'formal_test_accessed':False,'natural_language_ppl':'NOT_RUN','scientific_hypotheses':'NOT_EVALUATED',
         'model_seed_count':1,'model_seed':p['model_seed'],'training_targets_per_arm':p['training']['targets_per_arm'],
         'development_targets_committed':{},'training_seed_variance':'NOT_ESTIMABLE_SINGLE_SEED','execution':None}
    reports={};profiles={};stage='SOURCE'
    try:
        lock=json.loads((BASE/'source-lock.json').read_text())
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.candidate,text=True).strip()==lock['commit']
        for name,digest in lock['files'].items():assert file_sha(args.candidate/name)==digest,'Reference source mismatch: '+name
        write(out/'source-lock.json',lock);write(out/'protocol.json',p)
        prerequisite=json.loads((BASE/'prerequisite.json').read_text());assert prerequisite['engineering_status']=='WORKSPACE_ENGINEERING_PASS'
        write(out/'prerequisite.json',prerequisite)
        import torch
        stage='HARDWARE'
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; calibration training NOT_RUN')
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sys.path.insert(0,str(args.candidate.resolve()));sys.path.insert(0,str(BASE.parent/'M3C-002'))
        import mamba3
        assert Path(mamba3.__file__).resolve().is_relative_to(args.candidate.resolve())
        from identity import capture
        tag='M3C-002B-CAL-001:'+sha(p)
        stage='EXECUTION_IDENTITY'
        execution=capture(ROOT,p['kaggle_kernel'],tag)
        decision['execution']=execution;write(out/'execution-identity.json',execution)
        write(out/'environment.json',{'python':sys.version,'torch':str(torch.__version__),'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
              'packages':sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions())})
        stage='DATA'
        from data import build,validate
        train=build('train-dev',p['training']['episodes_per_arm'],p['data_seed'])
        dev=build('validation-dev',p['development_evaluation']['episodes'],p['data_seed'])
        data=validate(train,dev);write(out/'data-manifest.json',data)
        np.savez_compressed(out/'development-assets.npz',train_ids=train[0],train_labels=train[1],dev_ids=dev[0],dev_labels=dev[1])
        identity=sha({'source':source,'protocol':p,'data':data,'reference':lock})
        # Existing bounded recovery and immutable object transport; no new recovery validation.
        sys.path.insert(1,str(BASE.parent.parent/'clm-workspace-controller-001'))
        from recovery import restore,commit_bundle
        from dependencies import shared
        stage='RECOVERY'
        if not (out/'manifest.json').exists():restore(out,identity,min(started+300,time.monotonic()+45),artifact_dir='artifacts/experiments/m3c-002b-cal-001')
        if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())['identity']==identity,'incompatible continuation'
        else:write(out/'manifest.json',{'identity':identity,'source':source,'protocol':p,'data':data})
        manifest=json.loads((out/'manifest.json').read_text());sessions=manifest.setdefault('execution_sessions',[])
        if not any(row['script_version_id']==execution['script_version_id'] for row in sessions):sessions.append(execution)
        write(out/'manifest.json',manifest)
        if (out/'decision.json').exists():
            old=json.loads((out/'decision.json').read_text())
            if old.get('status')=='DEVELOPMENT_CALIBRATION_COMPLETE':publish(out,old);return
        from mamba3 import Mamba3Config
        # Legacy recovery imports a different generic 'model' module; bind Workspace explicitly.
        spec=importlib.util.spec_from_file_location('m3c002_workspace',BASE.parent/'M3C-002/model.py')
        workspace=importlib.util.module_from_spec(spec);spec.loader.exec_module(workspace);Model=workspace.Model
        sys.path.insert(0,str(BASE))
        from train import fit,paired_analysis
        from measure import measure
        cfg=Mamba3Config(**p['model']);reports={};initial_hash=None
        stage='TRAINING'
        for arm,window in p['arms'].items():
            model=Model(cfg,window=window,heads=p['workspace_heads'],seed=p['model_seed'],device='cuda')
            h=hashlib.sha256()
            for name,v in model.base.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
            if initial_hash is None:initial_hash=h.hexdigest()
            assert h.hexdigest()==initial_hash,'public parameter initialization differs'
            folder=out/arm;reports[arm]=fit(model,train,dev,p,folder,identity+':'+arm,
                started+p['budget']['execution_seconds'],shared.native._atomic_save,
                lambda path:commit_bundle(out,identity,path))
            decision['development_targets_committed'][arm]=reports[arm]['targets']
            del model;torch.cuda.empty_cache()
        # All training has exited; profiling cannot compete with another training worker.
        profiles={};stage='TRAINED_EVALUATION'
        for arm,window in p['arms'].items():
            if time.monotonic()>=started+p['budget']['evaluation_seconds']:raise TimeoutError('evaluation budget reached')
            model=Model(cfg,window=window,seed=p['model_seed'],device='cuda')
            saved=torch.load(out/arm/'checkpoint.pt',map_location='cuda',weights_only=False)
            assert saved['identity']==identity+':'+arm and saved['progress']['targets']==p['training']['targets_per_arm']
            model.load_state_dict(saved['model']);profiles[arm]=measure(model,dev[0][2:3]);del model,saved;torch.cuda.empty_cache()
        write(out/'trained-inference.json',profiles)
        differences=paired_analysis(out,list(p['arms']),p['development_evaluation']['bootstrap_samples'],p['development_evaluation']['bootstrap_seed'])
        write(out/'paired-development-analysis.json',differences)
        decision.update(status='DEVELOPMENT_CALIBRATION_COMPLETE',initial_common_parameter_sha256=initial_hash,
                        committed_development_targets=sum(r['targets'] for r in reports.values()),session_seconds=time.monotonic()-started)
    except Exception as error:
        # Do not spill credential/API response payloads; keep typed failure and local checkpoints.
        decision['stage']=stage
        if stage=='EXECUTION_IDENTITY':decision['status']='BLOCKED_EXECUTION_IDENTITY'
        elif stage=='HARDWARE':decision['status']='BLOCKED_GPU_UNAVAILABLE'
        decision['error']=type(error).__name__+': '+str(error)[:800]
        print({'stage':'CALIBRATION_STOP','failed_stage':stage,'error':decision['error']},flush=True)
        for path in out.glob('M3-*/progress.json'):
            try:decision['development_targets_committed'][path.parent.name]=json.loads(path.read_text())['targets']
            except Exception:pass
    write(out/'decision.json',decision)
    lines=['# M3C-002B-CAL-001 开发校准','', '状态：'+decision['status'],
           '', 'R线MQAR/Repeated Retrieval，单开发Seed27101；正式训练/测试/语言PPL未运行，科学假设NOT_EVALUATED。',
           '单Seed只提供Episode抽样波动，不能估计跨训练Seed方差。', '',
           '| Arm | Targets | Initial Query NLL | Final Query NLL | Dev EM | Repeat first/later EM | Train s | Targets/s | Update P95 s | Decode ms/token |',
           '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for arm,r in reports.items():
        progress=json.loads((out/arm/'progress.json').read_text());m=r['metrics'];initial=progress['evaluations']['0']['all']['query_nll']
        lines.append(f"| {arm} | {r['targets']} | {initial:.4f} | {m['all']['query_nll']:.4f} | {m['all']['episode_mean_em']:.4f} | {m['repeat_first']['query_em']:.4f}/{m['repeat_later']['query_em']:.4f} | {r['training_seconds']:.2f} | {r['query_targets_per_second']:.2f} | {r['update_seconds_p95']:.4f} | {profiles.get(arm,{}).get('decode_ms_per_token_p50','PENDING')} |")
    lines.extend(['','训练时间/TPS仅训练计算与优化器更新，环境准备、传输、评估、保存及发布另计；输入Token含Padding单独记账。',
                  'Gap和实际Key可见性条件见各臂result.json；逐查询预测见predictions.csv；去除事实Value的负控制用于排查任务捷径，不作为额外训练组。',
                  '置信区间仅开发Episode配对抽样，不能用于宣称H-R1/H-R2成立。不得据开发结果自动扩大训练预算或进入M3C-003。'])
    (out/'RESEARCH-REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    publish(out,decision)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--started',type=float);parser.add_argument('--publish-only',action='store_true')
    launch(parser.parse_args())
