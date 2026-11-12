"""Fixed-budget EPCLM-006 T4 screening with restartable evidence return."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import torch

from portable_model import BASE,EP4,P,PortableModel
from portability_data import datasets
from cell_artifact import compatibility
from harness import compare,lifecycle,trial,tensors,sync
from preflight import TRAIN,checks,gradient,prepare_restore,fresh_process
from e0 import ROOT,Sampler,load_checkpoint,write
from data import digest,file_hash
from training import fit

FORMAT='minicells.epclm006.portability.v1'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def source_identity():
    modules=('portable_model','portability_data','cell_artifact','preflight','harness','baselines',
             'memory','data','e0','training','cached_transformer','native_clm_runtime','trainer')
    paths=[Path(sys.modules[name].__file__).resolve() for name in modules]
    paths += [BASE/'config.json',Path(__file__),EP4/'m2/configs/protocol.json',
              EP4/'m2/vendor/samba_mamba1.py',EP4/'m2/vendor/selective_scan_ref.py']
    return dict(source_commit=git('rev-parse','HEAD'),
                files={str(p.relative_to(ROOT)):file_hash(p) for p in paths},
                config_sha256=digest([P,TRAIN]),
                tokenizer_sha256=file_hash(EP4/'assets/tokenizer.json'))


def schedule(rows):
    sampler=Sampler(rows,TRAIN['seed'],TRAIN['batch'])
    records=[]
    for _ in range(TRAIN['updates']):
        selected=sampler.propose()
        records.append(dict(ids=[r['id'] for r in selected],inputs=digest([r['input_ids'] for r in selected]),
                            labels=digest([r['labels'] for r in selected])))
        sampler.resolve(True,{'committed_updates':0},{})
    assert len({i for r in records for i in r['ids']})==P['train_episodes']
    return records


def budget(out,validation,deadline):
    path=out/'budget.json'
    if path.exists():
        return json.loads(path.read_text())
    model=PortableModel().cuda()
    torch.cuda.reset_peak_memory_stats()
    sync(model)
    t=time.monotonic()
    gradient(model,validation[:4])
    sync(model)
    update=time.monotonic()-t
    model.eval()
    begin=time.monotonic()
    identity=compatibility(model,'budget-probe-untrained')
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        trial(model,validation[0],identity,Path(tmp)/'artifacts')
    evaluation=time.monotonic()-begin
    estimate=1.5*(P['updates']*update+len(P['evaluation_seeds'])*P['evaluation_episodes_per_seed']*evaluation)+600
    result=dict(status='PASS' if time.monotonic()+estimate<deadline else 'BLOCKED_RESOURCE_LIMIT',
                update_seconds=update,episode_seconds=evaluation,estimated_seconds=estimate,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0)
    write(path,result)
    return result


def new_attempt(parent):
    parent.mkdir(parents=True,exist_ok=True)
    for index in range(100):
        candidate=parent/f'attempt-{index}'
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError('Too many recorded attempts')


def publish(out,decision):
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    assert decision['identity']['source_commit']==git('rev-parse','HEAD')
    for name,sha in json.loads((out/'integrity.json').read_text()).items():
        assert file_hash(out/name)==sha
    with tempfile.TemporaryDirectory(prefix='epclm006-publish-') as tmp:
        root=Path(tmp)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        subprocess.run(['git','remote','set-url','origin','https://github.com/ArcheLabs/mini-cells.git'],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',decision['identity']['source_commit']],cwd=root,check=True)
        stage=root/'results/epclm006';stage.mkdir(parents=True)
        names=[]
        for f in sorted(out.rglob('*')):
            if f.is_file() and f.suffix in ('.json','.jsonl','.md') and f.name not in ('publication.json','error.json'):
                name=f.relative_to(out);target=stage/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(f,target);names.append(str(name))
        spec=ExperimentSpec('results/epclm006','artifacts/experiments/epclm006/'+decision['run_id'],
                            tuple(names),decision['branch'],FORMAT)
        dest=prepare_artifacts(root,'epclm006',spec,None)
        push_results(root,dest,'epclm006',decision['branch'],'GITHUB_TOKEN','origin/main')
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    write(out/'publication.json',dict(status='REMOTE_SHA_VERIFIED',branch=decision['branch'],sha=sha))


def run(out,started):
    out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Actual Tesla T4 required')
    if git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Official execution requires committed source')
    identity=source_identity()
    train,val,banks,data=datasets(identity['tokenizer_sha256'])
    order=schedule(train)
    identity.update(dataset_sha256=digest(data),gpu=torch.cuda.get_device_name(),
                    driver=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=driver_version',
                                                     '--format=csv,noheader'],text=True,timeout=30).strip(),
                    torch=torch.__version__,cuda=torch.version.cuda,precision='fp32',tf32=False)
    header=out/'execution-identity.json'
    if header.exists() and json.loads(header.read_text())!=identity:
        raise ValueError('EPCLM-006 execution identity mismatch')
    write(header,identity)
    write(out/'dataset-manifest.json',data)
    write(out/'training-order.json',order)
    write(out/'execution-config.json',dict(protocol=P,training=TRAIN))
    for device in ('cpu','cuda'):
        proof=out/f'preflight-{device}.json'
        if not proof.exists():
            write(proof,checks(new_attempt(out/f'preflight-{device}'),device))
        assert json.loads(proof.read_text())['status']=='PASS'
    deadline=started+P['execution_seconds']
    if budget(out,val,deadline)['status']!='PASS':
        raise RuntimeError('BLOCKED_RESOURCE_LIMIT')
    model=PortableModel().cuda()
    torch.manual_seed(P['model_seed'])
    fit_result=fit(model,train,val,TRAIN,out/'training',identity,deadline)
    ledger=fit_result['ledger']
    assert fit_result['status']=='COMPLETE'
    assert (ledger['committed_updates'],ledger['input_tokens'],ledger['query_target_tokens'])==(
        P['updates'],P['input_tokens'],P['query_targets'])
    assert [r['input_sha256'] for r in fit_result['history']]==[r['inputs'] for r in order]
    checkpoint=out/'training/checkpoint.pt'
    sha=file_hash(checkpoint)
    assert sha==fit_result['checkpoint']['sha256']
    saved=load_checkpoint(checkpoint)
    assert saved['identity']==identity
    model.load_state_dict(saved['model'],strict=True)
    del saved
    model.eval().requires_grad_(False)
    cell_identity=compatibility(model,sha)
    write(out/'cell-compatibility.json',cell_identity)
    evidence={}
    lifecycle_path=out/'cell-lifecycle.json'
    torch.cuda.reset_peak_memory_stats()
    for seed,rows in banks.items():
        evidence[seed]=[]
        for index,row in enumerate(rows):
            folder=out/'episodes'/seed/str(index)
            result_path=folder/'result.json'
            if result_path.exists():
                record=json.loads(result_path.read_text())
                assert record['id']==row['id'] and record['checkpoint_sha256']==sha
                evidence[seed].append(record)
                continue
            if time.monotonic()>=deadline:
                raise TimeoutError('Portability evaluation deadline')
            attempt=new_attempt(folder)
            try:
                record,expected=trial(model,row,cell_identity,attempt/'artifacts')
                if not lifecycle_path.exists():
                    life=lifecycle(model,row,cell_identity,attempt/'artifacts')
                    prepare_restore(attempt,model,row,cell_identity,expected)
                    life['fresh_process']=fresh_process(attempt,'cuda',False,checkpoint)
                    write(lifecycle_path,life)
                record['checkpoint_sha256']=sha
                record['artifact_dir']=str((attempt/'artifacts').relative_to(out))
                write(result_path,record)
                evidence[seed].append(record)
            except BaseException as error:
                write(attempt/'error.json',dict(id=row['id'],error=repr(error)))
                raise
    assert all(len(rows)==P['evaluation_episodes_per_seed'] for rows in evidence.values())
    analysis=compare(evidence)
    primary=analysis['exact']
    contrasts=[primary[key] for key in ('C1-C0','C1-C2')]
    signal='INCONCLUSIVE' if len(analysis['argmax_counts']['C1'])==1 else (
        'POSITIVE_PORTABILITY_SIGNAL' if all(c['mean']>=P['minimum_gain'] and c['ci95'][0]>0 for c in contrasts)
        else 'NO_POSITIVE_SIGNAL')
    resources=dict(training=fit_result['training_peak_allocated_bytes'],
                   evaluation_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   checkpoint_bytes=checkpoint.stat().st_size,
                   resident_parameters=sum(p.numel() for p in model.parameters()))
    # Decode cost is measured on a continuation of a fresh target only.
    sample=tensors(model,banks[next(iter(banks))][0])
    with torch.no_grad():
        _,runtime=model.backbone.cached(sample['target_ids'][:,:16])
        sync(model)
        start=time.perf_counter()
        model.backbone.cached(sample['target_ids'][:,16:17],runtime)
        sync(model)
    resources['single_token_e3_decode_seconds']=time.perf_counter()-start
    run_id='epclm006-'+digest(identity)[:20]
    decision=dict(format=FORMAT,status='SCREENING_EXECUTED',signal=signal,identity=identity,
                  run_id=run_id,branch='kaggle/'+run_id,checkpoint_sha256=sha,ledger=ledger,
                  training_seeds=1,test_status='SEALED_NOT_ACCESSED')
    write(out/'metrics.json',evidence)
    write(out/'comparison.json',analysis)
    write(out/'resource-profile.json',resources)
    lines=['# EPCLM-006 跨会话 Cell 筛选','',f'状态 {signal}。单训练 Seed。','',
           '| 条件 | Query Recall@1 | Episode Exact Match | NLL |','|---|---:|---:|---:|']
    for c in primary['conditions']:
        lines.append(f"| {c} | {analysis['recall']['conditions'][c]:.2%} | "
                     f"{primary['conditions'][c]:.2%} | {analysis['nll']['conditions'][c]:.4f} |")
    lines+=['','C1−C0 和 C1−C2 的配对区间见 comparison.json。C3 保留创建者上下文。'
            'C2 记录无关 Cell 的召回。任务未定义显式拒答 Token。资源和恢复证据另存。']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    write(out/'decision.json',decision)
    write(out/'integrity.json',{str(f.relative_to(out)):file_hash(f) for f in out.rglob('*')
                                if f.is_file() and f.name not in ('integrity.json','error.json','publication.json')})
    return decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--started',type=float,default=time.monotonic())
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT))
    try:
        path=args.output/'decision.json'
        decision=json.loads(path.read_text()) if path.exists() and (args.output/'integrity.json').exists() else run(args.output,args.started)
        publish(args.output,decision)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'error.json',dict(status='INCOMPLETE',error=repr(error)))
        raise
