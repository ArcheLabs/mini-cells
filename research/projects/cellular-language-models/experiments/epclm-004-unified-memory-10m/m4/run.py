"""One launch: new E5 CPU preflight -> two-update T4 Smoke -> verified publication."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import torch
from fusion import ThreeWayFusion
from checks import correctness
from e0 import ROOT, BASE as M1, write, load_checkpoint
from data import digest, file_hash, MQARConfig, episode, dataset_manifest, batch
from training import fit
from gates import recall_control

BASE=Path(__file__).resolve().parent


def identity():
    return {str(p.relative_to(BASE)):file_hash(p) for p in sorted(BASE.glob('*.py'))}


def verify_frozen():
    lock=json.loads((BASE/'configs/source-lock.json').read_text())
    for name,sha in lock['files'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('frozen source changed: '+name)


def preflight(out,protocol):
    key=digest({'source':identity(),'protocol':protocol,'python':sys.version,'torch':torch.__version__})
    report=Path(out)/'preflight.json'
    if report.exists():
        result=json.loads(report.read_text())
        if result['identity']==key and result['status']=='PASS':
            for name,sha in result['artifact_sha256'].items():
                if file_hash(Path(out)/name)!=sha:raise ValueError('passed preflight artifact differs: '+name)
            return result
    fixture=Path(out)/'fixtures'/key[:16]
    result=correctness(fixture);result['identity']=key
    result['artifact_sha256']={str(p.relative_to(out)):file_hash(p) for p in fixture.rglob('*') if p.is_file()}
    write(report,result)
    return result


def gpu(out,protocol,run_identity,deadline):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('BLOCKED_GPU_ACCESS: actual Tesla T4 required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);finished=out/'smoke.json'
    if finished.exists():
        old=json.loads(finished.read_text())
        if old['identity']!=run_identity:raise ValueError('GPU output identity differs')
        if old['status']=='PASS':
            for name,sha in old['files'].items():
                if file_hash(out/name)!=sha:raise ValueError('completed GPU artifact differs: '+name)
            return old
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(protocol['training']['seed']);torch.cuda.manual_seed_all(protocol['training']['seed'])
    model=ThreeWayFusion(protocol['model'],**protocol['fusion']).cuda()
    params=model.resources();assert abs(params['static_parameters']/10000000-1)<.01
    cfg=MQARConfig(**protocol['mqar']);groups={k:[episode(cfg,k,i) for i in range(n)] for k,n in protocol['episodes'].items()}
    manifest=dataset_manifest(protocol['mqar'],groups,protocol['tokenizer_sha256']);write(out/'dataset-manifest.json',manifest)
    training_identity={**run_identity,'dataset':manifest,'model_id':model.model_id}
    event_file=out/'events.json';events=json.loads(event_file.read_text()) if event_file.exists() else []
    try:metrics=fit(model,groups['train'],groups['validation'],protocol['training'],out/'training',training_identity,deadline)
    finally:
        events+=model.events;write(event_file,events)
    assert metrics['status']=='COMPLETE' and metrics['ledger']['committed_updates']==2
    indexed={e['input_sha256']:e for e in events};committed=[indexed[r['input_sha256']] for r in metrics['history']]
    assert len(committed)==2 and all(e['fast_delta']>0 and e['momentum_norm']>0 for e in committed)
    metrics['parameters']=params;write(out/'training/metrics.json',metrics)
    saved=load_checkpoint(out/'training/checkpoint.pt')
    restored=ThreeWayFusion(protocol['model'],**protocol['fusion']).cuda();restored.load_state_dict(saved['model'])
    opt=torch.optim.AdamW(restored.parameters(),lr=protocol['training']['lr']);opt.load_state_dict(saved['optimizer'])
    assert all(int(v['step'])==2 for v in opt.state.values())
    probe=batch(groups['validation'][:1],'cuda').input_ids;model.eval();restored.eval()
    with torch.no_grad():restore_error=float((model(probe)-restored(probe)).abs().max())
    assert restore_error==0 and saved['ledger']==metrics['ledger'] and saved['identity']==training_identity
    del restored,opt,saved;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    model.set_mode('online_adapt').eval();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    with torch.no_grad():
        expected,cell,g,_=model.forward_state(probe);torch.cuda.synchronize();full_seconds=time.perf_counter()-started
        runtime=None;carried=None;parts=[]
        for part in probe.split(protocol['fusion']['chunk'],1):
            y,carried,_,runtime=model.forward_state(part,carried,runtime=runtime,cache=True);parts.append(y)
        error=float((torch.cat(parts,1)-expected).abs().max());assert error<2e-4
        for name in cell.fast:torch.testing.assert_close(cell.fast[name],carried.fast[name],atol=2e-4,rtol=2e-4)
        resources=model.resources(carried,runtime);gate_mean=g.mean((0,1)).cpu().tolist()
        ablations={}
        for disabled in (('swa',),('ssm',),('memory',)):
            torch.cuda.synchronize();start=time.perf_counter()
            y,c,w,_=model.forward_state(probe,carried,disabled=disabled)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            assert torch.isfinite(y).all() and w[...,model.branch_names.index(disabled[0])].eq(0).all()
            if 'memory' in disabled:assert c is carried
            ablations[disabled[0]]={'gate_mean':w.mean((0,1)).cpu().tolist(),'seconds':elapsed}
    resources.update(training_peak_allocated_bytes=metrics['training_peak_allocated_bytes'],
        inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),full_forward_seconds=full_seconds,
        timing_scope='single engineering probe, no formal throughput claim')
    model.reset_runtime();recall=recall_control(out,'cuda',model.memory,context={
        'backbone_checkpoint':file_hash(out/'training/checkpoint.pt'),'tokenizer':protocol['tokenizer_sha256'],
        'model_id':model.model_id,'source':run_identity['source_commit']})
    result={'identity':run_identity,'status':'PASS','hardware':{'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'fp32','tf32':False},
        'training':metrics,'events':committed,'inner_batch_updates':sum(e['inner_updates'] for e in committed),
        'checkpoint_restore_error':restore_error,'full_vs_chunk_cache_error':error,'gate_mean':gate_mean,
        'resources':resources,'ablations':ablations,'cross_session_memory':recall,'hypotheses':'NOT_EVALUATED'}
    result['files']={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p!=finished}
    write(finished,result);return result


def publish(out,decision):
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    # Same isolated-checkout pattern verified in M3; never switch the source checkout.
    origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory(prefix='epclm004-m4-publish-') as tmp:
        root=Path(tmp)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',decision['identity']['source_commit']],cwd=root,check=True)
        stage=root/'results/epclm004-m4/publish';stage.mkdir(parents=True);names=[]
        for p in Path(out).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.md') and p.name!='publication.json':
                name=str(p.relative_to(out));target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,target);names.append(name)
        hashes={name:file_hash(stage/name) for name in names};write(stage/'publication-files.json',hashes);names.append('publication-files.json')
        destination=root/'artifacts/experiments/epclm004-m4'/decision['run_id']
        spec=ExperimentSpec(str(stage.relative_to(root)),str(destination.relative_to(root)),tuple(names),decision['branch'],decision['format'])
        folder=prepare_artifacts(root,'epclm004-m4',spec,decision.get('script_version_id'))
        push_results(root,folder,'epclm004-m4',decision['branch'],'GITHUB_TOKEN','origin/main')
        for name,sha in hashes.items():assert file_hash(destination/name)==sha
        receipt={'status':'REMOTE_SHA_VERIFIED','branch':decision['branch'],
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'files':hashes}
    write(out/'publication.json',receipt);return receipt


def report(out,decision):
    cpu=decision.get('preflight',{});gpu_result=decision.get('gpu',{})
    text='\n'.join(['# EPCLM-004 M4 运行报告','',
        '状态：`'+decision['status']+'`。H1–H6：`NOT_EVALUATED`。','',
        'CPU 融合门禁：`'+cpu.get('status','NOT_RUN')+'`；真实 T4 Smoke：`'+gpu_result.get('status','NOT_RUN')+'`。',
        '原始 CPU 指标在 audit/preflight.json；真实 GPU 训练及资源在 gpu/smoke.json，未运行时不得补填。',
        '分支消融只检查本 E5 的关闭行为，不宣称数值退化为已冻结的 E3/E4。',
        '完整高阶 Memory 更新规则复用 M3；Mamba-1 与 SWA 复用 M2。持久 Cell、SSM、KV 分别统计。',
        '这不是质量、长程互补性或显著性结论。未启动 M5/M6、20M/100M。','',
        '```json',json.dumps(decision.get('error',{}),ensure_ascii=False,indent=2),'```',''])
    (Path(out)/'REPORT.zh-CN.md').write_text(text)


def finish(out,decision):
    decision['publication']=publish(out,decision)
    if decision['status']=='M4_SMOKE_PASS_PUBLICATION_PENDING':
        decision['status']='M4_ENGINEERING_COMPLETE';write(out/'decision.json',decision);report(out,decision);publish(out,decision)
    else:write(out/'decision.json',decision)


def launch(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.publish_only:finish(out,json.loads((out/'decision.json').read_text()));return
    protocol=json.loads((BASE/'configs/smoke.json').read_text());started=args.started or time.monotonic()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run_identity={'source_commit':source,'implementation':identity(),'protocol':digest(protocol)}
    path=out/'decision.json';previous=json.loads(path.read_text()) if path.exists() else None
    if previous and previous['identity']!=run_identity:raise ValueError('different source/config output; choose a new directory')
    run_id='epclm004-m4-'+source[:12]+'-'+uuid.uuid4().hex[:8]
    decision=previous or {'format':'minicells.epclm004.m4.v1','run_id':run_id,'branch':'kaggle/'+run_id,
        'identity':run_identity,'script_version_id':os.environ.get('KAGGLE_SCRIPT_VERSION_ID'),
        'hypotheses':'NOT_EVALUATED','independent_test_accessed':False}
    write(out/'protocol.json',protocol);exit_code=0
    def timeout(signum,frame):raise TimeoutError('execution budget reached; publication reserve preserved')
    signal.signal(signal.SIGALRM,timeout)
    seconds=int(started+protocol['limits']['execution_seconds']-time.monotonic())
    if seconds<=0:raise TimeoutError('no execution budget remains')
    signal.alarm(seconds)
    try:
        verify_frozen();decision['preflight']=preflight(out/'audit',protocol)
        if args.cpu_only:decision['status']='M4_CPU_ENGINEERING_PASS_T4_PENDING'
        else:
            decision['gpu']=gpu(out/'gpu',protocol,run_identity,started+protocol['limits']['execution_seconds'])
            decision['status']='M4_SMOKE_PASS_PUBLICATION_PENDING'
    except BaseException as error:
        exit_code=1;decision['status']='BLOCKED_GPU_ACCESS' if 'BLOCKED_GPU_ACCESS' in str(error) else 'INCOMPLETE'
        decision['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        signal.alarm(0);decision['wall_seconds']=time.monotonic()-started;write(path,decision);report(out,decision)
    if args.publish:finish(out,decision)
    print({'status':decision['status'],'report':str(path)},flush=True)
    if exit_code:raise SystemExit(exit_code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--started',type=float);parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--publish',action='store_true');parser.add_argument('--publish-only',action='store_true')
    launch(parser.parse_args())
