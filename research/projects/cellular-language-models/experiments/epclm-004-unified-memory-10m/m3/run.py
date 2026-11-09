"""Ordered fail-closed M3 gates; no full-size CPU training or silent fallback."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import torch
from memory import NeuralMemory, state_bytes
from mag_model import MAGLite, Config
import gates
from e0 import ROOT, BASE as M1, write, load_checkpoint
from data import file_hash, digest, MQARConfig, episode, batch, dataset_manifest
from training import fit

BASE=Path(__file__).resolve().parent


def implementation():
    return {str(p.relative_to(BASE)):file_hash(p) for p in sorted(BASE.glob('*.py'))}


def gate_identity(protocol):
    return digest({'implementation':implementation(),'protocol':protocol,'torch':torch.__version__,
                   'python':platform.python_version(),'device':'cpu','dtype':'float64/float32'})


def verify_frozen():
    lock=json.loads((BASE/'configs/source-lock.json').read_text())
    for relative,expected in lock['files'].items():
        if file_hash(ROOT/relative)!=expected:raise ValueError('frozen source/hash differs: '+relative)
    return lock


def cpu_gates(out,protocol):
    key=gate_identity(protocol);records={};out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for i in range(1,6):
        path=out/f'gate{i}.json'
        record=json.loads(path.read_text()) if path.exists() else None
        intact=record and all((out/name).exists() and file_hash(out/name)==sha for name,sha in record.get('artifact_sha256',{}).items())
        if intact and record.get('identity')==key and record.get('status')=='PASS':
            print({'gate':i,'status':'PASS','cached':True},flush=True)
        else:
            started=time.perf_counter()
            gate_out=out/f'gate{i}'/key[:16]
            try:record=getattr(gates,f'gate{i}')(gate_out)
            except BaseException as error:
                write(path,{'identity':key,'status':'FAIL','error':f'{type(error).__name__}: {error}'})
                raise
            record.update(identity=key,wall_seconds=time.perf_counter()-started,
                artifact_sha256={str(p.relative_to(out)):file_hash(p) for p in gate_out.rglob('*') if p.is_file()})
            write(path,record);print({'gate':i,'status':record['status']},flush=True)
        records[f'gate{i}']=record
    return records


def synchronize():torch.cuda.synchronize()


def gpu_smoke(out,protocol,identity,deadline,assets):
    if not torch.cuda.is_available():raise RuntimeError('BLOCKED_GPU_ACCESS: CUDA unavailable')
    name=torch.cuda.get_device_name()
    if 'T4' not in name:raise RuntimeError('BLOCKED_GPU_ACCESS: target is Tesla T4; found '+name)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    finished=out/'gate6.json'
    if finished.exists():
        previous=json.loads(finished.read_text())
        if previous.get('identity')==identity and previous.get('status')=='PASS':
            # Completed training is reused only with intact artifacts, not merely a PASS string.
            for relative,sha in previous['artifact_sha256'].items():
                if file_hash(out/relative)!=sha:raise ValueError('completed GPU artifact differs: '+relative)
            return previous
    dataset=MQARConfig(**protocol['mqar'])
    groups={split:[episode(dataset,split,i) for i in range(count)] for split,count in protocol['episodes'].items()}
    manifest=dataset_manifest(protocol['mqar'],groups,protocol['tokenizer_sha256'])
    write(out/'dataset-manifest.json',manifest)
    if file_hash(M1/'assets/tokenizer.json')!=protocol['tokenizer_sha256']:raise ValueError('frozen tokenizer differs')
    torch.manual_seed(protocol['training']['seed']);torch.cuda.manual_seed_all(protocol['training']['seed'])
    model=MAGLite(Config(**protocol['model']),**protocol['memory']).cuda()
    # Frozen E1 is exactly the common initialization prefix. No parameter resizing.
    from baselines import build
    reference=build('E1',protocol['model'],protocol['training']['seed'])
    shared=model.state_dict()
    assert all(torch.equal(p,shared[n].cpu()) for n,p in reference.state_dict().items())
    reference_count=sum(p.numel() for p in reference.parameters());del reference
    params=sum(p.numel() for p in model.parameters())
    assert abs(params/reference_count-1)<.01
    identity={**identity,'data':manifest,'model_id':model.model_id}
    event_path=out/'training-events.json'
    events=json.loads(event_path.read_text()) if event_path.exists() else []
    try:
        metrics=fit(model,groups['train'],groups['validation'],protocol['training'],out/'training',identity,deadline)
    finally:
        events.extend(model.events);write(event_path,events)
    assert metrics['status']=='COMPLETE' and metrics['ledger']['committed_updates']==2
    assert len(metrics['history'])==2 and all(r['parameter_change_norm']>0 for r in metrics['history'])
    # Keep only forwards represented in the durable committed ledger.
    indexed={e['input_sha256']:e for e in events}
    events=[indexed[r['input_sha256']] for r in metrics['history']]
    assert len(events)==2 and all(e['fast_delta']>0 and e['momentum_norm']>0 for e in events)
    assert all(e['inner_updates']==8 for e in events)
    saved=load_checkpoint(out/'training/checkpoint.pt')
    assert saved['ledger']==metrics['ledger'] and saved['identity']==identity
    model.eval();model.zero_grad(set_to_none=True)
    probe=batch(groups['validation'][:1],'cuda').input_ids
    with torch.no_grad():expected=model(probe).detach()
    # Restore the actual optimizer/RNG/sampler/ledger payload, then compare a new model.
    restored=MAGLite(Config(**protocol['model']),**protocol['memory']).cuda()
    restored.load_state_dict(saved['model']);restored.eval()
    with torch.no_grad():actual=restored(probe)
    restore_error=float((actual-expected).abs().max());assert restore_error==0
    optimizer=torch.optim.AdamW(restored.parameters(),lr=protocol['training']['lr'])
    optimizer.load_state_dict(saved['optimizer'])
    assert optimizer.state and all(int(s['step'])==2 for s in optimizer.state.values())
    del restored,optimizer,saved;gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    synchronize();started=time.perf_counter()
    with torch.no_grad():model(probe)
    synchronize();inference_seconds=time.perf_counter()-started
    inference_peak=torch.cuda.max_memory_allocated()
    model.reset_runtime()
    # Actual learned E4 static projections/init are frozen; persistent Cell alone is adapted.
    recall=gates.recall_control(out,'cuda',model.memory,context={'backbone_checkpoint':file_hash(out/'training/checkpoint.pt'),
        'tokenizer':protocol['tokenizer_sha256'],'model_id':model.model_id,'source':identity['source']})
    cell=model.memory.restore(out/'recall-cell',json.loads((out/'recall-cell/manifest.json').read_text())['compatibility'])
    resources={'static_parameters':params,'frozen_E1_parameters':reference_count,
        'parameter_difference_fraction':params/reference_count-1,
        'fast_weights_parameters_per_stream':state_bytes(cell)['fast_parameters_per_stream'],
        'momentum_bytes_per_stream':state_bytes(cell)['momentum'],
        'runtime_cell_snapshot_bytes':sum(state_bytes(cell)[k] for k in ('fast_weights','momentum')),
        'runtime_kv_or_ssm_bytes':0,'runtime_policy':'full-sequence SWA; no KV/SSM dependency; Cell is separate',
        'training_peak_allocated_bytes':metrics['training_peak_allocated_bytes'],
        'inference_peak_allocated_bytes':inference_peak,'inference_full_sequence_seconds':inference_seconds,
        'cell_checkpoint_bytes':recall['bytes'],'cell_checkpoint_write_seconds':recall['write_seconds'],
        'memory_update_seconds':recall['memory_update_seconds'],'recall_seconds':recall['recall_seconds'],
        'timing_scope':'memory/recall timings are 128-step latent positive control; not formal throughput',
        'model_checkpoint_write_seconds':metrics['checkpoint_policy']['write_seconds_this_process']}
    record={'identity':identity,'status':'PASS','hardware':{'gpu':name,'torch':torch.__version__,
        'cuda':torch.version.cuda,'precision':'fp32','tf32':False},'training':metrics,'events':events,
        'training_inner_updates':sum(e['inner_updates'] for e in events),'restore_max_abs_error':restore_error,
        'optimizer_restored':True,'cross_session':recall,'resources':resources}
    # Identity above includes the data; persist the gate identity expected by callers separately.
    record['training_identity']=record['identity'];record['identity']={k:v for k,v in identity.items() if k not in ('data','model_id')}
    record['artifact_sha256']={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p!=finished}
    write(finished,record);return record


def report(out,decision):
    rows=[]
    for i in range(1,6):
        path=Path(out)/'audit'/f'gate{i}.json'
        value=json.loads(path.read_text()) if path.exists() else {}
        rows.append(f"| Gate {i} | {value.get('status','NOT_RUN')} |")
    rows.append(f"| Gate 6 T4 | {decision.get('gate6',{}).get('status','NOT_RUN')} |")
    text="\n".join(["# M3 运行验收报告", "",f"状态：`{decision['status']}`；H3：`NOT_EVALUATED`。", "",
        "| 门禁 | 状态 |","|---|---|",*rows,"", "原始误差与容差见 audit/gate1.json；更新与持久化证据见 audit/gate3.json、audit/gate5.json。",
        "真实 T4 训练、Token 记账、Checkpoint 恢复及显存数据仅以 gpu/gate6.json 为准；文件不存在时均为尚未验证。",
        "gpu/training/ 保存独立的 Model / Optimizer / RNG / Sampler / Ledger 原子检查点；recall-cell/ 保存 Fast Weights 与 Momentum。",
        "跨会话召回为确定性的潜空间关联正向检查，不是 MQAR、自然语言质量或 H3 的科学结果。",
        "完整数学定义、梯度边界和已知限制见源码目录 README.md。", "",
        "```json",json.dumps(decision.get('failure',{}),ensure_ascii=False,indent=2),"```", ""])
    (Path(out)/'REPORT.zh-CN.md').write_text(text)


def artifacts(out):
    # Weights remain Kaggle Output. Reports, raw metrics and Cell manifests are published.
    return [p for p in Path(out).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log','.md')
            and p.name not in ('publication.json','publication-files.json')]


def publish(out,decision):
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    # The shared publisher switches its checkout. Each attempt gets a clean isolated
    # repository, so a second publication cannot dirty/switch the live experiment.
    source=decision['identity']['source']
    origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory(prefix='epclm004-m3-publish-') as temporary:
        root=Path(temporary)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',source],cwd=root,check=True)
        files=artifacts(out);stage=root/'results/epclm004-m3/publish';stage.mkdir(parents=True)
        names=[]
        for p in files:
            relative=str(p.relative_to(out));target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target);names.append(relative)
        hashes={name:file_hash(stage/name) for name in names}
        write(stage/'publication-files.json',hashes);names.append('publication-files.json')
        destination=root/'artifacts/experiments/epclm004-m3'/decision['run_id']
        spec=ExperimentSpec(str(stage.relative_to(root)),str(destination.relative_to(root)),tuple(names),decision['branch'],decision['format'])
        folder=prepare_artifacts(root,'epclm004-m3',spec,decision.get('script_version_id'))
        push_results(root,folder,'epclm004-m3',decision['branch'],'GITHUB_TOKEN','origin/main')
        for name,sha in hashes.items():
            if file_hash(destination/name)!=sha:raise ValueError('published report hash differs')
        receipt={'status':'REMOTE_SHA_VERIFIED','branch':decision['branch'],
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'file_count':len(hashes),'files':hashes}
    write(out/'publication.json',receipt)
    return receipt


def finish_publication(out,decision):
    decision['publication']=publish(out,decision)
    if decision['status']=='GPU_SMOKE_PASS_PUBLICATION_PENDING':
        decision['status']='M3_ENGINEERING_COMPLETE'
        write(out/'decision.json',decision);report(out,decision)
        # Link to a prior verified commit rather than inventing a self-referential SHA.
        publish(out,decision)
    else:write(out/'decision.json',decision)


def launch(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.publish_only:
        decision=json.loads((out/'decision.json').read_text());finish_publication(out,decision);return
    if (BASE/'freeze/BASELINE.json').exists():
        raise RuntimeError('M3_FROZEN_NO_FURTHER_TRAINING: use archive-only.ipynb; --publish-only remains available')
    started=args.started or time.monotonic();protocol=json.loads((BASE/'configs/smoke.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run_id='epclm004-m3-'+source[:12]+'-'+uuid.uuid4().hex[:8]
    previous=json.loads((out/'decision.json').read_text()) if (out/'decision.json').exists() else None
    identity={'source':source,'implementation':implementation(),'protocol':digest(protocol)}
    if previous and previous['identity']!=identity:raise ValueError('output belongs to another source/config; choose fresh output')
    decision=previous or {'format':'minicells.epclm004.m3.v1','run_id':run_id,'branch':'kaggle/'+run_id,
        'identity':identity,'hypotheses':'NOT_EVALUATED','independent_test_accessed':False}
    decision['script_version_id']=os.environ.get('KAGGLE_SCRIPT_VERSION_ID') or decision.get('script_version_id')
    write(out/'protocol.json',protocol)
    stage='SOURCE';exit_code=0
    def timeout(signum,frame):raise TimeoutError('M3 execution budget reached; publication reserve preserved')
    signal.signal(signal.SIGALRM,timeout)
    remaining=int(started+protocol['limits']['execution_seconds']-time.monotonic())
    if remaining<=0:raise TimeoutError('no execution budget remains')
    signal.alarm(remaining)
    try:
        verify_frozen();stage='GATES_1_TO_5'
        decision['gates']=cpu_gates(out/'audit',protocol)
        if args.cpu_only:
            decision['status']='BLOCKED_GPU_ACCESS';decision['gate6']={'status':'NOT_RUN','reason':'CPU-only development environment; T4 required'}
        else:
            stage='GATE6_T4';gpu=gpu_smoke(out/'gpu',protocol,identity,started+protocol['limits']['execution_seconds'],args.assets)
            decision['gate6']=gpu;decision['status']='GPU_SMOKE_PASS_PUBLICATION_PENDING'
    except BaseException as error:
        exit_code=1;decision['status']='BLOCKED_GPU_ACCESS' if 'BLOCKED_GPU_ACCESS' in str(error) else 'INCOMPLETE'
        decision['failure']={'stage':stage,'type':type(error).__name__,'message':str(error)}
    finally:
        signal.alarm(0);decision['wall_seconds']=time.monotonic()-started
        write(out/'decision.json',decision);report(out,decision)
    if args.publish:finish_publication(out,decision)
    print({'status':decision['status'],'report':str(out/'decision.json')},flush=True)
    if exit_code:raise SystemExit(exit_code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=Path('/kaggle/working/epclm004-m1-assets'))
    parser.add_argument('--started',type=float);parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--publish',action='store_true');parser.add_argument('--publish-only',action='store_true')
    launch(parser.parse_args())
