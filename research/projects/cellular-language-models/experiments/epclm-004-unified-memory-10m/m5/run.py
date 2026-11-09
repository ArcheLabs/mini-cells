"""Ordered M5 engineering chain. Gate 0 archive is mandatory, not a training fallback."""
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
import math
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'src'))
from bmojo import BMojo
from store import Store
import gate0
import m5_checks
from tasks import cases,evaluate
from e0 import ROOT,write,load_checkpoint
from training import fit
from data import MQARConfig,episode,dataset_manifest,digest,file_hash,prepare_wikitext,lm_rows


def implementation():
    return {str(p.relative_to(BASE)):file_hash(p) for p in [BASE/'run.py',*sorted((BASE/'src').glob('*.py'))]}


def verify_frozen():
    for name,sha in json.loads((BASE/'configs/source-lock.json').read_text())['files'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('frozen dependency changed: '+name)


def construct(protocol,arm):
    spec=protocol['arms'][arm]
    return BMojo(protocol['model'],**spec,summary_tokens=protocol['summary_tokens'],capacity=protocol['capacity'],
        read_top_k=protocol['read_top_k'],chunk=protocol['chunk'],state_size=protocol['state_size'],
        seed=protocol['seed'],beta=protocol['predictor']['beta'])


def weight_hash(model,backbone=False):
    import hashlib
    h=hashlib.sha256()
    for name,p in (model.backbone if backbone else model).state_dict().items():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_audit(protocol):
    result={};common=backbone=None
    for arm in protocol['arms']:
        model=construct(protocol,arm);count=sum(p.numel() for p in model.parameters())
        assert 9700000<=count<=10300000
        actual=weight_hash(model);shared=weight_hash(model,True)
        if backbone is None:backbone=shared
        assert shared==backbone
        if arm!='E3':
            if common is None:common=actual
            assert actual==common
        assert not any(type(m).__name__=='NeuralMemory' for m in model.modules())
        result[arm]={'static_parameters':count,'relative_to_10m':count/10000000-1,
            'initial_parameter_sha256':actual,'shared_backbone_sha256':shared,'learnable_predictor_parameters':0}
    return result


def ordered_checks(out,protocol,results):
    key=digest({'implementation':implementation(),'protocol':protocol,'torch':torch.__version__,'python':platform.python_version()})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for number in range(1,7):
        path=out/f'gate{number}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('identity')==key and saved['status']=='PASS':
                for name,sha in saved['files'].items():
                    if file_hash(out/name)!=sha:raise ValueError('passed gate artifact differs: '+name)
                results[str(number)]=saved;continue
        folder=out/f'gate{number}'/key[:16];folder.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        try:
            result=getattr(m5_checks,f'gate{number}')(folder)
            if number==6:result['canonical_parameter_audit']=parameter_audit(protocol)
        except BaseException as error:
            results[str(number)]={'identity':key,'status':'FAIL','error':f'{type(error).__name__}: {error}'}
            write(path,results[str(number)]);raise
        result.update(identity=key,wall_seconds=time.perf_counter()-started,
            files={str(p.relative_to(out)):file_hash(p) for p in folder.rglob('*') if p.is_file()})
        write(path,result);results[str(number)]=result;print({'gate':number,'status':'PASS'},flush=True)
    return results


def sync():torch.cuda.synchronize()


def gpu_arm(out,protocol,arm,groups,bank,lm,identity,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);report=out/'arm.json'
    if report.exists():
        old=json.loads(report.read_text())
        if old['identity']!=identity:raise ValueError('completed arm identity differs')
        if old['status']=='PASS':
            for name,sha in old['files'].items():
                if file_hash(out/name)!=sha:raise ValueError('completed arm artifact differs: '+name)
            return old
    torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed'])
    model=construct(protocol,arm).cuda();initial_lm=evaluate(model,lm,'cuda',False)
    training_identity={**identity,'arm':arm,'train_sha256':digest(groups['train']),'validation_sha256':digest(groups['validation'])}
    event_path=out/'events.json';events=json.loads(event_path.read_text()) if event_path.exists() else []
    began_fit=time.perf_counter();metrics_path=out/'training/metrics.json'
    previous=json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    if previous and previous['status']=='COMPLETE' and previous['ledger']['committed_updates']==2:
        saved=load_checkpoint(out/'training/checkpoint.pt')
        if saved['identity']!=training_identity or saved['ledger']!=previous['ledger'] or saved['history']!=previous['history']:
            raise ValueError('completed training ledger/identity differs')
        if file_hash(out/'training/checkpoint.pt')!=previous['checkpoint']['sha256']:raise ValueError('completed checkpoint differs')
        model.load_state_dict(saved['model']);metrics=previous;del saved
    else:
        try:metrics=fit(model,groups['train'],groups['validation'],protocol['training'],out/'training',training_identity,deadline)
        finally:events+=model.events;write(event_path,events)
    fit_seconds=time.perf_counter()-began_fit
    assert metrics['status']=='COMPLETE' and metrics['ledger']['committed_updates']==2
    indexed={e['input_sha256']:e for e in events};committed=[indexed[h['input_sha256']] for h in metrics['history']]
    if model.policy:
        assert all(e['store'][0]['writes']>=protocol['capacity'] and e['store'][0]['replacements']>0 for e in committed)
        if model.policy=='innovation':assert all(e['store'][0]['score_comparisons']>0 for e in committed)
    gradients=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)
    saved=load_checkpoint(out/'training/checkpoint.pt')
    optimizer_bytes=sum(v.numel()*v.element_size() for state in saved['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
    restored=construct(protocol,arm).cuda();restored.load_state_dict(saved['model']);restored.eval();model.eval()
    optimizer=torch.optim.AdamW(restored.parameters(),lr=protocol['training']['lr']);optimizer.load_state_dict(saved['optimizer'])
    assert all(int(state['step'])==2 for state in optimizer.state.values())
    assert saved['ledger']==metrics['ledger'] and saved['identity']==training_identity
    probe=torch.tensor([groups['validation'][0]['input_ids']],device='cuda')
    with torch.no_grad():
        expected=model(probe);actual=restored(probe);model_restore_error=float((actual-expected).abs().max());assert model_restore_error==0
    del restored,optimizer,saved,expected,actual;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
    with torch.no_grad():
        logits,state,trace=model.forward_state(probe,profile=True);timing=dict(model.last_timing);sync();inference_seconds=time.perf_counter()-start
        resources=model.resources(state)
        metrics['parameters']={'static_parameters':resources['static_parameters'],'breakdown':resources['static_parameter_breakdown'],'tied_embedding_head':True}
        write(out/'training/metrics.json',metrics)
        runtime_checkpoint=model.checkpoint(state,out/'runtime')
        tail=probe[:,-3:];first,_,_=model.forward_state(tail,state)
        model.reset_runtime();loaded=model.restore(out/'runtime');second,_,_=model.forward_state(tail,loaded)
        runtime_error=float((first-second).abs().max());assert runtime_error==0
        store_only=None
        if model.policy:
            model.reset_runtime();only=model.restore(out/'runtime',store_only=True);position=only.position
            read,_,retrieved=model.forward_state(tail,only)
            empty=model.new_runtime(1,position);blank,_,_=model.forward_state(tail,empty)
            delta=float((read-blank).abs().max())
            assert delta>0 and any(any(p>=0 for p in item['selected_positions']) for item in retrieved)
            store_only={'read_vs_empty_delta':delta,'actual_store_read':True}
    inference_peak=torch.cuda.max_memory_allocated()
    del logits,state,loaded,first,second,tail
    if model.policy:del only,empty,read,blank
    model.reset_runtime();gc.collect();torch.cuda.empty_cache()
    final_lm=evaluate(model,lm,'cuda',False)
    for obj in (initial_lm,final_lm):obj['ppl']=math.exp(obj['nll']) if obj['nll']<700 else None
    evaluations={}
    for capacity in protocol['capacity_tiers'] if model.policy else [0]:
        outcomes=[]
        for case in bank:
            if time.monotonic()>=deadline:raise TimeoutError('evaluation budget exhausted; no cases silently omitted')
            # Same static weights; capacity tier is an inference intervention, not extra training.
            old_capacity=model.capacity
            if capacity:model.capacity=capacity
            try:measurement=evaluate(model,[case['row']],'cuda')
            finally:model.capacity=old_capacity
            outcomes.append({'case':case['name'],'task':case['task'],'config':case['config'],
                'similarity':case['similarity'],'value_repeat':case['value_repeat'],**measurement})
        evaluations[str(capacity)]=outcomes
    resources.update({'training_peak_allocated_bytes':metrics['training_peak_allocated_bytes'],'inference_probe_peak_allocated_bytes':inference_peak,'inference_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'gradient_tensor_bytes_after_fit_this_process':gradients,'expected_full_static_gradient_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'optimizer_tensor_bytes':optimizer_bytes,
        'activation_workspace_other_residual_estimate_bytes':max(0,metrics['training_peak_allocated_bytes']-2*resources['static_weight_bytes']-optimizer_bytes),
        'residual_scope':'unattributed estimate; tensor residency/allocator overhead not independently separated',
        'inference_seconds_512_tokens':inference_seconds,'inference_tokens_per_second':probe.numel()/inference_seconds,
        'training_compute_tokens_per_second':metrics['ledger']['input_tokens']/sum(h['compute_seconds'] for h in metrics['history']),
        'fit_wall_seconds_this_process':fit_seconds,'new_committed_updates_this_process':len(model.events),'throughput_scope':'engineering smoke/probe only; fit wall includes validation and IO; profiled inference includes phase synchronization, not formal performance','read_update_latency':timing,
        'model_checkpoint_bytes':metrics['checkpoint']['bytes'],'runtime_checkpoint':runtime_checkpoint})
    result={'status':'PASS','identity':identity,'arm':arm,'training':metrics,'training_events':committed,
        'model_restore_error':model_restore_error,'runtime_restore_error':runtime_error,'store_only':store_only,
        'resources':resources,'wiki_validation':{'initial':initial_lm,'final':final_lm,'delta_nll':final_lm['nll']-initial_lm['nll']},
        'evaluations':evaluations,'capacity128_scope':'inference-only with the capacity64-trained model; no retraining',
        'hypotheses':'NOT_EVALUATED'}
    result['files']={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p!=report}
    write(report,result);del model;gc.collect();torch.cuda.empty_cache();return result


def gpu(out,protocol,identity,deadline):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('BLOCKED_HARDWARE: actual Tesla T4 required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=MQARConfig(**protocol['mqar']);groups={k:[episode(cfg,k,i) for i in range(v)] for k,v in protocol['data_sizes'].items() if k!='test'}
    data=dataset_manifest(protocol['mqar'],groups,protocol['tokenizer_sha256']);write(out/'dataset-manifest.json',data)
    reference=construct(protocol,'E6');embedding=reference.token_embedding.weight.detach().cpu().clone();del reference
    bank=cases(protocol,embedding)
    write(out/'evaluation-data-manifest.json',{'cases':[{'name':c['name'],'config':c['config'],'task':c['task'],
        'similarity':c['similarity'],'value_repeat':c['value_repeat'],'episode':c['row']['id'],
        'input_sha256':digest(c['row']['input_ids']),'labels_sha256':digest(c['row']['labels'])} for c in bank],
        'reference_embedding_sha256':__import__('hashlib').sha256(embedding.numpy().tobytes()).hexdigest(),
        'key_similarity_scope':'cosine strata of fixed common untrained embedding; not semantic similarity or post-training guarantee'})
    asset_manifest=prepare_wikitext(out.parent/'assets');write(out/'wiki-data-manifest.json',asset_manifest)
    lm=lm_rows(out.parent/'assets','validation',length=128,limit=protocol['evaluation']['lm_limit'])
    results={}
    for arm in protocol['arms']:
        if time.monotonic()>=deadline:raise TimeoutError('training budget exhausted')
        results[arm]=gpu_arm(out/arm,protocol,arm,groups,bank,lm,identity,deadline)
    hashes=[[h['input_sha256'] for h in record['training']['history']] for record in results.values()]
    assert all(value==hashes[0] for value in hashes)
    assert all(record['training']['ledger']['input_tokens']==1024 for record in results.values())
    return {'status':'PASS','arms':results,'paired_training_inputs':'IDENTICAL','total_training_input_tokens':5120,
        'hardware':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'fp32','tf32':False},
        'independent_test_accessed':False,'hypotheses':'NOT_EVALUATED'}


def report(out,decision):
    lines=['# M5 运行报告','',f"状态：`{decision['status']}`；H5：`NOT_EVALUATED`。",'', '| Gate | 状态 |','|---|---|']
    for number in range(8):lines.append(f"| {number} | {decision.get('gates',{}).get(str(number),{}).get('status','NOT_RUN')} |")
    lines+=['','所有质量结果为开发诊断，不是 M6 Screening/Confirmation。M4 不重训；M5 E3 是新协议的 fresh 控制。',
        'Capacity128 是同一模型的推理干预，没有额外训练；不声称与 Capacity128 独立训练等价。',
        '原始逐臂指标、Store/Runtime Manifest 与 SHA 位于 gpu/ 和 audit/。权重保留 Kaggle Output。',
        '','```json',json.dumps(decision.get('error',{}),ensure_ascii=False,indent=2),'```','']
    (Path(out)/'REPORT.zh-CN.md').write_text('\n'.join(lines))


def publish(out,decision):
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory(prefix='m5-publish-') as temp:
        root=Path(temp)/'repo';subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',decision['identity']['source_commit']],cwd=root,check=True)
        stage=root/'results/epclm004-m5/publish';stage.mkdir(parents=True);names=[]
        for path in Path(out).rglob('*'):
            relative=path.relative_to(out)
            if path.is_file() and path.suffix in ('.json','.jsonl','.log','.md') and not {'assets','source-cache'}&set(relative.parts) and path.name!='publication.json':
                dest=stage/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);names.append(str(relative))
        hashes={name:file_hash(stage/name) for name in names};write(stage/'publication-files.json',hashes);names.append('publication-files.json')
        destination=root/'artifacts/experiments/epclm004-m5'/decision['run_id']
        spec=ExperimentSpec(str(stage.relative_to(root)),str(destination.relative_to(root)),tuple(names),decision['branch'],decision['format'])
        folder=prepare_artifacts(root,'epclm004-m5',spec,decision.get('script_version_id'))
        push_results(root,folder,'epclm004-m5',decision['branch'],'GITHUB_TOKEN','origin/main')
        for name,sha in hashes.items():assert file_hash(destination/name)==sha
        receipt={'status':'REMOTE_SHA_VERIFIED','branch':decision['branch'],
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'files':hashes}
    write(out/'publication.json',receipt);return receipt


def finish(out,decision):
    decision['publication']=publish(out,decision)
    if decision['status']=='M5_GATES_PASS_PUBLICATION_PENDING':
        decision['status']='M5_ENGINEERING_COMPLETE';write(out/'decision.json',decision);report(out,decision);publish(out,decision)
    else:write(out/'decision.json',decision)


def launch(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.publish_only:finish(out,json.loads((out/'decision.json').read_text()));return
    protocol=json.loads((BASE/'configs/protocol.json').read_text());started=args.started or time.monotonic()
    identity={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'implementation':implementation(),'protocol':digest(protocol)}
    old=json.loads((out/'decision.json').read_text()) if (out/'decision.json').exists() else None
    if old and old['identity']!=identity:raise ValueError('output belongs to another source/config')
    run_id='epclm004-m5-'+identity['source_commit'][:12]+'-'+uuid.uuid4().hex[:8]
    decision=old or {'format':'minicells.epclm004.m5.v1','identity':identity,'run_id':run_id,'branch':'kaggle/'+run_id,
        'gates':{},'hypotheses':'NOT_EVALUATED','independent_test_accessed':False,'script_version_id':os.environ.get('KAGGLE_SCRIPT_VERSION_ID')}
    write(out/'protocol.json',protocol);exit_code=0;stage='0'
    def timeout(signum,frame):raise TimeoutError('M5 execution budget reached; publication reserve retained')
    signal.signal(signal.SIGALRM,timeout);remaining=int(started+protocol['budget']['execution_seconds']-time.monotonic())
    if remaining<=0:raise TimeoutError('no execution budget remains')
    signal.alarm(remaining)
    try:
        verify_frozen();decision['gates']={'0':gate0.run(out/'gate0')}
        if decision['gates']['0']['status']!='PASS':raise RuntimeError('M4 archive gate did not pass; later gates stopped')
        write(out/'decision.json',decision)
        stage='1-6';ordered_checks(out/'audit',protocol,decision['gates']);write(out/'decision.json',decision)
        stage='7';decision['gates']['7']=gpu(out/'gpu',protocol,identity,started+protocol['budget']['execution_seconds'])
        decision['status']='M5_GATES_PASS_PUBLICATION_PENDING'
    except BaseException as error:
        exit_code=1;decision['status']='BLOCKED_GATE0' if stage=='0' else 'INCOMPLETE'
        if stage in ('0','7'):decision['gates'][stage]={'status':'FAIL','error':str(error)}
        decision['error']={'stage':stage,'type':type(error).__name__,'message':str(error)}
    finally:
        signal.alarm(0);decision['wall_seconds']=time.monotonic()-started;write(out/'decision.json',decision);report(out,decision)
    if args.publish:finish(out,decision)
    print({'status':decision['status'],'report':str(out/'decision.json')},flush=True)
    if exit_code:raise SystemExit(exit_code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--started',type=float);parser.add_argument('--publish',action='store_true');parser.add_argument('--publish-only',action='store_true')
    launch(parser.parse_args())
