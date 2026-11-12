"""Bounded L1 factorial screening. Uses the frozen M2 transaction loop."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from adapter import (M4, MASKS, Factorial, digest, file_hash, frozen_hash,
                     load_checkpoint, load_session, save_session, sessions, tensor_digest, wrapper)
from data import MQARConfig, batch, dataset_manifest, episode
from e0 import ROOT, Sampler, write
from training import fit

BASE=Path(__file__).resolve().parent
P=json.loads((BASE/'protocol.json').read_text())
M4_CONFIG=json.loads((M4/'configs/smoke.json').read_text())
CONFIG={**M4_CONFIG['training'],'seed':P['model_seed'],'updates':32}
TOKENIZER=M4.parent/'assets/tokenizer.json'
CONDITIONS={'C00':(False,False),'C10':(True,False),'C01':(False,True),'C11':(True,True)}
FORMAT='minicells.epclm005.m4-h4-fast-screen.v1'


def build(arm,device):
    return Factorial(arm,M4_CONFIG['model'],{**M4_CONFIG['fusion'],'seed':P['model_seed']}).to(device)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def source_identity():
    files={}
    for sub in ('m2','m3','m4'):
        for path in (M4.parent/sub).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json') and not any(
                    part in ('audit','freeze','runs','tests') for part in path.relative_to(M4.parent).parts):
                files[str(path.relative_to(ROOT))]=file_hash(path)
    for name in ('data.py','e0.py','configs/model-e0.json'):
        path=M4.parent/name
        files[str(path.relative_to(ROOT))]=file_hash(path)
    return {'source_commit':git('rev-parse','HEAD'),'base_main':'ce9f1905b8d9a8badd2dbed111f68e180e71a619',
            'source_files':files,'config_sha256':digest([P,M4_CONFIG,CONFIG]),
            'tokenizer_sha256':file_hash(TOKENIZER),
            'm4_result_commit':'f0ab964502c1caa15d09ea1185ce16bec1674a20',
            'm4_verified_result_sha256':file_hash(M4/'audit/t4-result-verification.json')}


def datasets():
    spec=MQARConfig(**P['mqar'])
    splits={k:[episode(spec,k,i) for i in range(count)]
            for k,count in [('train',P['train_episodes']),('validation',P['validation_episodes'])]}
    evaluation={str(seed):[episode(replace(spec,seed=seed),'validation',i)
                            for i in range(P['evaluation_episodes_per_seed'])] for seed in P['evaluation_seeds']}
    all_rows=[*splits['train'],*splits['validation'],*(r for bank in evaluation.values() for r in bank)]
    assert len({r['id'] for r in all_rows})==len(all_rows)
    train_pairs={tuple(p) for r in splits['train'] for p in r['bindings']}
    held_pairs={tuple(p) for r in all_rows[len(splits['train']):] for p in r['bindings']}
    assert not train_pairs & held_pairs
    assert len(splits['train'])==CONFIG['batch']*CONFIG['updates']
    sessions(all_rows,M4_CONFIG['fusion']['chunk'],'cpu')
    delays=[q-(3+3*r['bindings'].index((r['input_ids'][q-1]-8,r['labels'][q]-264)))
            for r in all_rows for q in r['query_positions']]
    assert min(delays)>1+5*(32-1)
    manifest=dataset_manifest(P['mqar'],splits,file_hash(TOKENIZER))
    manifest.update(evaluation={seed:dict(episodes=len(rows),sha256=digest(rows))
                                for seed,rows in evaluation.items()},
                    train_heldout_pair_overlap=0,minimum_fact_query_delay=min(delays),
                    swa_receptive_field=156,input_sha256=digest([r['input_ids'] for r in splits['train']]),
                    target_sha256=digest([r['labels'] for r in splits['train']]))
    return splits,evaluation,manifest


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def state_report(cell,ssm):
    flat={} if ssm is None else {f'{k}.{i}':t for k,pair in ssm.mamba_state.items() for i,t in enumerate(pair)}
    return dict(ssm_bytes=sum(t.numel()*t.element_size() for t in flat.values()),
                ssm_sha256=tensor_digest(flat),
                ssm_norm=sum(float(t.square().sum()) for t in flat.values())**.5,
                fast_bytes=0 if cell is None else sum(t.numel()*t.element_size() for t in cell.fast.values()),
                momentum_bytes=0 if cell is None else sum(t.numel()*t.element_size() for t in cell.momentum.values()),
                fast_sha256=None if cell is None else tensor_digest(cell.fast),
                momentum_sha256=None if cell is None else tensor_digest(cell.momentum),
                momentum_norm=0 if cell is None else sum(float(t.square().sum()) for t in cell.momentum.values())**.5,
                updates=0 if cell is None else cell.update_step)


def record(rows,logits,targets):
    predictions=logits.argmax(-1)
    losses=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').reshape_as(targets)
    return [dict(id=r['id'],predictions=p.tolist(),targets=t.tolist(),query_correct=p.eq(t).tolist(),
                 exact_match=float(p.eq(t).float().mean()),episode_exact_match=bool(p.eq(t).all()),
                 nll=float(n.mean()),per_query_nll=n.tolist())
            for r,p,t,n in zip(rows,predictions,targets,losses)]


def features(model):
    captured={}
    hooks=[]
    for name,module in [('swa',model.swa.final_norm),('ssm',model.ssm.final_norm),('memory',model.memory_norm)]:
        hooks.append(module.register_forward_hook(
            lambda _m,_i,o,name=name:captured.__setitem__(name,o.detach())))
    return captured,hooks


def branch_diagnostics(captured,gates):
    norms={k:float(v.norm(dim=-1).mean()) for k,v in captured.items()}
    cosine={f'{a}:{b}':float(F.cosine_similarity(x,y,dim=-1).mean())
            for a,x in captured.items() for b,y in captured.items() if a<b}
    return dict(gate_mean=gates.mean((0,1)).tolist(),gate_min=gates.amin((0,1)).tolist(),
                gate_max=gates.amax((0,1)).tolist(),output_norm=norms,output_cosine=cosine)


@torch.no_grad()
def delayed(model,rows,disabled,raw_gates=None):
    model.memory_mode='online_adapt'
    b=batch(rows,next(model.parameters()).device)
    captured,hooks=features(model)
    try:
        logits,_,gates,_=model.forward_state(b.input_ids,disabled=disabled)
    finally:
        for h in hooks:h.remove()
    q=b.query_mask
    labels=b.labels[q].reshape(len(rows),-1)
    selected=logits[q].reshape(len(rows),labels.shape[1],-1)
    if raw_gates is not None:
        raw_gates.append(gates.cpu().numpy())
    return record(rows,selected,labels),branch_diagnostics(captured,gates)


@torch.no_grad()
def query_session(model,suffix,positions,cell,ssm):
    model.reset_runtime()
    model.memory_mode='read_only'
    rt=wrapper(model,len(suffix),ssm)
    logits,state,_,_=model.forward_state(suffix,cell,disabled=model.disabled,runtime=rt,cache=True)
    if cell is not None and state is not cell:
        raise RuntimeError('Query mutated the persistent Cell')
    return logits[torch.arange(len(suffix),device=suffix.device)[:,None],positions]


@torch.no_grad()
def cross(model,rows,output=None,identity=None):
    device=next(model.parameters()).device
    prefix,suffix,positions,targets,padding=sessions(rows,model.chunk,device)
    model.reset_runtime()
    model.memory_mode='online_adapt'
    _,cell,_,rt=model.forward_state(prefix,disabled=model.disabled,cache=True)
    assert (cell is None)==('memory' in model.disabled)
    assert (rt.ssm is None)==('ssm' in model.disabled)
    natural=(rt.ssm is not None,cell is not None)
    matrix=CONDITIONS if model.arm=='F11' else {'natural':natural}
    results={}
    for name,(keep_ssm,keep_cell) in matrix.items():
        chosen_cell=cell if keep_cell else None
        chosen_ssm=rt.ssm if keep_ssm else None
        assert not keep_ssm or chosen_ssm is not None
        assert not keep_cell or chosen_cell is not None
        report=state_report(chosen_cell,chosen_ssm)
        assert report['ssm_bytes']>0 if keep_ssm else report['ssm_bytes']==0
        assert report['updates']>0 if keep_cell else report['updates']==0
        sync(device)
        started=time.perf_counter()
        actual=query_session(model,suffix,positions,chosen_cell,chosen_ssm)
        sync(device)
        results[name]=dict(rows=record(rows,actual,targets),state=report,seconds=time.perf_counter()-started,
                           padding=padding,swa_kv_carried_bytes=0)
        if output is not None:
            key={**identity,'condition':name,'batch_sha256':digest([r['id'] for r in rows]),
                 'branches':list(model.disabled),'revision_sha256':tensor_digest(model.state_dict())}
            begin=time.perf_counter()
            hashes=save_session(model,output/name,chosen_cell,chosen_ssm,key,actual)
            write(output/name/'receipt.json',dict(identity=key,hashes=hashes,suffix=suffix.tolist(),
                                                  positions=positions.tolist(),save_seconds=time.perf_counter()-begin))
    resources=model.resources(cell,rt)
    if cell is not None:
        resources['cell']['fast_delta']=sum(float((v-model.memory.initial[k]).square().sum()) for k,v in cell.fast.items())**.5
        assert resources['cell']['fast_delta']>0 and state_report(cell,None)['momentum_norm']>0
    if rt.ssm is not None:
        assert state_report(None,rt.ssm)['ssm_norm']>0
    return results,resources


def fresh(model,folder):
    device=next(model.parameters()).device
    results={}
    for f in sorted(folder.glob('*/receipt.json')):
        r=json.loads(f.read_text())
        key=r['identity']
        if key['branches']!=list(model.disabled) or key['revision_sha256']!=tensor_digest(model.state_dict()):
            raise ValueError('Static checkpoint revision differs')
        begin=time.perf_counter()
        cell,ssm,expected=load_session(model,f.parent,key,r['hashes'])
        actual=query_session(model,torch.tensor(r['suffix'],device=device),
                             torch.tensor(r['positions'],device=device),cell,ssm)
        torch.testing.assert_close(actual.cpu(),expected,**P['cache_tolerance'])
        results[f.parent.name]=dict(max_abs=float((actual.cpu()-expected).abs().max()),
                                   seconds=time.perf_counter()-begin,pid=__import__('os').getpid())
    assert len(results)==(4 if model.arm=='F11' else 1)
    return results


def train_diagnostic(model,rows):
    model.memory_mode='meta_train'
    model.train()
    b=batch(rows,next(model.parameters()).device)
    logits=model(b.input_ids)
    loss=F.cross_entropy(logits[b.query_mask],b.labels[b.query_mask])
    loss.backward()
    groups={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.isfinite().all()
            prefix=name.split('.')[0]
            groups[prefix]=groups.get(prefix,0.)+float(p.grad.square().sum())
        else:
            assert p.grad is None
    model.zero_grad(set_to_none=True)
    return {k:v**.5 for k,v in groups.items()}


def compare(metrics):
    """Episode-paired differences, stratified by the three evaluation banks."""
    rng=np.random.default_rng(P['bootstrap_seed'])
    out={}
    for task,arms in metrics.items():
        keys=list(arms['F00'])
        ids={seed:[r['id'] for r in arms['F00'][seed]] for seed in keys}
        out[task]={}
        for metric,sign in [('exact_match',1),('nll',-1)]:
            a=[]
            for arm in MASKS:
                assert list(arms[arm])==keys
                assert all([r['id'] for r in arms[arm][seed]]==ids[seed] for seed in keys)
                a.append(np.array([[r[metric]*sign for r in arms[arm][seed]] for seed in keys]))
            y00,y10,y01,y11=a
            entry=dict(per_arm={arm:{seed:float(arr[i].mean()) for i,seed in enumerate(keys)}
                                for arm,arr in zip(MASKS,a)})
            for name,delta in [('F11-F10',y11-y10),('F11-F01',y11-y01),
                               ('interaction',y11-y10-y01+y00)]:
                indices=rng.integers(delta.shape[1],size=(len(keys),P['bootstrap_replicates'],delta.shape[1]))
                samples=np.take_along_axis(delta[:,None,:],indices,axis=2).mean(axis=(0,2))
                entry[name]=dict(mean=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())
            out[task][metric]=entry
    return out


def publish(output,decision):
    if decision['identity']['source_commit']!=git('rev-parse','HEAD'):
        raise ValueError('Publication source identity mismatch')
    for name,sha in json.loads((output/'integrity.json').read_text()).items():
        if file_hash(output/name)!=sha:
            raise ValueError('Publication artifact hash mismatch: '+name)
    import shutil
    import tempfile
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    remote='https://github.com/ArcheLabs/mini-cells.git'
    with tempfile.TemporaryDirectory(prefix='h4-index-') as temp:
        checkout=Path(temp)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(checkout)],check=True)
        subprocess.run(['git','remote','set-url','origin',remote],cwd=checkout,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',decision['identity']['source_commit']],cwd=checkout,check=True)
        stage=checkout/'results/h4';stage.mkdir(parents=True)
        names=[]
        for f in sorted(output.rglob('*')):
            if f.is_file() and f.suffix in ('.json','.md','.jsonl','.npz') and f.name!='publication.json':
                name=f.relative_to(output);target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(f,target);names.append(str(name))
        write(stage/'publication-files.json',{name:file_hash(stage/name) for name in names})
        names.append('publication-files.json')
        spec=ExperimentSpec('results/h4','artifacts/experiments/epclm005-h4/'+decision['run_id'],
                            tuple(names),decision['branch'],FORMAT)
        dest=prepare_artifacts(checkout,'epclm005-h4',spec,None)
        push_results(checkout,dest,'epclm005-h4',decision['branch'],'GITHUB_TOKEN','origin/main')
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    write(output/'publication.json',dict(status='REMOTE_SHA_VERIFIED',branch=decision['branch'],sha=sha))
    write(output/'execution-status.json',dict(status='M4_H4_FAST_SCREENING_COMPLETE',
                                              branch=decision['branch'],sha=sha,signal=decision['signal']))


def budget_probe(output,sample,deadline):
    path=output/'t4-budget-probe.json'
    if path.exists():
        return json.loads(path.read_text())
    estimate=0.
    arms={}
    for arm in MASKS:
        model=build(arm,'cuda')
        torch.cuda.reset_peak_memory_stats()
        sync('cuda')
        started=time.monotonic()
        train_diagnostic(model,sample)
        sync('cuda')
        update=time.monotonic()-started
        model.eval()
        started=time.monotonic()
        delayed(model,sample,model.disabled)
        cross(model,sample)
        if arm=='F11':
            for disabled in ('swa','ssm','memory'):
                delayed(model,sample,(disabled,))
        sync('cuda')
        evaluation=time.monotonic()-started
        arms[arm]=dict(gradient_seconds=update,evaluation_batch_seconds=evaluation,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated())
        estimate+=32*update+64*evaluation
        del model
        torch.cuda.empty_cache()
    predicted=1.5*estimate+600
    result=dict(arms=arms,projected_seconds=predicted,optimizer_updates=0,
                status='PASS' if time.monotonic()+predicted<deadline else 'BLOCKED_RESOURCE_LIMIT')
    write(path,result)
    return result


def launch(output,execute,started):
    output.mkdir(parents=True,exist_ok=True)
    identity=source_identity()
    if identity['tokenizer_sha256']!=M4_CONFIG['tokenizer_sha256']:
        raise ValueError('Tokenizer mismatch')
    splits,evaluation,data=datasets()
    identity['dataset_sha256']=digest(data)
    write(output/'dataset-manifest.json',data)
    write(output/'execution-config.json',dict(protocol=P,model=M4_CONFIG['model'],
                                               fusion={**M4_CONFIG['fusion'],'seed':P['model_seed']},training=CONFIG))
    write(output/'source-audit.json',identity)
    common={}
    for arm in MASKS:
        model=build(arm,'cpu')
        states=model.state_dict()
        groups=('token_embedding','swa','ssm','memory','gate','post_mlp')
        common[arm]=dict(sha256=tensor_digest(states),disabled=list(model.disabled),
                         component_sha256={g:tensor_digest({k:v for k,v in states.items() if k.startswith(g+'.')})
                                           for g in groups},
                         resident=sum(p.numel() for p in model.parameters()),
                         active=sum(p.numel() for p in model.parameters() if p.requires_grad))
        assert common[arm]['resident']==10065158
        del model
    assert len({x['sha256'] for x in common.values()})==1
    write(output/'initialization.json',common)
    if not execute:
        write(output/'decision.json',dict(format=FORMAT,status='CPU_DATA_AND_INITIALIZATION_PASS',
                                         identity=identity,hypotheses='NOT_EVALUATED'))
        return
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Actual Tesla T4 required for official execution')
    if git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit source before official execution')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    identity.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                    driver=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=driver_version',
                                                    '--format=csv,noheader'],text=True,timeout=30).strip(),
                    precision='fp32',tf32=False)
    run_id='h4-l1-'+digest(identity)[:20]
    decision=dict(format=FORMAT,run_id=run_id,branch='kaggle/'+run_id,identity=identity,status='INCOMPLETE')
    header=output/'execution-identity.json'
    if header.exists() and json.loads(header.read_text())!=identity:
        raise ValueError('Output identity mismatch')
    write(header,identity)
    from preflight import run_checks
    for device in ('cpu','cuda'):
        receipt=output/('preflight-'+device+'.json')
        if not receipt.exists():
            proof=run_checks(output/('preflight-'+device),device)
            write(receipt,proof)
        assert json.loads(receipt.read_text())['status']=='PASS'
    deadline=started+P['execution_seconds']
    probe=budget_probe(output,splits['validation'][:4],deadline)
    if probe['status']!='PASS':
        raise RuntimeError('BLOCKED_RESOURCE_LIMIT: T4 forecast exceeds the L1 session budget')
    metrics={'delayed':{},'cross_session':{},'persistence':{}}
    sampler=Sampler(splits['train'],CONFIG['seed'],CONFIG['batch'])
    expected=[]
    planned=[]
    for _ in range(32):
        rows=sampler.propose()
        expected.append(digest([r['input_ids'] for r in rows]))
        planned.append(dict(ids=[r['id'] for r in rows],input_sha256=expected[-1],
                            target_sha256=digest([r['labels'] for r in rows])))
        sampler.resolve(True,{'committed_updates':0},{})
    write(output/'training-order.json',planned)
    for arm in MASKS:
        folder=output/arm
        model=build(arm,'cuda')
        frozen=frozen_hash(model)
        arm_identity={**identity,'arm':arm,'initial_sha256':common[arm]['sha256']}
        begin=time.monotonic()
        summary=fit(model,splits['train'],splits['validation'],CONFIG,folder,arm_identity,deadline)
        train_wall=time.monotonic()-begin
        assert summary['status']=='COMPLETE' and summary['ledger']['input_tokens']==65536
        assert summary['ledger']['query_target_tokens']==1024
        assert summary['ledger']['committed_updates']==32
        assert [r['input_sha256'] for r in summary['history']]==expected
        assert frozen==frozen_hash(model)
        fit_seconds=sum(r['compute_seconds'] for r in summary['history'])
        checkpoint=folder/'checkpoint.pt'
        checkpoint_sha=file_hash(checkpoint)
        assert checkpoint_sha==summary['checkpoint']['sha256']
        write(folder/'inner-events-this-session.json',model.events)
        diagnostic=train_diagnostic(model,splits['validation'][:1])
        model.eval()
        metrics['delayed'][arm]={}
        metrics['cross_session'][arm]={}
        raw=[]
        for seed,rows in evaluation.items():
            direct,cross_rows,diagnostics,resources=[],[],[],[]
            ablation={k:[] for k in ('ssm','memory','swa')} if arm=='F11' else {}
            for i in range(0,len(rows),4):
                if time.monotonic()>=deadline:
                    raise TimeoutError('Evaluation budget exhausted; retained checkpoint')
                group=rows[i:i+4]
                scored,diag=delayed(model,group,model.disabled,raw)
                direct.extend(scored)
                for branch,records in ablation.items():
                    records.extend(delayed(model,group,(branch,))[0])
                diagnostics.append(diag)
                session_folder=folder/'sessions' if not i and seed==next(iter(evaluation)) else None
                results,resource=cross(model,group,session_folder,{**arm_identity,'checkpoint_sha256':checkpoint_sha})
                natural='C11' if arm=='F11' else 'natural'
                cross_rows.extend(results[natural]['rows'])
                resources.append(resource)
                if arm=='F11':
                    for condition,res in results.items():
                        metrics['persistence'].setdefault(condition,{}).setdefault(seed,[]).extend(res['rows'])
                write(folder/'session-observations'/f'{seed}-{i}.json',results)
            metrics['delayed'][arm][seed]=direct
            metrics['cross_session'][arm][seed]=cross_rows
            write(folder/f'diagnostics-{seed}.json',dict(branch=diagnostics,resources=resources,inference_ablations=ablation))
        np.savez_compressed(folder/'gate-weights.npz',weights=np.concatenate(raw))
        write(folder/'arm-summary.json',dict(frozen_sha256=frozen,gradient_norm_by_module=diagnostic,
                    training_wall_seconds_this_session=train_wall,evaluation_wall_seconds=time.monotonic()-begin-train_wall,checkpoint_sha256=checkpoint_sha,
                    train_compute_seconds=fit_seconds,training_tokens_per_compute_second=65536/fit_seconds,
                    peak_allocated_bytes=summary.get('training_peak_allocated_bytes')))
        del model
        subprocess.run([sys.executable,__file__,'--output',str(output),'--restore-arm',arm],
                       check=True,timeout=max(1,deadline-time.monotonic()))
    comparison=compare({k:metrics[k] for k in ('delayed','cross_session')})
    persistence=compare({'persistence':dict(zip(MASKS,(metrics['persistence'][k] for k in CONDITIONS)))})['persistence']
    comparison['persistence']={k:{name.replace('F','C'):value for name,value in item.items()}
                               for k,item in persistence.items()}
    for item in comparison['persistence'].values():
        item['per_arm']={arm.replace('F','C'):value for arm,value in item['per_arm'].items()}
    write(output/'metrics.json',metrics)
    write(output/'comparison.json',comparison)
    delayed_means={a:float(np.mean([r['exact_match'] for bank in metrics['delayed'][a].values() for r in bank])) for a in MASKS}
    cross_means={a:float(np.mean([r['exact_match'] for bank in metrics['cross_session'][a].values() for r in bank])) for a in MASKS}
    floor=all(x<.05 for x in delayed_means.values())
    ceiling=all(x>=.95 for x in delayed_means.values())
    signal='INCONCLUSIVE' if floor or ceiling else 'NO_POSITIVE_SCREENING_SIGNAL'
    gains=[]
    if not floor and not ceiling:
        gains=[means['F11']-max(means['F10'],means['F01']) for means in (delayed_means,cross_means)]
        if any(x>0 for x in gains):
            signal='PARTIAL_SIGNAL'
        # A positive designation requires human review of collapse and resource cost.
    intervals=[comparison[k]['exact_match']['interaction']['ci95'] for k in ('delayed','cross_session')]
    positive_quality_candidate=(len(gains)==2 and min(gains)>=P['minimum_practical_gain'] and
                                any(lo>0 for lo,_ in intervals) and all(hi>=0 for _,hi in intervals))
    decision.update(status='SCREENING_EXECUTED_REVIEW_REQUIRED',signal=signal,
                    positive_quality_candidate=positive_quality_candidate,
                    practical_threshold=P['minimum_practical_gain'],delayed=delayed_means,
                    cross_session=cross_means,gates={'0':'PASS','1':'PASS','2':'PASS','3':'PASS',
                                                     '4':'PASS','5':'PASS','6':'REVIEW_REQUIRED'},
                    training_seed_count=1,test_status='SEALED_NOT_ACCESSED',
                    escalation='Review level budget before any L2/L3 execution' if floor or ceiling else 'NOT_TRIGGERED')
    write(output/'decision.json',decision)
    (output/'REPORT.md').write_text('# EPCLM-005 H4 L1 Screening\n\n'+
        'One training seed. Paired episode intervals condition on that seed.\n\n'+
        'See comparison.json and the per-arm reports for evidence. Review the joint decision before confirmation.\n\n'+
        '~~~json\n'+json.dumps(decision,indent=2)+'\n~~~\n')
    write(output/'integrity.json',{str(f.relative_to(output)):file_hash(f) for f in output.rglob('*')
                                   if f.is_file() and f.name not in ('integrity.json','error.json','publication.json','execution-status.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--started',type=float,default=time.monotonic())
    parser.add_argument('--restore-arm',choices=list(MASKS))
    args=parser.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT))
    try:
        if args.restore_arm:
            arm=args.restore_arm
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            folder=args.output/arm
            model=build(arm,'cuda').eval()
            summary=json.loads((folder/'arm-summary.json').read_text())
            assert file_hash(folder/'checkpoint.pt')==summary['checkpoint_sha256']
            model.load_state_dict(load_checkpoint(folder/'checkpoint.pt')['model'])
            write(folder/'fresh-restore.json',fresh(model,folder/'sessions'))
        elif args.publish and (args.output/'decision.json').exists() and json.loads(
                (args.output/'decision.json').read_text())['status']=='SCREENING_EXECUTED_REVIEW_REQUIRED':
            publish(args.output,json.loads((args.output/'decision.json').read_text()))
        else:
            launch(args.output,args.execute,args.started)
            if args.publish:
                publish(args.output,json.loads((args.output/'decision.json').read_text()))
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'error.json',dict(status='INCOMPLETE',error=repr(error)))
        raise
