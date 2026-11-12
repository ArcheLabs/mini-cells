"""Bounded contextual H4 screening and publication entry."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from model import BASE, Contextual, MASKS, A_LENGTH, B_LENGTH
from adapter import M4, frozen_hash, tensor_digest
from dataset import bank
from evaluation import evaluate_batch,persist,restore
from data import batch,digest,file_hash
from e0 import ROOT,Sampler,load_checkpoint,masked_loss,write
from training import fit
from screening import git,source_identity,sync

P=json.loads((BASE/'config.json').read_text())
S=json.loads((M4/'configs/smoke.json').read_text())
TRAIN={**S['training'],'seed':P['model_seed'],'updates':P['updates'],'batch':P['batch']}
FORMAT='minicells.epclm005.h4-l1c.v1'


def build(arm,device,fixture=False):
    model=S['model']
    fusion={**S['fusion'],'seed':P['model_seed']}
    if fixture:
        model=dict(vocab=1024,dim=16,layers=2,heads=2,kv_heads=2,ffn=32,window=32)
        fusion.update(ssm_layers=1,state_size=4,memory_dim=4,memory_hidden=8)
    return Contextual(arm,model,fusion).to(device)


def train_sessions(model,train,validation,folder,identity,deadline,config=TRAIN):
    """The adapter executes A and B inside each transactional trainer step."""
    for row in train+validation:
        if row['session_lengths']!=[A_LENGTH,B_LENGTH] or any(y!=-100 for y in row['labels'][:A_LENGTH]):
            raise ValueError('Invalid multi-session training contract')
    return fit(model,train,validation,config,folder,identity,deadline)


def gradient_probe(model,rows):
    model.train()
    b=batch(rows,next(model.parameters()).device)
    loss=masked_loss(model(b.input_ids),b.labels)
    assert loss.isfinite()
    loss.backward()
    norms={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.isfinite().all(),name
            part=name.split('.')[0]
            norms[part]=norms.get(part,0.)+float(p.grad.square().sum())
        else:
            assert p.grad is None
    model.zero_grad(set_to_none=True)
    return {k:v**.5 for k,v in norms.items()}


def expected_order(rows):
    sampler=Sampler(rows,TRAIN['seed'],TRAIN['batch'])
    order=[]
    for _ in range(TRAIN['updates']):
        group=sampler.propose()
        order.append(dict(ids=[r['id'] for r in group],inputs=digest([r['input_ids'] for r in group]),
                          labels=digest([r['labels'] for r in group])))
        sampler.resolve(True,{'committed_updates':0},{})
    return order


def compare(arms):
    rng=np.random.default_rng(P['bootstrap_seed'])
    seeds=list(arms['F00'])
    for arm in MASKS:
        for seed in seeds:
            assert [x['id'] for x in arms[arm][seed]]==[x['id'] for x in arms['F00'][seed]]
    results={}
    for name in ('source_a_em','source_b_em','exact_match','episode_em','nll'):
        sign=-1 if name=='nll' else 1
        values=np.array([[[sign*r[name] for r in arms[arm][seed]] for seed in seeds] for arm in MASKS])
        draws=rng.integers(values.shape[2],size=(len(seeds),P['bootstrap_replicates'],values.shape[2]))
        bootstrap=np.stack([np.take_along_axis(v[:,None,:],draws,axis=2).mean((0,2)) for v in values])
        means=values.mean((1,2))

        def contrasts(m):
            a,b,c,d=m
            return dict(interaction=d-b-c+a,over_strongest=d-np.maximum(b,c),F11_minus_F10=d-b,F11_minus_F01=d-c)

        observed=contrasts(means)
        samples=contrasts(bootstrap)
        results[name]=dict(by_arm=dict(zip(MASKS,means.tolist())),per_seed={
            arm:dict(zip(seeds,values[i].mean(1).tolist())) for i,arm in enumerate(MASKS)},
            contrasts={k:dict(mean=float(v),ci95=np.quantile(samples[k],[.025,.975]).tolist())
                       for k,v in observed.items()},direction='negative_nll' if sign<0 else 'higher_is_better')
    return results


def predict_profile(rows,model):
    tokens=Counter(p for r in rows for p in r['predictions'])
    return dict(argmax=dict(tokens),count=sum(tokens.values()),unique=len(tokens),
                token_432_embedding_norm=float(model.token_embedding.weight[432].detach().norm()),
                token_432_logit_mean=float(np.mean([x for r in rows for x in r['token_432_logit']])))


def identity_for(data):
    identity=source_identity()
    identity['config_sha256']=digest([P,S,TRAIN])
    identity['dataset_sha256']=digest(data)
    assert identity['tokenizer_sha256']==S['tokenizer_sha256']
    return identity


def budget_probe(output,rows,deadline):
    path=output/'budget-probe.json'
    if path.exists():
        return json.loads(path.read_text())
    records={}
    for arm in MASKS:
        model=build(arm,'cuda')
        torch.cuda.reset_peak_memory_stats()
        sync('cuda')
        t=time.monotonic()
        gradient_probe(model,rows)
        sync('cuda')
        update=time.monotonic()-t
        t=time.monotonic()
        evaluate_batch(model,rows,arm=='F11')
        sync('cuda')
        evaluation=time.monotonic()-t
        records[arm]=dict(backward_seconds=update,evaluation_seconds=evaluation,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del model
        torch.cuda.empty_cache()
    estimate=1.5*sum(32*r['backward_seconds']+64*r['evaluation_seconds'] for r in records.values())+600
    result=dict(status='PASS' if time.monotonic()+estimate<deadline else 'BLOCKED_RESOURCE_LIMIT',
                estimated_seconds=estimate,arms=records,optimizer_updates=0)
    write(path,result)
    return result


def publish(out,decision):
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    identity=decision['identity']
    assert identity['source_commit']==git('rev-parse','HEAD')
    for name,sha in json.loads((out/'integrity.json').read_text()).items():
        assert file_hash(out/name)==sha
    with tempfile.TemporaryDirectory(prefix='h4-l1c-publish-') as tmp:
        checkout=Path(tmp)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(checkout)],check=True)
        subprocess.run(['git','remote','set-url','origin','https://github.com/ArcheLabs/mini-cells.git'],cwd=checkout,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',identity['source_commit']],cwd=checkout,check=True)
        stage=checkout/'results/l1c';stage.mkdir(parents=True)
        names=[]
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.md') and path.name not in ('publication.json','error.json'):
                name=path.relative_to(out);target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target);names.append(str(name))
        spec=ExperimentSpec('results/l1c','artifacts/experiments/epclm005-h4-l1c/'+decision['run_id'],
                            tuple(names),decision['branch'],FORMAT)
        result=prepare_artifacts(checkout,'epclm005-h4-l1c',spec,None)
        push_results(checkout,result,'epclm005-h4-l1c',decision['branch'],'GITHUB_TOKEN','origin/main')
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    write(out/'publication.json',dict(status='REMOTE_SHA_VERIFIED',branch=decision['branch'],sha=sha))


def run(out,started):
    out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Actual Tesla T4 required')
    if git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Official execution requires committed source')
    train,validation,evaluation,data=bank()
    order=expected_order(train)
    identity=identity_for(data)
    identity.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                    driver=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=driver_version',
                                                     '--format=csv,noheader'],text=True,timeout=30).strip(),
                    precision='fp32',tf32=False)
    header=out/'execution-identity.json'
    if header.exists() and json.loads(header.read_text())!=identity:
        raise ValueError('L1-C execution identity mismatch')
    write(header,identity)
    write(out/'dataset-manifest.json',data)
    write(out/'training-order.json',order)
    write(out/'execution-config.json',dict(config=P,training=TRAIN,model=S['model'],fusion=S['fusion']))
    from checks_l1c import checks
    for device in ('cpu','cuda'):
        path=out/f'preflight-{device}.json'
        if not path.exists():
            write(path,checks(out/f'preflight-{device}',device))
    deadline=started+P['execution_seconds']
    probe=budget_probe(out,validation[:4],deadline)
    if probe['status']!='PASS':
        raise RuntimeError('T4 budget probe rejected the fixed L1-C workload')
    arm_results={}
    initial=None
    for arm in MASKS:
        folder=out/arm
        folder.mkdir(exist_ok=True)
        result_path=folder/'evaluation.json'
        if result_path.exists() and (folder/'fresh-restore.json').exists():
            existing=json.loads(result_path.read_text())
            assert file_hash(folder/'checkpoint.pt')==existing['checkpoint_sha256']
            arm_results[arm]=existing
            continue
        model=build(arm,'cuda')
        init_hash=tensor_digest(model.state_dict())
        if initial is None:initial=init_hash
        assert initial==init_hash
        resident=sum(p.numel() for p in model.parameters())
        assert resident==10065158
        active=sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen=frozen_hash(model)
        arm_identity={**identity,'arm':arm,'initial_sha256':init_hash}
        pre=folder/'initial-query.json'
        if not pre.exists():
            rows=evaluate_batch(model,validation[:4])['metrics']['natural']
            write(pre,dict(rows=rows,profile=predict_profile(rows,model)))
        t=time.monotonic()
        fit_result=train_sessions(model,train,validation,folder,arm_identity,deadline)
        training_wall=time.monotonic()-t
        ledger=fit_result['ledger']
        assert fit_result['status']=='COMPLETE'
        assert (ledger['committed_updates'],ledger['input_tokens'],ledger['query_target_tokens'])==(
                P['updates'],P['input_budget'],P['query_budget'])
        assert [r['input_sha256'] for r in fit_result['history']]==[r['inputs'] for r in order]
        assert frozen==frozen_hash(model)
        sha=file_hash(folder/'checkpoint.pt')
        assert sha==fit_result['checkpoint']['sha256']
        gradients=gradient_probe(model,validation[:1])
        result=dict(checkpoint_sha256=sha,initial_sha256=init_hash,resident_parameters=resident,
                    active_parameters=active,gradient_probe=gradients,frozen_sha256=frozen,
                    training_wall_seconds_this_call=training_wall,
                    training_tokens_per_compute_second=P['input_budget']/sum(r['compute_seconds'] for r in fit_result['history']),
                    training_peak_allocated_bytes=fit_result['training_peak_allocated_bytes'],
                    checkpoint_bytes=fit_result['checkpoint']['bytes'],evaluation={},ledger=ledger)
        model.eval()
        first_seed=next(iter(evaluation))
        persist(model,evaluation[first_seed][:1],folder/'session',
                {**arm_identity,'checkpoint_sha256':sha},P['tolerance'])
        for seed,rows in evaluation.items():
            observations=[]
            sync('cuda')
            started_eval=time.monotonic()
            for i in range(0,len(rows),4):
                if time.monotonic()>=deadline:
                    raise TimeoutError('L1-C evaluation deadline')
                observations.append(evaluate_batch(model,rows[i:i+4],arm=='F11'))
            sync('cuda')
            result['evaluation'][seed]=dict(observations=observations,seconds=time.monotonic()-started_eval)
        scored=[r for bank in result['evaluation'].values() for ob in bank['observations'] for r in ob['metrics']['natural']]
        result['prediction_profile']=predict_profile(scored,model)
        write(result_path,result)
        del model
        subprocess.run([sys.executable,__file__,'--output',str(out),'--restore-arm',arm],check=True,
                       timeout=max(1,deadline-time.monotonic()))
        arm_results[arm]=result
    assert len({r['initial_sha256'] for r in arm_results.values()})==1
    metrics={a:{s:[r for ob in v['observations'] for r in ob['metrics']['natural']]
                for s,v in result['evaluation'].items()} for a,result in arm_results.items()}
    assert all(sum(len(v) for v in r.values())==192 for r in metrics.values())
    comparison=compare(metrics)
    a=comparison['source_a_em']['contrasts']['F11_minus_F10']['mean']
    b=comparison['source_b_em']['contrasts']['F11_minus_F01']['mean']
    gain=comparison['exact_match']['contrasts']['over_strongest']['mean']
    collapse=arm_results['F11']['prediction_profile']['unique']==1 or all(
        r['prediction_profile']['unique']==1 for r in arm_results.values())
    signal='INCONCLUSIVE' if collapse else (
        'POSITIVE_SCREENING_SIGNAL' if min(a,b,gain)>=P['minimum_practical_gain'] else 'NO_POSITIVE_SIGNAL')
    run_id='h4-l1c-'+digest(identity)[:20]
    decision=dict(format=FORMAT,status='SCREENING_EXECUTED',signal=signal,identity=identity,
                  run_id=run_id,branch='kaggle/'+run_id,training_seed_count=1,test_status='SEALED_NOT_ACCESSED',
                  collapse=collapse)
    write(out/'metrics.json',metrics)
    write(out/'comparison.json',comparison)
    write(out/'decision.json',decision)
    lines=['# H4 L1-C contextual memory screening','',f'Status {signal}. One training seed.','',
           '| Arm | A recall | B recall | Query EM | Episode EM |','|---|---:|---:|---:|---:|']
    for arm in MASKS:
        vals=[comparison[key]['by_arm'][arm] for key in ('source_a_em','source_b_em','exact_match','episode_em')]
        lines.append('| '+arm+' | '+' | '.join(f'{x:.2%}' for x in vals)+' |')
    lines+=['','Paired interactions, source-specific gains and intervals are in comparison.json. '
            'Gate, top-k, rejection, ablation and resource observations are in the per-arm records.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'integrity.json',{str(f.relative_to(out)):file_hash(f) for f in out.rglob('*')
                                if f.is_file() and f.name not in ('integrity.json','publication.json','error.json')})
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--started',type=float,default=time.monotonic())
    parser.add_argument('--restore-arm',choices=list(MASKS))
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT))
    try:
        if args.restore_arm:
            folder=args.output/args.restore_arm
            model=build(args.restore_arm,'cuda').eval()
            result=json.loads((folder/'evaluation.json').read_text())
            assert file_hash(folder/'checkpoint.pt')==result['checkpoint_sha256']
            model.load_state_dict(load_checkpoint(folder/'checkpoint.pt')['model'])
            write(folder/'fresh-restore.json',restore(model,folder/'session',P['tolerance']))
        else:
            done=args.output/'decision.json'
            decision=json.loads(done.read_text()) if done.exists() and (args.output/'integrity.json').exists() else run(args.output,args.started)
            publish(args.output,decision)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'error.json',dict(status='INCOMPLETE',error=repr(error)))
        raise
