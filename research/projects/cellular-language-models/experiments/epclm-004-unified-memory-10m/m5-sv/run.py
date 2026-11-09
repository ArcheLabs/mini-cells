"""One bounded, resumable M5-SV run; never enters M6 or expands budgets."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'src'))
from sv_data import bank, read_bank, training_rows, validation_rows, audit_splits, sha_file
from sv_model import ValidationModel, evaluate_episode
from sv_analysis import condition_report, scientific_decision
from data import digest, file_hash
from e0 import ROOT, write, load_checkpoint
from training import fit


class Pause(RuntimeError): pass


def check_time(deadline, reserve=120):
    if time.monotonic()+reserve>=deadline:raise Pause('session budget reached; resume exact saved Output')


def construct(protocol, arm, seeds):
    if arm not in protocol['arms'] and arm!='E7-Oracle':raise ValueError('unknown experiment arm')
    return ValidationModel(protocol['model'],policy=protocol['arms'].get(arm,'innovation'),
        summary_enabled=True,capacity=protocol['capacity'],summary_tokens=protocol['summary_tokens'],
        chunk=protocol['chunk'],state_size=protocol['state_size'],read_top_k=protocol['read_top_k'],
        seed=seeds['model_seed'],store_seed=seeds['store_seed'],beta=protocol['predictor_beta'])


def source_identity(protocol):
    files=[BASE/'run.py',*sorted((BASE/'src').glob('*.py')),*sorted((BASE/'tests').glob('*.py'))]
    return {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config_sha256':digest(protocol),'tokenizer_sha256':protocol['tokenizer_sha256'],
        'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        'implementation':{str(p.relative_to(BASE)):file_hash(p) for p in files}}


def verify_baseline():
    for p,h in json.loads((BASE/'configs/baseline-lock.json').read_text())['files'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('frozen M5 dependency changed: '+p)


def resume_output(out, identity):
    """Use exact-identity attached outputs, never guessed latest-version downloads."""
    manifest=out/'run-manifest.json'
    if manifest.exists():
        saved_manifest=json.loads(manifest.read_text())
        if saved_manifest['identity']!=identity:raise ValueError('working Output belongs to another identity')
        if not saved_manifest.get('requires_attached_output',False):return
    candidates=[]
    root=Path('/kaggle/input')
    for path in root.rglob('run-manifest.json') if root.exists() else []:
        try:r=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if r.get('identity')==identity:candidates.append((r.get('saved_at_utc',''),path))
    if candidates:
        path=max(candidates,key=lambda item:(item[0],str(item[1])))[1]
        shutil.copytree(path.parent,out,dirs_exist_ok=True)
        return
    branch='refs/heads/kaggle/epclm004-m5-sv-'+identity['source_commit'][:12]
    found=subprocess.check_output(['git','ls-remote','origin',branch],cwd=ROOT,text=True)
    if found.strip():raise RuntimeError('BLOCKED_MISSING_OUTPUT: this identity has a prior run; attach its saved Kaggle Output to resume, not retrain')


def functional_gate(out, identity):
    path=out/'audit/sv0-functional.json';key=digest(identity)
    if path.exists():
        r=json.loads(path.read_text())
        if r['identity']==key and r['status']=='PASS' and file_hash(out/'audit/sv0-functional.log')==r['log_sha256']:return r
    path.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(BASE/'tests')],cwd=ROOT,
        env={**os.environ,'OMP_NUM_THREADS':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'},text=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (out/'audit/sv0-functional.log').write_text(result.stdout)
    if result.returncode:raise RuntimeError('SV0 functional regression failed; inspect audit/sv0-functional.log')
    r={'status':'PASS','identity':key,'log_sha256':file_hash(out/'audit/sv0-functional.log'),
       'scope':'synthetic CPU functional checks, not scientific effectiveness'}
    write(path,r);return r


def freeze_data(out, protocol):
    path=out/'datasets';path.mkdir(parents=True,exist_ok=True)
    ref=construct(protocol,'E6',protocol['seeds'][0]);embedding=ref.token_embedding.weight.detach().cpu().clone();del ref
    dev_conditions=[{'name':'development-'+str(n),'length':n,'bindings':8,'queries':4,'interference':1.} for n in (256,512)]
    dev=bank(path/'development.jsonl',dev_conditions,'validation',protocol['split_seeds']['validation'],4,embedding)
    test=bank(path/'test.jsonl',protocol['conditions'],'test',protocol['split_seeds']['test'],protocol['test_episodes_per_condition'],embedding)
    manifests={};all_train=[]
    for seed in [protocol['pilot'],*protocol['seeds']]:
        budget=protocol['pilot']['input_tokens_per_arm'] if seed is protocol['pilot'] else protocol['training']['input_tokens_per_arm_per_seed']
        rows=training_rows(seed['data_seed'],budget,protocol['training']['length']);all_train.extend(rows)
        manifests[str(seed['data_seed'])]={'input_tokens':sum(len(r['input_ids']) for r in rows),'episodes':len(rows),'sha256':digest(rows)}
    isolation=audit_splits({'train':all_train,'validation':read_bank(path/'development.jsonl'),'test':read_bank(path/'test.jsonl')})
    result={'development':dev,'test':test,'training':manifests,'isolation':isolation,'tokenizer_sha256':protocol['tokenizer_sha256']}
    write(out/'dataset-manifest.json',result);return result


def observed_result(model, row, path, identity, deadline, **kwargs):
    path=Path(path);key=digest({'identity':identity,'episode':row['id'],'kwargs':kwargs,
                             'capacity':model.capacity,'policy':model.policy})
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=key or digest(saved['result'])!=saved['sha256']:raise ValueError('evaluation artifact identity/hash differs')
        return saved['result']
    check_time(deadline,300 if len(row['history'])>=8192 else 90)
    runtime_folder=path.parent/'runtime'
    result=evaluate_episode(model,row,runtime_folder=str(runtime_folder) if not (runtime_folder/'manifest.json').exists() else None,**kwargs)
    write(path,{'identity':key,'result':result,'sha256':digest(result)})
    return result


def development_diagnostics(out, protocol, identity, deadline, seeds, checkpoint_root=None):
    """Fixed-checkpoint policies, plus independent E6 recoverability; no Test."""
    complete=out/'metrics/development'/str(seeds['model_seed'])/'complete.json'
    if complete.exists():
        saved=json.loads(complete.read_text())
        if saved.get('identity')!=identity:raise ValueError('development diagnostics identity differs')
        return saved
    rows=list(read_bank(out/'datasets/development.jsonl'));raw=out/'raw-evaluation/development'/str(seeds['model_seed'])
    e6=construct(protocol,'E6',seeds).cuda()
    if checkpoint_root is not None:e6.load_state_dict(load_checkpoint(checkpoint_root/'E6/checkpoint.pt')['model'])
    baseline=[observed_result(e6,r,raw/'E6'/(r['id']+'.json'),identity,deadline,recover_all=True,no_history=True) for r in rows]
    del e6;gc.collect();torch.cuda.empty_cache()
    model=construct(protocol,'E7-Innovation',seeds).cuda()
    if checkpoint_root is not None:model.load_state_dict(load_checkpoint(checkpoint_root/'E7-Innovation/checkpoint.pt')['model'])
    reports={}
    for capacity in protocol['capacity_tiers']:
        model.capacity=capacity
        for arm in ('E7-Random','E7-FIFO','E7-Innovation','E7-Oracle'):
            model.policy=protocol['arms'].get(arm,'innovation')
            values=[observed_result(model,r,raw/arm/str(capacity)/(r['id']+'.json'),identity,deadline,oracle=arm=='E7-Oracle') for r in rows]
            report=condition_report(values,baseline)
            write(complete.parent/(arm+'-'+str(capacity)+'.json'),report);reports[arm+':'+str(capacity)]=report['query_summary']
    result={'identity':identity,'status':'PASS','scope':'SV1/SV2 measurement execution on development only; effectiveness not assumed',
            'weights':'fresh preflight' if checkpoint_root is None else 'trained checkpoint; policy interventions share Innovation weights',
            'reports':reports,'random_guess_expected_em':1/256,
            'random_guess_realized_em':random_guess(rows,protocol['decision']['bootstrap_seed']),
            'no_history_em':sum(q['correct'] for r in baseline for q in r['no_history'])/sum(len(r['no_history']) for r in baseline)}
    write(complete,result);del model;gc.collect();torch.cuda.empty_cache();return result


def random_guess(rows, seed):
    import random
    rng=random.Random(seed);correct=total=0
    for row in rows:
        for q in row['queries']:correct+=int(264+rng.randrange(256)==q['expected_value_id']);total+=1
    return correct/total


def train_arm(out, protocol, arm, seeds, budget, identity, deadline, pilot=False):
    folder=out/('pilot' if pilot else 'training')/str(seeds['model_seed'])/arm
    rows=training_rows(seeds['data_seed'],budget,protocol['training']['length'])
    val=validation_rows(protocol['split_seeds']['validation'],protocol['validation_episodes'])
    training_id={'run':identity,'arm':arm,'seeds':{k:seeds[k] for k in ('model_seed','data_seed','store_seed')},
                 'input_budget':budget,'train_sha256':digest(rows),'validation_sha256':digest(val)}
    curve_path=folder/'curve.json';curve=json.loads(curve_path.read_text()) if curve_path.exists() else []
    model=construct(protocol,arm,seeds)
    initial=hashlib.sha256()
    for name,value in model.state_dict().items():initial.update(name.encode());initial.update(value.detach().cpu().contiguous().numpy().tobytes())
    initial_record={'seeds':{k:seeds[k] for k in ('model_seed','data_seed','store_seed')},'static_parameter_sha256':initial.hexdigest()}
    initial_path=folder/'initialization.json'
    if initial_path.exists() and json.loads(initial_path.read_text())!=initial_record:raise ValueError('paired initialization changed')
    write(initial_path,initial_record);model=model.cuda()
    config={k:v for k,v in protocol['training'].items() if k not in ('input_tokens_per_arm_per_seed','total_input_tokens','length','curve_fractions')}
    config.update(seed=seeds['data_seed'],updates=len(rows))
    targets=[len(rows)] if pilot else sorted(set(math.ceil(f*len(rows)) for f in protocol['training']['curve_fractions']))
    try:
        for target in targets:
            prior=next((r for r in curve if r['target_updates']==target),None)
            if prior:
                if prior['identity']!=training_id:raise ValueError('training curve identity differs')
                continue
            existing_metrics=folder/'metrics.json'
            if existing_metrics.exists() and json.loads(existing_metrics.read_text())['ledger']['committed_updates']>target:
                raise ValueError('missing historical validation boundary; refusing to invent a convergence curve')
            check_time(deadline,180)
            metrics=fit(model,rows,val,config,folder,training_id,deadline-150,stop_after=target)
            if metrics['ledger']['committed_updates']<target:raise Pause('training checkpoint saved at session deadline')
            metrics['parameters']={'static_parameters':sum(p.numel() for p in model.parameters()),
                'breakdown':model.resources(model.new_runtime(1))['static_parameter_breakdown'],'tied_embedding_head':True}
            write(folder/'metrics.json',metrics)
            prior_resources=json.loads((folder/'resources.json').read_text()) if (folder/'resources.json').exists() else {}
            write(folder/'resources.json',{'static_parameters':sum(p.numel() for p in model.parameters()),
                'static_weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                'training_peak_allocated_bytes':max(prior_resources.get('training_peak_allocated_bytes') or 0,metrics['training_peak_allocated_bytes'] or 0),
                'checkpoint':metrics['checkpoint'],'checkpoint_write':metrics['checkpoint_policy'],
                'training_compute_tokens_per_second':metrics['ledger']['input_tokens']/sum(h['compute_seconds'] for h in metrics['history']),
                'training_flops':None,'flops_reason':'no complete validated operator/FLOPs counter for this reference path',
                'precision':'fp32','tf32':False})
            curve.append({'target_updates':target,'identity':training_id,'validation':metrics['validation'],
                          'ledger':metrics['ledger'],'checkpoint_sha256_at_boundary':metrics['checkpoint']['sha256']})
            write(curve_path,curve)
        saved=load_checkpoint(folder/'checkpoint.pt');metrics=json.loads((folder/'metrics.json').read_text())
        if saved['identity']!=training_id or saved['ledger']!=metrics['ledger'] or file_hash(folder/'checkpoint.pt')!=metrics['checkpoint']['sha256']:raise ValueError('completed checkpoint failed identity/hash/ledger validation')
        assert saved['ledger']['input_tokens']==budget and saved['ledger']['committed_updates']==len(rows)
        return metrics,curve
    finally:del model;gc.collect();torch.cuda.empty_cache()


def pilot_and_freeze(out, protocol, identity, data, deadline):
    receipt=out/'freeze.json'
    expected={'identity':identity,'data':data,'protocol':protocol}
    if receipt.exists():
        frozen=json.loads(receipt.read_text())
        if frozen['frozen']!=expected:raise ValueError('immutable scientific freeze differs')
        return frozen
    results={}
    for arm in protocol['arms']:
        metrics,_=train_arm(out,protocol,arm,protocol['pilot'],protocol['pilot']['input_tokens_per_arm'],identity,deadline,True)
        before=metrics['initial_evaluation'];after=metrics['validation'];p=protocol['pilot']
        passed=before['nll']-after['nll']>=p['min_validation_nll_improvement'] and after['query_exact_match']*after['query_targets']>=p['min_correct_validation_queries']
        results[arm]={'learnability_pass':passed,'initial':before,'validation':after,'ledger':metrics['ledger']}
    write(out/'pilot/result.json',results)
    if not all(r['learnability_pass'] for r in results.values()):raise RuntimeError('PILOT_INCONCLUSIVE: held-out learnability gate failed; no main training or Test access, no budget extension')
    frozen={'format':'minicells.m5-sv.freeze.v1','frozen':expected,'pilot':results,'test_accessed':False,
            'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
    write(receipt,frozen);return frozen


def test_condition(out, protocol, identity, seeds, arm, capacity, model, condition, deadline):
    target=out/'metrics/test'/str(seeds['model_seed'])/arm/str(capacity)/(condition['name']+'.json')
    if target.exists():
        report=json.loads(target.read_text())
        sha=report.pop('result_sha256',None)
        if sha!=digest(report) or report['identity']!=identity or report['episode_count']!=protocol['test_episodes_per_condition']:raise ValueError('completed test report identity/hash/size differs')
        return
    raw=out/'raw-evaluation/test'/str(seeds['model_seed']);values=[];baseline=[]
    for row in read_bank(out/'datasets/test.jsonl'):
        if row['condition']!=condition['name']:continue
        base_path=raw/'E6'/'0'/(row['id']+'.json')
        if arm=='E6':
            result=observed_result(model,row,base_path,identity,deadline,recover_all=True,no_history=True);base=result
        else:
            saved=json.loads(base_path.read_text());base=saved['result']
            if saved['sha256']!=digest(base):raise ValueError('E6 paired result corrupted')
            result=observed_result(model,row,raw/arm/str(capacity)/(row['id']+'.json'),identity,deadline,oracle=arm=='E7-Oracle')
        values.append(result);baseline.append(base)
    assert len(values)==protocol['test_episodes_per_condition']
    report=condition_report(values,baseline);report.update(identity=identity,capacity=capacity,arm=arm,seeds=seeds,
        checkpoint_sha256=file_hash(out/'training'/str(seeds['model_seed'])/('E7-Innovation' if arm=='E7-Oracle' else arm)/'checkpoint.pt'))
    report['result_sha256']=digest(report);write(target,report)


def main_training_and_test(out, protocol, identity, deadline):
    adequacy={}
    for seeds in protocol['seeds']:
        for arm in protocol['arms']:
            metrics,curve=train_arm(out,protocol,arm,seeds,protocol['training']['input_tokens_per_arm_per_seed'],identity,deadline)
            previous,last=curve[-2]['validation'],curve[-1]['validation'];rules=protocol['decision']
            learned=metrics['initial_evaluation']['nll']-last['nll']>=rules['main_min_validation_nll_improvement'] and last['query_exact_match']*last['query_targets']>=rules['main_min_validation_correct']
            plateau=abs(previous['nll']-last['nll'])<=rules['validation_stability_nll_tolerance'] and abs(last['query_exact_match']-previous['query_exact_match'])<=rules['validation_stability_em_tolerance']
            adequacy[str(seeds['model_seed'])+':'+arm]={'stable':bool(plateau and learned),'learned':bool(learned),'plateau':bool(plateau),'last_quarter_previous':previous,'final':last}
    paired=[]
    for seeds in protocol['seeds']:
        records=[json.loads((out/'training'/str(seeds['model_seed'])/arm/'metrics.json').read_text()) for arm in protocol['arms']]
        inputs=[[h['input_sha256'] for h in r['history']] for r in records]
        init=[json.loads((out/'training'/str(seeds['model_seed'])/arm/'initialization.json').read_text())['static_parameter_sha256'] for arm in protocol['arms']]
        assert all(h==inputs[0] for h in inputs) and len(set(init))==1,'paired training/initialization mismatch'
        paired.append({'model_seed':seeds['model_seed'],'paired_input_order_sha256':digest(inputs[0]),'initial_parameter_sha256':init[0],'budget_per_arm':protocol['training']['input_tokens_per_arm_per_seed']})
    write(out/'paired-training-audit.json',paired)
    write(out/'adequacy.json',adequacy)
    # Re-run trained development diagnostics before unsealing Test, without tuning.
    for seeds in protocol['seeds']:
        development_diagnostics(out,protocol,identity,deadline,seeds,out/'training'/str(seeds['model_seed']))
    write(out/'test-access.json',{'identity':identity,'test_sha256':json.loads((out/'dataset-manifest.json').read_text())['test']['sha256'],
        'reason':'all twelve main checkpoints completed; no further optimization is permitted after Test access',
        'first_access_utc':datetime.now(timezone.utc).isoformat()}) if not (out/'test-access.json').exists() else None
    for seeds in protocol['seeds']:
        for arm in (*protocol['arms'],'E7-Oracle'):
            check_time(deadline,180);model=construct(protocol,arm,seeds).cuda()
            saved=load_checkpoint(out/'training'/str(seeds['model_seed'])/('E7-Innovation' if arm=='E7-Oracle' else arm)/'checkpoint.pt');model.load_state_dict(saved['model']);del saved
            try:
                for capacity in ([0] if arm=='E6' else protocol['capacity_tiers']):
                    if capacity:model.capacity=capacity
                    for condition in protocol['conditions']:test_condition(out,protocol,identity,seeds,arm,capacity,model,condition,deadline)
            finally:del model;gc.collect();torch.cuda.empty_cache()
    return scientific_decision(out/'metrics/test',protocol,adequacy)


def publish(out, decision):
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    with tempfile.TemporaryDirectory(prefix='m5-sv-publish-') as temp:
        root=Path(temp)/'repo';subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','--detach',decision['identity']['source_commit']],cwd=root,check=True)
        stage=root/'results/epclm004-m5-sv/publish';stage.mkdir(parents=True);names=[]
        for p in out.rglob('*'):
            rel=p.relative_to(out)
            if p.is_file() and p.suffix in ('.json','.jsonl','.md','.log') and not set(rel.parts)&{'datasets','raw-evaluation'} and p.name!='publication.json':
                dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);names.append(str(rel))
        hashes={n:file_hash(stage/n) for n in names};write(stage/'publication-files.json',hashes);names.append('publication-files.json')
        destination='artifacts/experiments/epclm004-m5-sv/'+decision['run_id']
        spec=ExperimentSpec(str(stage.relative_to(root)),destination,tuple(names),decision['branch'],'minicells.m5-sv.results.v1')
        folder=prepare_artifacts(root,'epclm004-m5-sv',spec,decision.get('script_version_id'))
        push_results(root,folder,'epclm004-m5-sv',decision['branch'],'GITHUB_TOKEN','origin/main')
        for n,h in hashes.items():assert file_hash(root/destination/n)==h
        proof={'status':'REMOTE_SHA_VERIFIED','branch':decision['branch'],'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'files':hashes}
    write(out/'publication.json',proof);return proof


def report(out, decision):
    write(out/'decision.json',decision)
    protocol=json.loads((BASE/'configs/protocol.json').read_text())
    lines=['# M5-SV actual execution report','', '## Executive summary','',
        'Status: '+decision['status'], 'H5_STATUS: '+decision['H5_STATUS'],
        '', 'M6 is not authorized. Engineering/protocol PASS does not establish scientific support.',
        'The frozen M5 baseline was a two-update engineering Smoke, not scientific validation.',
        '', '## Experimental setup','', 'Source: '+decision['identity']['source_commit'],
        'Config hash: '+decision['identity']['config_sha256'],
        'Tokenizer hash: '+decision['identity']['tokenizer_sha256'],
        'Hardware: '+json.dumps(decision.get('hardware',{})),
        'Seeds: '+json.dumps(protocol['seeds']),
        'Data: independent modulo-three binding partitions; Test is hashed before Pilot and sealed until all main checkpoints finish.',
        'Budgets: 1,000,000 inputs per arm/seed, 12M main total; separately bounded 65,536-input Pilot.',
        '', '## Actual training and resource evidence','',
        '| Run | Updates | Input tokens | Targets | Validation EM | Peak bytes | Compute tokens/s | Checkpoint bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ('pilot','training'):
        for path in sorted((out/kind).glob('*/*/metrics.json')):
            r=json.loads(path.read_text());ledger=r['ledger'];resources_path=path.parent/'resources.json'
            resources=json.loads(resources_path.read_text()) if resources_path.exists() else {}
            validation=r.get('validation') or {}
            values=[str(path.parent.relative_to(out)),ledger['committed_updates'],ledger['input_tokens'],
                ledger['supervised_target_tokens'],validation.get('query_exact_match'),
                resources.get('training_peak_allocated_bytes'),resources.get('training_compute_tokens_per_second'),r['checkpoint']['bytes']]
            lines.append('| '+' | '.join(str(v) for v in values)+' |')
    lines+=['','Training FLOPs are not claimed: no validated complete operator counter is installed.',
        'Static module counts, checkpoint write timings, optimizer/RNG/sampler identities and validation curves are retained beside each run.',
        '', '## SV1 and SV2','',
        'Raw per-record innovation scores and E6 recoverability labels, including undefined single-class AUROC/AUPRC, are retained in Output.',
        'Development summaries are in metrics/development/. Independent Test reports include admission precision/recall/coverage,',
        'Hit@1/Hit@K/Recall@K/MRR, per-layer candidates, paired On/Off EM, conditional EM, rescue and regression.',
        'Primary retrieval uses the last SWA layer, not an optimistic union. Oracle is diagnostic admission only.',
        '', '## SV3 independent evaluation','']
    final=out/'scientific-decision.json'
    if final.exists():
        scientific=json.loads(final.read_text());lines+=[scientific['outcome'],'',
            '| Capacity | Arm | Metric | Per-seed values | Mean | Sample SD | Seed t CI95 |',
            '|---|---|---|---|---:|---:|---|']
        for cap,arms in scientific['seed_tables'].items():
            for arm,metrics in arms.items():
                for name,r in metrics.items():
                    lines.append('| '+' | '.join(map(str,[cap,arm,name,r['per_seed'],r['mean'],r['sample_std'],r['seed_t_ci95']]))+' |')
        lines+=['', 'Paired policy/On-Off differences and95% crossed seed/Episode bootstrap intervals are in scientific-decision.json.',
            'All conditions, including failures, are displayed separately in condition_tables; primary long conditions were frozen before Test.',
            'Continuing validation improvement or unestimable recoverability evidence forces INCONCLUSIVE.']
    else:lines+=['Independent scientific evaluation is incomplete. No final accuracy or H5 support is inferred.']
    lines+=['','## Resource analysis and limitations','',
        'Each capacity report includes weights, SSM, local KV, derived Summary, predictor, payload and metadata bytes, plus actual query/prefill/read/update timing.',
        'Same-checkpoint policy interventions are iso-parameter. Resource ratios compare measured dynamic bytes and Query latency to E6; no extra training group is added.',
        'Query latency includes snapshot cloning and three-token computation. It is an instrumented diagnostic, not a production benchmark.',
        'Capacity128 is inference-only on capacity64-trained weights; cosine strata are not semantic similarity. E3 remains an original Smoke reference.',
        'Full query traces and Model/Optimizer/Runtime binaries stay in Kaggle Output; Git receives reports and hash manifests.',
        '', '## Scientific conclusion','',
        'Innovation validity, admission, retrieval, utilization, final accuracy and resource acceptability must all pass the frozen criteria before SUPPORTED.',
        'The actual H5_STATUS above is the only current verdict. No automatic M6 decision or budget expansion follows.',
        '', '```json',json.dumps(decision.get('error'),indent=2),'```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def launch(args):
    protocol=json.loads((BASE/'configs/protocol.json').read_text());identity=source_identity(protocol)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    started=args.started or time.monotonic();deadline=started+protocol['budget']['execution_seconds']
    run_id='epclm004-m5-sv-'+identity['source_commit'][:12]
    decision={'format':'minicells.m5-sv.results.v1','identity':identity,'run_id':run_id,'branch':'kaggle/'+run_id,
              'status':'RUNNING','H5_STATUS':'NOT_EVALUATED','m6_authorized':False,
              'script_version_id':os.environ.get('KAGGLE_SCRIPT_VERSION_ID') or None,'stages':{}}
    try:
        verify_baseline();resume_output(out,identity)
        previous=json.loads((out/'decision.json').read_text()) if (out/'decision.json').exists() else None
        if previous and previous['status']=='SCIENTIFIC_EVALUATION_COMPLETE':
            if file_hash(out/'scientific-decision.json')!=previous.get('scientific_result_sha256'):raise ValueError('completed scientific verdict hash differs')
            decision=previous
        else:
            decision['stages']['SV0']=functional_gate(out,identity)
            data=freeze_data(out,protocol);write(out/'protocol.json',protocol)
            if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('BLOCKED_HARDWARE: Tesla T4 required; no substitute precision/algorithm')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            decision['hardware']={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'fp32','tf32':False}
            decision['stages']['SV1_SV2']=development_diagnostics(out,protocol,identity,deadline,protocol['pilot'])
            pilot_and_freeze(out,protocol,identity,data,deadline)
            scientific=main_training_and_test(out,protocol,identity,deadline);write(out/'scientific-decision.json',scientific)
            decision.update(status=scientific['status'],H5_STATUS=scientific['H5_STATUS'],scientific_result_sha256=file_hash(out/'scientific-decision.json'))
    except Pause as error:
        decision.update(status='PARTIAL_RESUME_REQUIRED',H5_STATUS='INCONCLUSIVE',error=str(error))
    except Exception as error:
        decision.update(status='BLOCKED',H5_STATUS='INCONCLUSIVE',error=f'{type(error).__name__}: {error}')
    finally:
        decision['wall_seconds_this_session']=time.monotonic()-started;report(out,decision)
        write(out/'run-manifest.json',{'identity':identity,'run_id':run_id,'saved_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':decision['status'],'requires_attached_output':'BLOCKED_MISSING_OUTPUT' in str(decision.get('error','')),
            'resume':'attach exact saved Output; no source/config substitution'})
    if args.publish:
        proof=publish(out,decision);print({'publication':proof['status'],'branch':proof['branch'],'commit':proof['commit']},flush=True)
    print({'status':decision['status'],'H5_STATUS':decision['H5_STATUS'],'report':str(out/'REPORT.md')},flush=True)
    if decision['status']=='BLOCKED':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--started',type=float);p.add_argument('--publish',action='store_true')
    launch(p.parse_args())
