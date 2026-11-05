"""Bounded staged jobs: fresh language training, independent evaluations, existing result return."""
import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
from common import BASE,ROOT,INCREMENTAL,native,phase_c,digest,model_hash,sha,write,protocol,bounded
from architecture import build
from runtime import Session
from flops import report as flop_report
from evaluate import language,offsets,all_evaluations,sync
from mechanisms import run as mechanism_run
from decisions import choose_g1,choose_g2,choose_g3
from audit import run as repository_audit
from cell import SelectiveCellFFN

EXPERIMENT='epclm-x4c-001'
MODEL_FILES=('config.json','training-metrics.json','validation-results.json','holdout-results.json','generation-samples.json',
             'flops-report.json','inference-profile.json','memory-profile.json','state-diagnostics.json','decision.json')


def g0(device,plan):
    torch.manual_seed(713);original=phase_c.build_phase_c(phase_c.X4_NAME).to(device).eval()
    candidate=build('A0').to(device).eval();candidate.load_state_dict(original.state_dict(),strict=True)
    ids=torch.randint(0,2048,(1,128),device=device)
    with torch.no_grad():
        reference=original(ids);a=candidate(ids);cached=Session(candidate).prefill(ids)
        altered=ids.clone();altered[:,80:]=(altered[:,80:]+1)%2048;b=candidate(altered)
        checks={'original_parameter_count':sum(p.numel() for p in candidate.parameters())==1893912,
                'reference_equivalence':torch.allclose(reference,a,atol=plan['atol'],rtol=plan['rtol']),
                'deterministic':torch.equal(a,candidate(ids)),
                'causality':torch.allclose(a[:,:80],b[:,:80],atol=plan['atol'],rtol=plan['rtol']),
                'cache_equivalence':torch.allclose(a,cached,atol=plan['atol'],rtol=plan['rtol']),
                'flops':flop_report(candidate)['forward_flops']==phase_c.estimate_forward_flops(phase_c.X4_NAME,125)}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'max_reference_error':float((reference-a).abs().max()),'max_cache_error':float((a-cached).abs().max()),
            'weights':'fresh random initialization for correctness only, not a quality evaluation'}


def model_gate(model,plan):
    model.eval();device=next(model.parameters()).device;ids=torch.randint(0,2048,(1,70),device=device)
    with torch.no_grad():reference=model(ids)
    session=Session(model);actual=session.prefill(ids);split=Session(model)
    divided=torch.cat([split.prefill(chunk) for chunk in ids.split(8,1)],1)
    return bool(torch.allclose(reference,actual,atol=plan['atol'],rtol=plan['rtol']) and torch.equal(actual,divided))


def profile_record(profiler,path):
    profiler.export_chrome_trace(str(path))
    rows=[{'operator':event.key,'count':event.count,'reported_flops':event.flops,
           'cpu_time_microseconds':event.cpu_time_total,'device_time_microseconds':getattr(event,'device_time_total',0),
           'self_cpu_memory_bytes':event.self_cpu_memory_usage,'self_device_memory_bytes':getattr(event,'self_device_memory_usage',0)} for event in profiler.key_averages()]
    return {'status':'MEASURED_TRAINING_UPDATE','operators':rows,'raw_trace':str(path),
            'coverage':'PyTorch profiler known-operator FLOPs only; omissions are not zero-cost claims',
            'cuda_kernel_events':sum('CUDA' in str(event.device_type) for event in profiler.events())}


def train_language(model,corpus,starts,job,device,folder,deadline,metadata):
    plan=job['training'];profile=plan['profiles'][job['profile']];target=profile['tokens']
    if (plan['sequence_length'],plan['batch'],plan['lr'],tuple(plan['betas']),plan['weight_decay'],plan['clip'])!=(125,8,.0003,(.9,.95),.1,1.):raise ValueError('Native training configuration changed; explicit protocol amendment required')
    flops=flop_report(model);per_token=flops['forward_flops_per_prediction_token']*3
    if job['match']=='compute':
        baseline_cost=flop_report(build('A0'))['forward_flops_per_prediction_token']*3
        target=int((profile['tokens']*baseline_cost/per_token)//1000)*1000
    total_steps=target//1000;warmup=max(1,round(profile['warmup']*target/profile['tokens']))
    metadata.update(target_prediction_tokens=target,total_updates=total_steps,warmup_updates=warmup,match_axis=job['match'])
    train=native.open_memmap(corpus.train_path);validation=native.open_memmap(corpus.validation_path)
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],betas=tuple(plan['betas']),weight_decay=plan['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=plan['initial_scale'])
    rng=random.Random(job['seed']);high=len(train)-126;tokens=updates=attempted_tokens=0;metrics=[];overflows=[];wall=time.monotonic();measured=None
    milestones={min(target,n) for n in (100000,250000,500000,750000,1000000,2500000,5000000,7500000,target)}
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(1,total_steps+1):
            if job['match']=='compute' and attempted_tokens>=target:break
            if time.monotonic()>=deadline:raise TimeoutError('language training execution budget exhausted')
            batch=tuple(rng.randrange(high) for _ in range(8));inputs,targets=native.memmap_batch(train,batch,125,device)
            lr=plan['lr']*native.lr_multiplier(step,total_steps,warmup)
            for group in optimizer.param_groups:group['lr']=lr
            committed=False
            for attempt in range(plan['maximum_consecutive_overflows']):
                if job['match']=='compute' and attempted_tokens>=target:break
                attempted_tokens+=targets.numel()
                model.train();optimizer.zero_grad(set_to_none=True);sync(device);began=time.monotonic();profiler=None
                if updates==0:
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
                    try:
                        profiler=torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_flops=True);profiler.__enter__()
                    except Exception as error:
                        measured={'status':'FAILED','error':type(error).__name__+': '+str(error)};profiler=None
                try:
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        logits=model(inputs);loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
                    if not bool(loss.isfinite()):raise FloatingPointError('nonfinite language loss')
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    gradients=[p.grad for p in model.parameters() if p.grad is not None]
                    finite=bool(torch.stack([g.isfinite().all() for g in gradients]).all())
                    if finite:
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip'],error_if_nonfinite=True)
                        scaler.step(optimizer);scaler.update()
                    else:
                        scale=scaler.get_scale()
                        if not scaler.is_enabled():raise FloatingPointError('nonfinite unscaled gradients; optimizer update skipped')
                        scaler.step(optimizer);scaler.update()
                        overflows.append({'attempted_update':step,'attempt':attempt+1,'committed_updates':updates,'committed_tokens':tokens,
                                          'loss':float(loss.detach()),'scale':scale,'next_scale':scaler.get_scale(),'input_offsets':list(batch)})
                finally:
                    if profiler is not None:
                        try:profiler.__exit__(None,None,None)
                        except Exception as error:
                            measured={'status':'FAILED','error':type(error).__name__+': '+str(error)};profiler=None
                if finite:committed=True;break
                if attempt+1>=plan['maximum_consecutive_overflows']:raise FloatingPointError('consecutive FP16 overflows exceeded; same input retried, no tokens committed')
            if not committed:
                if job['match']=='compute' and attempted_tokens>=target:break
                raise FloatingPointError('no committed optimizer update')
            if not bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all()):raise FloatingPointError('nonfinite optimizer parameters')
            tokens+=targets.numel();updates+=1;sync(device);elapsed=time.monotonic()-began
            row={'update':updates,'prediction_tokens':tokens,'update_prediction_tokens':targets.numel(),'loss':float(loss.detach()),
                 'learning_rate':lr,'gradient_norm':float(norm),'seconds':elapsed,'tokens_per_second':targets.numel()/max(elapsed,1e-12),
                 'cumulative_training_flops_estimate':per_token*attempted_tokens,'committed_update_flops_estimate':per_token*tokens,'input_offsets':list(batch),'grad_scaler_scale':scaler.get_scale()}
            if isinstance(model.stages[1].ff,SelectiveCellFFN):
                cell=model.stages[1].ff
                row['cell_specific_gradient_norm']=None if cell.down.grad is None else cell.down.grad.detach().flatten(1).norm(dim=1).cpu().tolist()
            metrics.append(row)
            if profiler is not None and measured is None:
                trace=folder/'private/first-update-trace.json';trace.parent.mkdir(exist_ok=True)
                try:measured=profile_record(profiler,trace)
                except Exception as error:measured={'status':'FAILED','error':type(error).__name__+': '+str(error),'raw_trace':str(trace)}
            if tokens in milestones:
                try:row['validation']=language(model,validation,starts,device,deadline)
                except Exception as error:row['validation']={'status':'FAILED','error':type(error).__name__+': '+str(error)}
                write(folder/'training-metrics.json',{'status':'IN_PROGRESS','metrics':metrics,'overflows':overflows})
                print(json.dumps({'model':metadata['model'],'tokens':tokens,'validation_ppl':row['validation'].get('ppl')}),flush=True)
        if job['match']=='tokens' and (tokens!=target or updates!=total_steps):raise ValueError('language token/update budget mismatch')
        if job['match']=='compute' and (attempted_tokens!=target or updates<1):raise ValueError('analytic compute attempt budget not met')
        metadata['training_status']='COMPLETE'
    finally:
        if job['match']=='tokens':metadata['training_status']='COMPLETE' if tokens==target and updates==total_steps else 'INCOMPLETE'
        elif metadata['training_status']!='COMPLETE':metadata['training_status']='INCOMPLETE'
        optimizer_bytes=sum(v.numel()*v.element_size() for entry in optimizer.state.values() for v in entry.values() if torch.is_tensor(v))
        if isinstance(model.stages[1].ff,SelectiveCellFFN):
            grad=model.stages[1].ff.down.grad
            metadata['cell_specific_gradient_norm_last_training_update']=None if grad is None else grad.detach().flatten(1).norm(dim=1).cpu().tolist()
        metadata.update(training_tokens=tokens,optimizer_updates=updates,training_wall_seconds=time.monotonic()-wall,
                        optimizer_state_bytes=optimizer_bytes,training_flops_estimate=per_token*attempted_tokens,attempted_prediction_tokens=attempted_tokens,committed_update_flops_estimate=per_token*tokens,
                        peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        write(folder/'training-metrics.json',{'status':metadata['training_status'],'metrics':metrics,'overflows':overflows,
             'optimizer_state_bytes':optimizer_bytes,'training_wall_seconds':metadata['training_wall_seconds'],
             'tokens_per_second':tokens/max(metadata['training_wall_seconds'],1e-12),
             'accepted_update_compute_seconds':sum(row['seconds'] for row in metrics),
             'tokens_per_accepted_update_compute_second':tokens/max(sum(row['seconds'] for row in metrics),1e-12),
             'timing_scope':'wall includes rejected attempts, validation, report writes and first-update profiling; update seconds cover accepted attempts only and exclude validation/report serialization',
             'attempted_prediction_tokens':attempted_tokens,'analytic_training_flops_including_rejected_attempts':per_token*attempted_tokens,
             'compute_scope':'forward/backward attempts counted; optimizer excluded by the declared analytic convention; schedule advances only on committed updates',
             'tbptt_chunks':4,'detach':'Session reset between independent 125-token samples; no intra-sample detach'})
        write(folder/'flops-report.json',{**flops,'training_tokens':tokens,'total_training_flops_estimate':per_token*attempted_tokens,'committed_update_flops_estimate':per_token*tokens,'attempted_prediction_tokens':attempted_tokens,'runtime_profile':measured})
    return metrics


def save_completed_weights(model,folder,out,identity,meta):
    path=folder/'private/final-model.pt'
    native._atomic_save({'configuration':model.configuration,'model':model.state_dict(),'identity':identity},path)
    meta['model_weight_identity']=model_hash(model)
    meta['weights']={'file':str(path.relative_to(out)),'location':'Kaggle Output only; owner access required if private; not committed to Git'}


def publish(out,branch):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    spec=EXPERIMENTS[EXPERIMENT];destination=ROOT/spec.source_dir
    if destination.exists():shutil.rmtree(destination)
    destination.mkdir(parents=True)
    names=[]
    for path in out.rglob('*'):
        if not path.is_file() or 'private' in path.relative_to(out).parts or path.suffix not in ('.json','.md'):continue
        name=path.relative_to(out).as_posix();target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);names.append(name)
    spec=replace(spec,files=tuple(sorted(names)))
    artifact=prepare_artifacts(ROOT,EXPERIMENT,spec,None);push_results(ROOT,artifact,EXPERIMENT,branch,'GITHUB_TOKEN','origin/main')


def run(args):
    p=protocol();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run_id='epclm-x4c-001-'+source[:12]+'-'+uuid.uuid4().hex[:8];out=args.output/run_id;out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+min(args.execution_seconds,p['budget']['execution_seconds']);device=torch.device('cuda:0')
    identity={'source_commit':source,'protocol_sha256':digest(BASE/'protocol.json'),'seed':p['seed'],'profile':args.profile,'match':args.match,
              'independent_test_accessed':False,'run_id':run_id,'implementation_sha256':sha({p.name:digest(p) for p in sorted(BASE.glob('*.py'))})}
    decision={'format':'minicells.epclm-x4c-001.v1',**identity,'status':'INCOMPLETE','training_status':'NOT_STARTED','evaluation_status':'NOT_STARTED',
              'publication_status':'PENDING','scientific_confirmation':False,'phases':{phase:('NOT_STARTED' if args.stage in ('auto',phase) else 'NOT_REQUESTED') for phase in ('G1','G2','G3')}}
    models={};selection={**identity,'G1':None,'G2':None,'G3':None};write(out/'protocol.json',p)
    def persist():write(out/'decision.json',decision);write(out/'stage-results.json',models);write(out/'selection.json',selection)
    try:
        repository_audit(out)
        if not torch.cuda.is_available() or torch.cuda.get_device_name()!=p['resources']['target_gpu']:
            decision['status']=decision['training_status']='NOT_RUN_RESOURCE_UNAVAILABLE';return decision
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from scripts.research.publish_experiment_results import validate_origin,load_github_token
        validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
        test=subprocess.run([sys.executable,'-m','pytest','-q',str(BASE/'tests')],cwd=ROOT,env=env,capture_output=True,text=True,timeout=min(120,max(1,deadline-time.monotonic())))
        preflight={'cpu_tests_returncode':test.returncode,'cpu_tests':test.stdout[-12000:],'stderr':test.stderr[-2000:]}
        if test.returncode:write(out/'g0-report.json',preflight);raise RuntimeError('required correctness tests failed; no performance comparison permitted')
        with bounded(max(1,deadline-time.monotonic())):preflight['target_gpu']=g0(device,p['evaluation'])
        write(out/'g0-report.json',preflight)
        if preflight['target_gpu']['status']!='PASS':raise RuntimeError('target GPU G0 correctness failed')
        from epclm001.retrain_data import prepare
        historical=json.loads((INCREMENTAL.parent/'retrain_config.json').read_text())
        with bounded(max(1,deadline-time.monotonic())):corpus=prepare(args.cache,historical)
        tokenizer=native.load_tokenizer(corpus.tokenizer_path);starts=offsets(len(native.open_memmap(corpus.validation_path)),p['data']['validation_seed'],p['data']['holdout_seed'])
        identity.update(dataset_manifest_sha256=sha(corpus.manifest),tokenizer_sha256=digest(corpus.tokenizer_path),
                        train_sha256=digest(corpus.train_path),validation_sha256=digest(corpus.validation_path),
                        environment={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'training_precision':p['training']['precision']})
        decision.update(identity);selection.update(identity)
        write(out/'manifest.json',{**identity,'data':corpus.manifest,'evaluation_offsets':starts,'historical_weights':'HISTORICAL_WEIGHTS_UNAVAILABLE; local weights absent; fresh models only'})
        if args.selection:
            saved=json.loads(args.selection.read_text())
            for key in ('source_commit','protocol_sha256','dataset_manifest_sha256','tokenizer_sha256','implementation_sha256'):
                if saved.get(key)!=identity[key]:raise ValueError('selection identity differs: '+key)
            selection.update({key:saved.get(key) for key in ('G1','G2','G3')})
        phases=['G1','G2','G3'] if args.stage=='auto' else [args.stage]
        for phase in phases:
            base_iterations=None
            if phase in ('G2','G3'):
                g1=selection.get('G1') or {}
                if g1.get('status')!='G1_BASE_SELECTED' or not g1.get('stage_complete'):
                    decision['phases'][phase]='NOT_RUN_G1_SELECTION_REQUIRED';break
                base_iterations=g1['base_iterations']
            if phase=='G3' and not (selection.get('G2') or {}).get('g3_candidate_allowed'):
                decision['phases'][phase]='NOT_RUN_G2_MECHANISM_EVIDENCE_REQUIRED';break
            results={};decision['phases'][phase]='IN_PROGRESS'
            for name in p['stages'][phase]:
                key=phase+'/'+name;folder=out/'models'/phase/name;folder.mkdir(parents=True,exist_ok=True)
                meta={**identity,'phase':phase,'model':name,'training_status':'NOT_STARTED','evaluation_status':'NOT_STARTED','model_weight_identity':None}
                results[name]=meta;models[key]=meta;model=None
                try:
                    if time.monotonic()>=deadline:raise TimeoutError('job execution budget exhausted')
                    torch.manual_seed(p['seed']);model=build(name,base_iterations).to(device=device,dtype=torch.float32)
                    meta['configuration']=copy.deepcopy(model.configuration);meta['parameters']=sum(v.numel() for v in model.parameters())
                    write(folder/'config.json',meta)
                    if not model_gate(model,p['evaluation']):raise RuntimeError('model causal/cache/Chunk consistency gate failed')
                    meta['training_status']='IN_PROGRESS';job={**p,'profile':args.profile,'match':args.match}
                    with bounded(max(1,deadline-time.monotonic())):train_language(model,corpus,starts[0],job,device,folder,deadline,meta)
                    model.zero_grad(set_to_none=True)
                    if not model_gate(model,p['evaluation']):raise RuntimeError('trained model cache/Chunk equivalence failed; quality comparisons blocked')
                    meta['trained_correctness']='PASS'
                    save_completed_weights(model,folder,out,identity,meta)
                    with bounded(max(1,deadline-time.monotonic())):
                        measured=all_evaluations(model.eval(),corpus,tokenizer,starts,p['evaluation'],device,folder,deadline)
                    if 'routing' in measured.get('state-diagnostics.json',{}):
                        measured['state-diagnostics.json']['routing']['cell_specific_gradient_norm_last_training_update']=meta.get('cell_specific_gradient_norm_last_training_update')
                        write(folder/'state-diagnostics.json',measured['state-diagnostics.json'])
                    meta.update(measured);meta['flops-report.json']=json.loads((folder/'flops-report.json').read_text())
                    allowed=('COMPLETE','NOT_APPLICABLE','MEASURED')
                    meta['evaluation_status']='COMPLETE' if all(value['status'] in allowed for value in measured.values()) else 'INCOMPLETE'
                    if phase=='G3' and meta['flops-report.json'].get('runtime_profile',{}).get('status')!='MEASURED_TRAINING_UPDATE':meta['evaluation_status']='INCOMPLETE'
                    if phase in ('G2','G3'):
                        try:
                            with bounded(max(1,deadline-time.monotonic())):meta['mechanisms']=mechanism_run(model,p['mechanisms'],device,deadline,folder/'private')
                        except Exception as error:meta['mechanisms']={'status':'FAILED','error':str(error)}
                        diagnostics=meta.get('state-diagnostics.json',{});diagnostics['mechanisms']=meta['mechanisms'];write(folder/'state-diagnostics.json',diagnostics)
                        if meta['mechanisms'].get('status')!='COMPLETE' or meta['mechanisms'].get('M5',{}).get('status')=='FAIL':meta['evaluation_status']='INCOMPLETE'
                except Exception as error:
                    meta['error']=type(error).__name__+': '+str(error)
                    if meta['training_status']=='IN_PROGRESS':meta['training_status']='INCOMPLETE'
                finally:
                    if model is not None and meta['training_status']=='COMPLETE' and meta.get('model_weight_identity') is None:
                        try:save_completed_weights(model,folder,out,identity,meta)
                        except Exception as error:meta['weight_save_error']=type(error).__name__+': '+str(error)
                    write(folder/'decision.json',meta)
                    for filename in MODEL_FILES:
                        if not (folder/filename).exists():write(folder/filename,{'status':'NOT_RUN','reason':meta.get('error','stage not reached'),**identity})
                    if model is not None:del model
                    torch.cuda.empty_cache();persist()
            complete=all(r['training_status']=='COMPLETE' and r['evaluation_status']=='COMPLETE' for r in results.values())
            decision['phases'][phase]='COMPLETE' if complete else 'INCOMPLETE'
            if phase=='G1':selection['G1']={**choose_g1(results,args.profile,args.match,p['g1_selection']),'stage_complete':complete}
            elif phase=='G2':selection['G2']=choose_g2(results,p['mechanisms'])
            else:selection['G3']=choose_g3(results)
            persist()
        decision['training_status']='COMPLETE' if models and all(v['training_status']=='COMPLETE' for v in models.values()) else 'INCOMPLETE'
        decision['evaluation_status']='COMPLETE' if models and all(v['evaluation_status']=='COMPLETE' for v in models.values()) else 'INCOMPLETE'
        decision['status']='REQUESTED_PHASES_COMPLETE' if decision['training_status']==decision['evaluation_status']=='COMPLETE' else 'INCOMPLETE'
        if decision['status']=='REQUESTED_PHASES_COMPLETE' and any(value.startswith('NOT_RUN') or value=='NOT_STARTED' for value in decision['phases'].values()):decision['status']='STOPPED_AT_STAGE_DECISION'
        decision['scope']='Completed phases only; blocked later phases and unperformed confirmation remain unproven'
    except Exception as error:decision['error']=type(error).__name__+': '+str(error)
    finally:
        persist()
        for name in ('manifest.json','g0-report.json'):
            if not (out/name).exists():write(out/name,{'status':'NOT_RUN','reason':decision.get('error',decision['status']),**identity})
        from research_report import render
        (out/'RESEARCH-REPORT.zh-CN.md').write_text(render(decision,selection,models))
        branch='kaggle/'+run_id
        try:
            with bounded(p['budget']['publication_seconds']):publish(out,branch)
            decision['publication_status']='REMOTE_SHA_VERIFIED';decision['result_branch']=branch;persist()
        except Exception as error:
            decision['publication_status']='FAILED';decision['publication_error']=type(error).__name__+': '+str(error);persist()
        print(json.dumps({**decision,'output':str(out)},ensure_ascii=False),flush=True)
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('auto','G1','G2','G3'),default='auto')
    parser.add_argument('--profile',choices=('smoke','dev','development'),default='dev');parser.add_argument('--match',choices=('tokens','compute'),default='tokens')
    parser.add_argument('--selection',type=Path);parser.add_argument('--output',type=Path,default=Path('/kaggle/working/epclm-x4c-001'))
    parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/x4c-native-corpus'));parser.add_argument('--execution-seconds',type=float,default=5400)
    run(parser.parse_args())
