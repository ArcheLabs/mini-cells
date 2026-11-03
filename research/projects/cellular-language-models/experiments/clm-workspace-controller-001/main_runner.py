"""Explicit bounded modes and three-seed ledger; never auto-start formal Main."""
import argparse,copy,json,time,subprocess,uuid,math
from pathlib import Path
import torch
from dependencies import BASE,ROOT,shared,assets
from models import Model
from costs import match
from training import train,identity,targets
from evaluate import evaluate
from recovery import restore,commit_bundle
from run import publish
from precision import select


def plan(documents,p,budget):
    # Deterministic repeated complete-document groups; never truncate a document.
    # A fixed subset-sum tail attains exact budgets without partial documents.
    groups=[];total=0;cursor=0;size=p['groups']['documents']
    while total<budget:
        group=[documents[(cursor+i)%len(documents)] for i in range(size)];n=targets(group)
        if total+n>budget:break
        groups.append(group);total+=n;cursor+=size
    if total<budget:
        # Deterministic exact-budget tail using complete documents only. Skips and
        # chosen doc IDs are frozen by this rule, identical in every arm/seed.
        remaining=budget-total;reachable=1;parents={};limit=(1<<(remaining+1))-1
        candidates=[documents[(cursor+i)%len(documents)] for i in range(max(len(documents),2*size))]
        for index,doc in enumerate(candidates):
            n=len(doc['ids'])-1;new=((reachable<<n)&limit)&~reachable;reachable|=new
            while new:
                bit=new&-new;value=bit.bit_length()-1;parents[value]=(value-n,index);new-=bit
            if reachable&(1<<remaining):break
        if reachable&(1<<remaining):
            chosen=[];value=remaining
            while value:value,index=parents[value];chosen.append(index)
            tail=[candidates[i] for i in sorted(chosen)]
            groups.extend(tail[i:i+size] for i in range(0,len(tail),size));total=budget
    return groups,total


def validate_freeze(freeze,source,p,manifest):
    if freeze.get('status')!='FROZEN' or freeze.get('gate_decision')!='MAIN_READY':raise ValueError('Main requires separately frozen budget and MAIN_READY')
    if freeze['source_commit']!=source or freeze['protocol_sha256']!=shared.sha(p) or freeze['dataset_manifest_sha256']!=shared.sha(manifest):raise ValueError('frozen Main identity mismatch')
    if freeze['precision_policy']!=select(p['precision'].get('requested','auto'),'cuda'):raise ValueError('formal precision/hardware policy mismatch')
    if freeze['seeds']!=[17,29,43]:raise ValueError('formal seed cohort must be 17/29/43')
    if freeze['targets_per_seed']!=32000000 and not freeze.get('separate_budget_decision'):raise ValueError('cannot silently reduce planned 32M budget')


def launch(args):
    if args.mode=='pilot':
        from pilot import launch as pilot
        pilot(argparse.Namespace(started=args.started,output=args.output,cache=args.cache,resume_run=args.run_dir,no_recovery=False));return
    started=args.started if args.started is not None else time.monotonic();deadline=started+min(args.session_seconds,10800)-600
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();p=json.loads((BASE/'protocol.json').read_text())
    if args.mode=='publish-only':publish(args.run_dir);return
    if args.mode=='smoke':
        from run import launch as smoke
        smoke(argparse.Namespace(started=started,output=args.output,cache=args.cache,resume_run=args.run_dir,no_recovery=False));return
    if args.mode=='main' and not args.freeze:raise ValueError('explicit --freeze required; default never starts Main')
    if not torch.cuda.is_available():raise RuntimeError('real CUDA runtime required; local synthetic tests do not close GPU gate')
    if args.mode=='main' and (not args.freeze.exists()):raise ValueError('missing formal budget freeze')
    docs,manifest=assets.prepare(args.cache,p);configs=match(p['model']);device=torch.device('cuda')
    budget=args.dry_run_targets if args.mode=='main-dry-run' else 0
    frozen=json.loads(args.freeze.read_text()) if args.freeze else None
    if args.mode=='main':validate_freeze(frozen,source,p,manifest);budget=frozen['targets_per_seed']
    elif args.mode=='evaluate-only':budget=0
    elif args.dry_run_targets>65536:raise ValueError('dry run capped at 65536 targets per arm/seed')
    groups,actual=plan(docs['train'],p,budget) if budget else ([],0)
    if budget and actual!=budget and args.mode=='main':raise ValueError('freeze must use exact full-group budget; no automatic rounding of Main')
    p['schedule_target_budget']=actual;p['execution_mode']=args.mode;p['main']['frozen']=args.mode=='main'
    # Session stop is operational and excluded from checkpoint identity/scheduler.
    immutable={'source':source,'mode':args.mode,'configs':configs,'protocol':p,'manifest':manifest,'seeds':[17,29,43],'actual_targets_per_arm_seed':actual,'plan_sha256':shared.sha([[d['doc_id'] for d in g] for g in groups])}
    run_identity=shared.sha(immutable)
    out=args.run_dir or args.output/('clm-v04-'+args.mode+'-'+source[:12]);out.mkdir(parents=True,exist_ok=True)
    if not (out/'manifest.json').exists() and (args.require_recovery or args.mode=='main'):
        proof=restore(out,run_identity,min(deadline,time.monotonic()+45))
        if proof is None and args.require_recovery:raise RuntimeError('RECOVERY_BLOCKED: required previous session not found')
    if args.mode=='evaluate-only':
        if not (out/'manifest.json').exists():raise ValueError('evaluate-only requires original run manifest')
        original=json.loads((out/'manifest.json').read_text())
        if original['source']!=source or original['manifest']!=manifest:raise ValueError('evaluation source/data mismatch')
        p=original['protocol'];configs=original['configs'];run_identity=original['identity']
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())['identity']!=run_identity:raise ValueError('run identity mismatch; no restart')
    else:shared.write(out/'manifest.json',{'identity':run_identity,**immutable})
    report={};decision={'format':'minicells.clm-workspace-controller-001.v04','run_id':out.name+'-'+uuid.uuid4().hex[:8],'mode':args.mode,'status':'INCOMPLETE','source_commit':source,'main_training_started':args.mode=='main','main_results':'NOT_RUN' if args.mode!='main' else 'INCOMPLETE','scientific_confirmation':False}
    try:
        for seed in (17,29,43):
            for arm in args.arms:
                if time.monotonic()>=deadline:break
                folder=out/str(seed)/arm;folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/'checkpoint.pt'
                model=Model(arm,configs[arm]['config'],seed).to(device);q=copy.deepcopy(p);q['smoke']['seed']=seed
                committed_id=identity(q,arm,source,manifest)
                if args.mode=='evaluate-only':
                    if not checkpoint.exists():raise RuntimeError('evaluate-only checkpoint missing; training prohibited')
                    saved=torch.load(checkpoint,map_location=device,weights_only=False)
                    # Evaluation validates original manifest, source, model config and precision separately.
                    if saved['identity']!=committed_id:raise ValueError('evaluation checkpoint identity mismatch')
                    model.load_state_dict(saved['model']);del saved
                    entry={'training':'NOT_RUN_EVALUATION_ONLY'}
                else:
                    if args.stop_after_groups:
                        q['session_stop_targets']=sum(targets(g) for g in groups[:args.stop_after_groups])
                    with shared.bounded(max(.1,deadline-time.monotonic())):
                        progress,performance=train(model,groups,q,deadline,checkpoint,committed_id,checkpoint_hook=lambda path:commit_bundle(out,run_identity,path))
                    entry={'training':progress,'performance':performance,'checkpoint_sha256':shared.digest(checkpoint) if checkpoint.exists() else None}
                    if progress['status']!='COMPLETE':report[str(seed)+'/'+arm]=entry;break
                with shared.bounded(max(.1,deadline-time.monotonic())):entry['validation']=evaluate(model,docs['validation'][:(p['evaluation']['documents'] if args.mode in ('main','evaluate-only') else 8)],q,deadline)
                report[str(seed)+'/'+arm]=entry;shared.write(folder/'result.json',entry);del model;torch.cuda.empty_cache()
            if args.stop_after_groups:break
        expected=len(args.arms)*3
        completed=len(report)==expected and all(e.get('validation',{}).get('status')=='COMPLETE' for e in report.values())
        decision['status']=('DRY_RUN_COMPLETE' if args.mode=='main-dry-run' else 'MAIN_COMPLETE' if args.mode=='main' else 'EVALUATION_COMPLETE') if completed else 'PARTIAL'
        if args.mode=='main' and completed:decision['main_results']='COMPLETE'
        aggregate={}
        for arm in args.arms:
            values=[e['validation']['mean_nll'] for key,e in report.items() if key.endswith('/'+arm) and e.get('validation',{}).get('status')=='COMPLETE']
            if values:
                import statistics
                aggregate[arm]={'seed_count':len(values),'mean_nll':statistics.mean(values),'std_nll':statistics.stdev(values) if len(values)>1 else None,'all_seeds_complete':len(values)==3}
        shared.write(out/'seed-summary.json',aggregate)
    except Exception as error:decision['error']=type(error).__name__+': '+str(error)
    finally:
        shared.write(out/'protocol.json',p);shared.write(out/'training-metrics.json',report);shared.write(out/'decision.json',decision)
        (out/'RESEARCH-REPORT.zh-CN.md').write_text('# CLM v0.4 '+args.mode+'\n\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n\nDry Run不纳入Main统计；三Seed训练结果分目录记录，未完成不填PASS。\n')
        try:
            with shared.bounded(max(.1,started+10800-time.monotonic())):publish(out,'clm-workspace-controller-001-main-dry-run' if args.mode=='main-dry-run' else 'clm-workspace-controller-001-main')
        except Exception as error:shared.write(out/'publication-failure.json',{'error':str(error)})
        print(decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','main-dry-run','main','evaluate-only','publish-only','pilot'],default='main-dry-run');parser.add_argument('--started',type=float);parser.add_argument('--output',type=Path,default=Path('/kaggle/working/clm-v04-main'));parser.add_argument('--run-dir',type=Path);parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/stateful-routing-data'));parser.add_argument('--freeze',type=Path);parser.add_argument('--dry-run-targets',type=int,default=4096);parser.add_argument('--stop-after-groups',type=int,default=0);parser.add_argument('--require-recovery',action='store_true');parser.add_argument('--session-seconds',type=int,default=10800);parser.add_argument('--arms',nargs='+',default=['C0','C1','C2','C3','T0-P','T0-F'])
    launch(parser.parse_args())
