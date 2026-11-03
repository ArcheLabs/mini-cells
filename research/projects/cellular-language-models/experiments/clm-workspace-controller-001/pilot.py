"""Two isolated T4 subprocesses, real concurrent benchmark, exact 12M Pilot cap."""
import argparse,copy,csv,json,os,subprocess,sys,time,uuid,gc
from pathlib import Path
import torch
from dependencies import BASE,ROOT,shared,assets
from costs import match
from matching_gate import freeze
from precision import select
from gpu_gate import correctness,benchmark,controller_equivalence
from pilot_queue import initialize,rows,predict
from recovery import restore,commit_bundle
from run import publish

ARMS=('T0-P','T0-F','C0','C1','C2','C3')
NAME='clm-workspace-controller-001-pilot-2m-seed17'


def inventory():
    if torch.cuda.device_count()!=2:raise RuntimeError('DUAL_GPU_BLOCKED: exactly two GPUs required; no single-GPU fallback')
    result=[]
    for i in range(2):
        if torch.cuda.get_device_name(i)!='Tesla T4':raise RuntimeError('DUAL_GPU_BLOCKED: both devices must be Tesla T4')
        props=torch.cuda.get_device_properties(i)
        result.append({'logical_index':i,'name':props.name,'total_memory':props.total_memory,'uuid':str(getattr(props,'uuid','NOT_EXPOSED')),'capability':list(torch.cuda.get_device_capability(i))})
    return result


def telemetry(path,started,tasks):
    fields='index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw'
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
        active={r['worker']:r['arm'] for r in tasks if r['status']=='RUNNING'}
        masks=[v.strip().replace('GPU-','').lower() for v in os.getenv('CUDA_VISIBLE_DEVICES','0,1').split(',')]
        bindings={mask:active.get('worker-'+str(i),'IDLE') for i,mask in enumerate(masks)}
        new=not path.exists()
        with path.open('a',newline='') as stream:
            writer=csv.writer(stream)
            if new:writer.writerow(['monotonic_seconds','elapsed_seconds']+fields.split(',')+['active_model'])
            for row in csv.reader(result.stdout.splitlines()):
                if row:writer.writerow([time.monotonic(),time.monotonic()-started]+[v.strip() for v in row]+[bindings.get(row[0].strip(),bindings.get(row[1].strip().replace('GPU-','').lower(),'IDLE'))])
    except Exception as error:
        with path.with_suffix('.errors').open('a') as stream:stream.write(str(error)+'\n')


def run_workers(out,spec_path,phase,deadline,started,priorities,run_identity):
    phase_started=time.monotonic()
    queue=out/(phase+'-queue.sqlite');initialize(queue,[(a,priorities.get(a,1)) for a in ARMS])
    # An explicit resume may continue unfinished tasks; completed tasks remain complete.
    from pilot_queue import connect
    with connect(queue) as db:
        if phase=='benchmark':db.execute("UPDATE tasks SET status='PENDING',result=NULL WHERE 1=1")
        else:db.execute("UPDATE tasks SET status='PENDING' WHERE status IN ('RUNNING','INCOMPLETE')")
    masks=[v.strip() for v in os.getenv('CUDA_VISIBLE_DEVICES','0,1').split(',')]
    if len(masks)!=2 or masks[0]==masks[1]:raise RuntimeError('ambiguous physical GPU binding')
    processes=[];logs=[];folder=out/phase;folder.mkdir(exist_ok=True)
    for gpu in range(2):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':masks[gpu],'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        log=(out/(phase+'-worker-'+str(gpu)+'.log')).open('a');logs.append(log)
        command=[sys.executable,str(BASE/'pilot_worker.py'),'--spec',str(spec_path),'--queue',str(queue),'--worker','worker-'+str(gpu),'--phase',phase,'--deadline',str(deadline),'--output',str(folder)]
        processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=False))
    seen={};last_sample=0
    try:
        while any(process.poll() is None for process in processes):
            now=time.monotonic()
            if now>=deadline+10:
                for process in processes:
                    if process.poll() is None:process.terminate()
                break
            if now-last_sample>=5:telemetry(out/'dual-gpu-telemetry.csv',started,rows(queue));last_sample=now
            # Parent is the only writer of the shared recovery index.
            if phase=='train':
                for event in folder.glob('*/checkpoint-event.json'):
                    content=event.read_text()
                    if seen.get(str(event))==content:continue
                    value=json.loads(content);checkpoint=Path(value['checkpoint'])
                    commit_bundle(out,run_identity,checkpoint);seen[str(event)]=content
            time.sleep(.5)
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for log in logs:log.close()
    result=rows(queue)
    shared.write(out/(phase+'-timing.json'),{'started':phase_started,'ended':time.monotonic(),'seconds':time.monotonic()-phase_started,'from_notebook_start':phase_started-started,'worker_returncodes':[process.returncode for process in processes]})
    if phase=='train':
        for checkpoint in folder.glob('*/checkpoint.pt'):commit_bundle(out,run_identity,checkpoint)
    return result


def utilization(path,started,tasks):
    intervals={worker:[(r['started'],r['ended']) for r in tasks if r['worker']==worker and r['started'] is not None and r['ended'] is not None] for worker in ('worker-0','worker-1')}
    active={w:sum(b-a for a,b in values) for w,values in intervals.items()}
    overlap=sum(max(0,min(b,d)-max(a,c)) for a,b in intervals['worker-0'] for c,d in intervals['worker-1'])
    union=sum(active.values())-overlap;samples={0:[],1:[]}
    if path.exists():
        for row in csv.DictReader(path.open()):
            gpu=int(row['index']);when=float(row.get('monotonic_seconds') or started+float(row['elapsed_seconds']))
            worker='worker-'+str(gpu)
            for task in tasks:
                mask=(task.get('result') or {}).get('physical_gpu_mask','').replace('GPU-','').lower()
                if mask and mask in (row['index'],row.get('uuid','').replace('GPU-','').lower()):worker=task['worker'];break
            if gpu in samples and any(a<=when<=b for a,b in intervals[worker]):
                try:samples[gpu].append(float(row['utilization.gpu']))
                except ValueError:pass
    means={str(i):sum(v)/len(v) if v else None for i,v in samples.items()}
    return {'active_worker_seconds':active,'overlap_seconds':overlap,'overlap_over_union':overlap/max(1e-9,union),'overlap_target_90pct':overlap/max(1e-9,union)>=.9,'mean_training_gpu_utilization':means,'utilization_target_60pct':all(v is not None and v>=60 for v in means.values()),'note':'performance targets, not artificial work or scientific gates; sampled only while assigned training task active'}


def launch(args):
    started=args.started if args.started is not None else time.monotonic();hard=started+7140
    torch.set_num_threads(1)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();p=json.loads((BASE/'protocol.json').read_text())
    p.update(execution_mode='pilot',pin_memory=True,checkpoint_interval_targets=250000)
    p['smoke']['seed']=17;p['precision'].setdefault('requested','auto');p['pilot']={'name':NAME,'seed':17,'targets_per_model':2000000,'model_count':6,'total_targets':12000000,'gpus':2,'hard_limit_seconds':7200,'platform_reserve_seconds':60,'training_deadline_seconds':6000,'evaluation_deadline_seconds':6900,'cross_session_recovery_validation':'REMOVED_BY_USER','data_pipeline':{'bulk_resident_documents':True,'pin_memory':True,'num_workers':0,'persistent_workers':False,'prefetch_factor':'NOT_APPLICABLE; whole documents cached and copied once asynchronously'}}
    out=args.resume_run or args.output/('clm-v04-pilot-2m-seed17-'+source[:12]);out.mkdir(parents=True,exist_ok=True)
    decision={'format':'minicells.clm-workspace-controller-001.v04','run_id':out.name+'-'+uuid.uuid4().hex[:8],'experiment':NAME,'status':'PILOT_BLOCKED','source_commit':source,'seed':17,'targets_per_model':2000000,'total_targets':12000000,'main_training_started':False,'pilot_training_started':False,'scientific_confirmation':False,'hypotheses':dict.fromkeys(('H1','H2','H3','H4'),'NOT_EVALUATED'),'recovery_validation':'REMOVED_BY_USER; existing infrastructure reused'}
    audit={};phases={};training=[];evaluation=[]
    try:
        audit['observed_gpu_count']=torch.cuda.device_count()
        try:audit['nvidia_smi_L']=subprocess.check_output(['nvidia-smi','-L'],text=True,timeout=5)
        except Exception as error:audit['inventory_error']=str(error)
        devices=inventory();audit['devices']=devices
        audit['nvidia_smi_L']=subprocess.check_output(['nvidia-smi','-L'],text=True)
        audit['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
        for i in range(2):
            free,total=torch.cuda.mem_get_info(i)
            if total-free>2*1024**3:raise RuntimeError('DUAL_GPU_BLOCKED: pre-existing GPU allocation exceeds 2GiB')
        policies=[select(p['precision']['requested'],'cuda:'+str(i)) for i in range(2)]
        if policies[0]!=policies[1]:raise RuntimeError('dual GPU precision policies differ')
        audit['precision_policy']=policies[0]
        configs=match(p['model']);shared.write(out/'model-matching.json',configs)
        if not all(configs[a]['matched'] for a in ('T0-P','T0-F')):raise RuntimeError('corrected baseline matching failed')
        tests=subprocess.run([sys.executable,'-m','pytest','-q',str(BASE/'tests')],cwd=ROOT,env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'},capture_output=True,text=True,timeout=min(120,max(1,hard-time.monotonic())))
        shared.write(out/'correctness-cpu.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr})
        if tests.returncode:raise RuntimeError('CPU correctness gate failed')
        from scripts.research.publish_experiment_results import validate_origin,load_github_token
        validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        with shared.bounded(max(.1,min(started+300,hard)-time.monotonic())):
            docs,manifest=assets.prepare(args.cache,p)
            gate_groups=[docs['train'][i:i+8] for i in range(0,40,8)]
            phases['precision']=benchmark(configs,gate_groups,p,out/'precision-gate',min(started+300,hard))
            phases['controller']=controller_equivalence(p['model'],p,min(started+300,hard))
            phases['gpu_correctness']=correctness(configs,p,min(started+300,hard))
        shared.write(out/'engineering-gates.json',phases);controls=freeze(p['model']);controls['review_budget']={'targets_per_model':2000000,'seeds':[17],'activation':'future independent decision only; excluded from current12M Pilot'};controls['pilot_trigger']='relative NLL or counterfactual gain >=1% is only a follow-up design signal, never single-seed statistical confirmation';shared.write(out/'hypothesis-matching-freeze.json',controls)
        if not all(v['status']=='PASS' for v in phases.values()):raise RuntimeError('engineering GPU gate failed or blocked')
        from main_runner import plan
        plans={}
        for phase,target in [('benchmark',16384),('dry-run',4096),('train',2000000),('evaluate',2000000)]:
            planned,actual=plan(docs['train'],p,target)
            if actual!=target:raise RuntimeError('cannot freeze exact target budget for '+phase)
            plans[phase]=shared.sha([[doc['doc_id'] for doc in group] for group in planned])
        immutable={'plans':plans,'source':source,'protocol':p,'configs':configs,'data':manifest,'precision_policy':policies[0]};run_identity=shared.sha(immutable)
        if not args.no_recovery and not (out/'manifest.json').exists():restore(out,run_identity,min(hard,time.monotonic()+45),artifact_dir='artifacts/experiments/clm-workspace-controller-001/pilot-2m-seed17')
        if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())['identity']!=run_identity:raise ValueError('Pilot resume identity mismatch; fresh restart forbidden')
        if not (out/'manifest.json').exists():shared.write(out/'manifest.json',{'identity':run_identity,**immutable,'devices':devices})
        history_path=out/'session-history.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else []
        prior_active_seconds=sum(item['elapsed_before_publication'] for item in history if item.get('pilot_training_started'))
        hard=min(hard,started+max(0,7140-prior_active_seconds))
        shared.write(out/'protocol.json',p)
        hashes={}
        if (out/'recovery.json').exists():
            bundle=json.loads((out/'recovery.json').read_text())
            if bundle['identity']!={'run_identity':run_identity}:raise RuntimeError('RECOVERY_BLOCKED: root bundle identity mismatch')
            for logical,digest in bundle['files'].items():
                if not logical.startswith('train/') or not logical.endswith('/checkpoint.pt'):continue
                checkpoint=out/logical
                if not checkpoint.exists() or shared.digest(checkpoint)!=digest:
                    snapshot=out/bundle['objects'][logical]
                    if not snapshot.exists() or shared.digest(snapshot)!=digest:raise RuntimeError('RECOVERY_BLOCKED: no verified checkpoint object')
                    import shutil
                    checkpoint.parent.mkdir(parents=True,exist_ok=True);pending=checkpoint.with_suffix('.restoring');shutil.copyfile(snapshot,pending);pending.replace(checkpoint)
                hashes[Path(logical).parent.name]=digest
        for checkpoint in (out/'train').glob('*/checkpoint.pt') if (out/'train').exists() else []:
            if checkpoint.parent.name not in hashes:raise RuntimeError('RECOVERY_BLOCKED: checkpoint lacks immutable hash manifest')
        spec={'protocol':p,'configs':configs,'manifest':manifest,'source_commit':source,'data_cache':str(args.cache),'precision_policy':policies[0],'budgets':{'benchmark':16384,'dry-run':4096,'train':2000000,'evaluate':2000000},'training_output':str(out/'train'),'resume_checkpoint_sha256':hashes,'plan_sha256':plans}
        spec_path=out/'worker-spec.json';shared.write(spec_path,spec)
        gc.collect();torch.cuda.empty_cache()
        priorities={'C2':6,'C3':5,'C1':4,'C0':3,'T0-P':2,'T0-F':1}
        measured=run_workers(out,spec_path,'benchmark',min(started+600,hard-1200),started,priorities,run_identity)
        if len({row['worker'] for row in measured})!=2 or any(row['status']!='COMPLETE' for row in measured):raise RuntimeError('dual-worker benchmark failed')
        rates={r['arm']:r['result']['performance']['session_targets_per_second'] for r in measured}
        uuids={r['result']['gpu_uuid'] for r in measured}
        if 'NOT_EXPOSED' not in uuids and len(uuids)!=2:raise RuntimeError('DUAL_GPU_BLOCKED: workers mapped to the same physical GPU')
        if any(r<=0 for r in rates.values()):raise RuntimeError('invalid concurrent training rate')
        overheads={r['arm']:max(0,r['result']['wall_seconds']-r['result']['performance']['session_seconds']) for r in measured}
        prediction=predict(rates,overhead=overheads);elapsed=time.monotonic()-started
        prediction.update(measured_rates=rates,measured_concurrently=True,elapsed_before_launch=elapsed,training_margin_fraction=.2,finalization_seconds_reserved=600,predicted_complete_seconds=prior_active_seconds+elapsed+prediction['lpt_seconds']*1.2+600,previous_pilot_active_seconds=prior_active_seconds)
        shared.write(out/'dual-gpu-profile.json',{'benchmark_tasks':measured,'prediction':prediction})
        if prediction['predicted_complete_seconds']>6300 or prior_active_seconds+elapsed+prediction['lpt_seconds']*1.2>6000:raise RuntimeError('TIME_BUDGET_BLOCKED: exact 2M per model retained; no reduction')
        dry=run_workers(out,spec_path,'dry-run',min(started+900,hard-1200),started,priorities,run_identity)
        shared.write(out/'pilot-dry-run.json',dry)
        if any(r['status']!='COMPLETE' for r in dry):raise RuntimeError('Pilot dry run failed; no 2M training started')
        if prior_active_seconds+time.monotonic()+prediction['lpt_seconds']*1.2+600>started+6300:raise RuntimeError('TIME_BUDGET_BLOCKED after dry run')
        prediction['elapsed_before_formal_training']=time.monotonic()-started;prediction['predicted_complete_seconds']=prior_active_seconds+prediction['elapsed_before_formal_training']+prediction['lpt_seconds']*1.2+600
        shared.write(out/'pilot-budget-freeze.json',{'status':'FROZEN_FOR_THIS_PILOT_ONLY','identity':run_identity,'source_commit':source,'seed':17,'targets_per_model':2000000,'total_targets':12000000,'model_configs':configs,'precision_policy':policies[0],'schedule':prediction,'no_future_budget_auto_start':True})
        decision['pilot_training_started']=True
        training=run_workers(out,spec_path,'train',started+max(0,6000-prior_active_seconds),started,{a:2000000/r for a,r in rates.items()},run_identity)
        shared.write(out/'training-metrics.json',training)
        complete=len({r['worker'] for r in training})==2 and all(r['status']=='COMPLETE' and r['result'].get('committed_targets')==2000000 for r in training)
        if complete:
            evaluation=run_workers(out,spec_path,'evaluate',started+max(0,6900-prior_active_seconds),started,priorities,run_identity)
            shared.write(out/'evaluation-results.json',evaluation)
        decision['status']='PILOT_PASS' if complete and all((r['result'] or {}).get('loss_declined',False) for r in training) and len(evaluation)==6 and all(r['status']=='COMPLETE' and r['result'].get('validation',{}).get('status')=='COMPLETE' for r in evaluation) else 'PILOT_INCOMPLETE'
        audit['dual_gpu_training']=utilization(out/'dual-gpu-telemetry.csv',started,training)
        shared.write(out/'dual-gpu-utilization.json',audit['dual_gpu_training'])
        decision['committed_targets']=sum(r['result'].get('committed_targets',0) for r in training if r['result'])
        decision['hypotheses']=dict.fromkeys(('H1','H2','H3','H4'),'INCONCLUSIVE')
    except Exception as error:
        decision['error']=type(error).__name__+': '+str(error)
        decision['blocking_gate']='TIME_BUDGET_BLOCKED' if 'TIME_BUDGET_BLOCKED' in str(error) else 'DUAL_GPU_BLOCKED' if 'DUAL_GPU_BLOCKED' in str(error) else 'ENGINEERING_GATE_BLOCKED'
        if decision['pilot_training_started']:decision['status']='PILOT_INCOMPLETE'
    finally:
        decision['elapsed_before_publication']=time.monotonic()-started;decision['wall_clock_limit_seconds']=7200
        previous=locals().get('history',[])
        decision['cumulative_pilot_active_seconds']=locals().get('prior_active_seconds',0)+decision['elapsed_before_publication']
        if decision['status']=='PILOT_PASS' and decision['cumulative_pilot_active_seconds']>7140:decision['status']='PILOT_INCOMPLETE';decision['error']='cumulative pilot time budget exceeded'
        previous.append({'run_id':decision['run_id'],'pilot_training_started':decision['pilot_training_started'],'elapsed_before_publication':decision['elapsed_before_publication'],'status':decision['status']})
        shared.write(out/'session-history.json',previous)
        if 'run_identity' in locals() and (out/'manifest.json').exists():commit_bundle(out,run_identity,out/'session-history.json')
        for name,value in [('audit',audit),('protocol',p),('decision',decision),('routing-diagnostics',{r['arm']:r['result'].get('validation',{}).get('records',[]) for r in evaluation if r['result']}),('counterfactual-routing',{r['arm']:r['result'].get('counterfactual',{}) for r in evaluation if r['result']}),('state-diagnostics',{'scope':'token-local controller; route records and module gradients retained','single_seed':True}),('recovery-report',{'validation':'REMOVED_BY_USER','strategy':'existing immutable-object recovery infrastructure','models':{r['arm']:r['result'].get('checkpoint_sha256') for r in training if r['result']}}),('compute-profile',{r['arm']:r['result'].get('compute_profile',{}) for r in evaluation if r['result']})]:shared.write(out/(name+'.json'),value)
        lines=['# CLM v0.4 双 T4 2M Pilot','',json.dumps(decision,ensure_ascii=False,indent=2),'','| Model | Committed Targets | Training | Validation NLL |','|---|---:|---|---:|']
        for arm in ARMS:
            tr=next((r['result'] for r in training if r['arm']==arm),{}) or {};ev=next((r['result'] for r in evaluation if r['arm']==arm),{}) or {}
            lines.append(f"| {arm} | {tr.get('committed_targets',0)} | {tr.get('status','NOT_RUN')} | {ev.get('validation',{}).get('mean_nll','NOT_RUN')} |")
        lines+=['','本轮只包含六模型、Seed17、每模型2M；缺失诊断为PENDING，不视为门禁通过。不执行32M×三Seed，不宣称单Seed统计显著。跨Kaggle恢复验证按用户要求删除。']
        (out/'RESEARCH-REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
        try:
            with shared.bounded(max(.1,hard-time.monotonic())):publish(out,'clm-workspace-controller-001-pilot-2m-seed17')
        except Exception as error:shared.write(out/'publication-failure.json',{'error':str(error),'output_preserved':True})
        print({'status':decision['status'],'report':str(out/'decision.json')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--started',type=float);parser.add_argument('--output',type=Path,default=Path('/kaggle/working/clm-v04-pilot'));parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/stateful-routing-data'));parser.add_argument('--resume-run',type=Path);parser.add_argument('--no-recovery',action='store_true');launch(parser.parse_args())
