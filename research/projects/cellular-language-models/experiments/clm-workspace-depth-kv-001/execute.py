"""Four existing arms, two isolated workers, three frozen seeds; reuses v0.4 plumbing."""
import argparse,copy,csv,hashlib,json,os,random,subprocess,sys,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'clm-workspace-controller-001'
sys.path.insert(0,str(OLD))
import numpy as np
import torch
from dependencies import ROOT,shared,assets
from models import Model
from training import train,identity
from main_runner import plan
from evaluate import evaluate
from recovery import restore,commit_bundle
from pilot_queue import initialize,claim,finish,rows,predict,connect
from pilot import inventory
from gpu_gate import correctness
from run import publish
from measure import benchmark,compute

NAME='clm-workspace-depth-kv-001'


def worker(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4':raise RuntimeError('worker requires exactly one assigned T4')
    spec=json.loads(args.spec.read_text());p=spec['protocol'];out=Path(spec['output']);phase=args.phase
    docs=json.loads((Path(spec['cache'])/'train.json').read_text());validation=json.loads((Path(spec['cache'])/'validation.json').read_text())
    budget=p['experiment']['smoke_targets'] if phase=='smoke' else 2000000
    groups,actual=plan(docs,p,budget)
    if actual!=budget or shared.sha([[d['doc_id'] for d in g] for g in groups])!=spec['plans']['smoke' if phase=='smoke' else 'train']:raise ValueError('worker data plan differs from freeze')
    for doc in docs:doc['_pinned_ids']=torch.tensor(doc['ids'],dtype=torch.long).pin_memory()
    while time.monotonic()<args.deadline:
        job=claim(args.queue,args.worker)
        if job is None:return
        seed_text,arm=job.split('-',1);seed=int(seed_text);folder=out/phase/job;folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/'checkpoint.pt';began=time.monotonic()
        q=copy.deepcopy(p);q['smoke']['seed']=seed;q.update(schedule_target_budget=budget,execution_mode=NAME+'-'+('train' if phase=='evaluate' else phase))
        # C2 and C-DKV share all training RNG streams within each seed.
        rng_arm='C2' if arm=='C-DKV' else arm
        stream=int.from_bytes(hashlib.sha256((str(seed)+':'+rng_arm+':global').encode()).digest()[:4],'little')
        q['router_seed']=int.from_bytes(hashlib.sha256((str(seed)+':'+rng_arm+':router').encode()).digest()[:4],'little')
        run_identity=identity(q,arm,spec['source'],spec['data']);result={'seed':seed,'arm':arm,'phase':phase,'status':'INCOMPLETE','training_identity':run_identity,'plan_sha256':spec['plans']['smoke' if phase=='smoke' else 'train'],'rng_substreams':{'global':stream,'router':q['router_seed']},'initialization_seed':seed,'source_commit':spec['source'],'gpu':torch.cuda.get_device_name(0),'physical_mask':os.environ['CUDA_VISIBLE_DEVICES']}
        try:
            model=Model(arm,spec['configs'][arm]['config'],seed).cuda()
            random.seed(stream);np.random.seed(stream);torch.manual_seed(stream);torch.cuda.manual_seed_all(stream)
            if phase=='evaluate':
                checkpoint=out/'train'/job/'checkpoint.pt';saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
                expected=json.loads((out/'train'/job/'result.json').read_text())
                if shared.digest(checkpoint)!=expected['checkpoint_sha256'] or saved['identity']!=run_identity or saved['progress']['targets']!=2000000:raise ValueError('evaluation checkpoint identity/hash/targets mismatch')
                model.load_state_dict(saved['model']);del saved
                result['training_performance']=expected['performance'];result['committed_targets']=2000000
                result['validation']=evaluate(model,validation,q,args.deadline)
                if result['validation']['status']!='COMPLETE':raise RuntimeError('validation incomplete')
                result['benchmark']=benchmark(model,q,validation[0]['ids'],args.deadline)
            else:
                if checkpoint.exists():
                    bundle=json.loads((out/'recovery.json').read_text());logical=checkpoint.relative_to(out).as_posix()
                    if logical not in bundle['files'] or shared.digest(checkpoint)!=bundle['files'][logical]:raise ValueError('checkpoint lacks a matching immutable recovery hash')
                def event(path):
                    pending=folder/'checkpoint.pending';pending.write_text(json.dumps({'checkpoint':str(path),'mtime_ns':path.stat().st_mtime_ns}));pending.replace(folder/'checkpoint-event.json')
                progress,performance=train(model,groups,q,args.deadline,checkpoint,run_identity,checkpoint_hook=event)
                result.update(training=progress,performance=performance,checkpoint_sha256=shared.digest(checkpoint),committed_targets=progress['targets'])
                if progress['status']!='COMPLETE':raise TimeoutError('training budget unfinished; checkpoint retained')
                if arm in ('C2','C-DKV') and any(not any(row['module_gradients'].get(module,0)>0 for row in progress['rows']) for module in ('router','controller')):raise RuntimeError('Router/Controller CE path gradient missing')
                if phase=='smoke':result['validation']=evaluate(model,validation[:4],q,args.deadline)
            result['status']='COMPLETE'
        except Exception as error:result['error']=type(error).__name__+': '+str(error)
        finally:
            result['seconds']=time.monotonic()-began;shared.write(folder/'result.json',result);finish(args.queue,job,result['status'],result)
            if 'model' in locals():del model
            torch.cuda.empty_cache()
        if result['status']!='COMPLETE':return


def run_workers(out,spec,phase,jobs,deadline,run_identity):
    queue=out/(phase+'-queue.sqlite');initialize(queue,jobs)
    # Reports are the portable queue state; no completed model is retrained.
    with connect(queue) as db:
        for job,_ in jobs:
            report=out/phase/job/'result.json'
            if report.exists():
                value=json.loads(report.read_text())
                if value['status']=='COMPLETE':db.execute("UPDATE tasks SET status='COMPLETE',result=? WHERE arm=?",(json.dumps(value),job))
        db.execute("UPDATE tasks SET status='PENDING' WHERE status IN ('RUNNING','INCOMPLETE')")
    masks=os.getenv('CUDA_VISIBLE_DEVICES','0,1').split(',');processes=[];logs=[];seen={}
    if len(masks)!=2 or len(set(masks))!=2:raise RuntimeError('ambiguous dual GPU masks')
    try:
        for gpu in range(2):
            log=(out/(phase+'-worker-'+str(gpu)+'.log')).open('a');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':masks[gpu],'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            command=[sys.executable,str(HERE/'execute.py'),'--worker','worker-'+str(gpu),'--phase',phase,'--spec',str(spec),'--queue',str(queue),'--deadline',str(deadline)]
            processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(proc.poll() is None for proc in processes):
            for event in (out/phase).glob('*/checkpoint-event.json'):
                value=event.read_text()
                if seen.get(str(event))!=value:commit_bundle(out,run_identity,Path(json.loads(value)['checkpoint']));seen[str(event)]=value
            for report in (out/phase).glob('*/result.json'):
                value=report.read_text()
                if seen.get(str(report))!=value:commit_bundle(out,run_identity,report);seen[str(report)]=value
            failed=any(row['status']=='INCOMPLETE' for row in rows(queue))
            if failed or time.monotonic()>=deadline:break
            time.sleep(.5)
    finally:
        for proc in processes:
            if proc.poll() is None:proc.terminate()
        for proc in processes:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for log in logs:log.close()
        for path in (out/phase).glob('*/checkpoint.pt'):commit_bundle(out,run_identity,path)
        for path in (out/phase).glob('*/result.json'):commit_bundle(out,run_identity,path)
    result=rows(queue);shared.write(out/(phase+'-tasks.json'),result);commit_bundle(out,run_identity,out/(phase+'-tasks.json'))
    if any(proc.returncode!=0 for proc in processes):raise RuntimeError(phase+' worker exited unsuccessfully; see worker logs')
    if len(result)!=len(jobs) or any(row['status']!='COMPLETE' for row in result):
        detail='; '.join(row['arm']+': '+(row['result'] or {}).get('error',row['status']) for row in result if row['status']!='COMPLETE')
        raise RuntimeError(phase+' incomplete: '+detail+'; completed models retained')
    return result


def analyze(out,results,p):
    import statistics
    values={(r['seed'],r['arm']):r for r in results};seeds=p['experiment']['seeds'];arms=p['experiment']['arms'];delta=[];lines=['# Depth-indexed Workspace results','','Validation is previously used development data; independent confirmation NOT CLAIMED.','','| Seed | Model | NLL | PPL |','|---|---|---:|---:|']
    for seed in seeds:
        metrics={a:values[(seed,a)]['validation'] for a in arms};difference=metrics['C-DKV']['mean_nll']-metrics['C2']['mean_nll'];delta.append(difference)
        if metrics['C2']['documents'] and [(r['doc_id'],r['targets']) for r in metrics['C2']['documents']]!=[(r['doc_id'],r['targets']) for r in metrics['C-DKV']['documents']]:raise ValueError('paired validation document mismatch')
        paired=[{'doc_id':a['doc_id'],'targets':a['targets'],'nll_sum_difference':b['nll_sum']-a['nll_sum']} for a,b in zip(metrics['C2']['documents'],metrics['C-DKV']['documents'])]
        shared.write(out/'metrics'/('seed-'+str(seed)+'.json'),{'seed':seed,'models':metrics,'delta_nll':difference,'paired_documents':paired,'scope':'development validation; doc pairs are not independent training seeds'})
        for arm in arms:lines.append('| '+str(seed)+' | '+arm+' | '+str(metrics[arm]['mean_nll'])+' | '+str(metrics[arm]['ppl'])+' |')
    aggregate={'status':'THREE_SEED_DEVELOPMENT_COMPARISON_COMPLETE','seed_count':len(seeds),'delta_nll_by_seed':dict(zip(map(str,seeds),delta)),'delta_nll_mean':statistics.mean(delta),'delta_nll_std':statistics.stdev(delta),'models':{a:{'nll_mean':statistics.mean(values[(s,a)]['validation']['mean_nll'] for s in seeds),'nll_std':statistics.stdev(values[(s,a)]['validation']['mean_nll'] for s in seeds),'ppl_mean':statistics.mean(values[(s,a)]['validation']['ppl'] for s in seeds),'ppl_std':statistics.stdev(values[(s,a)]['validation']['ppl'] for s in seeds)} for a in arms},'scientific_confirmation':False,'decision':'REVIEW_REQUIRED; no frozen significance or quality-versus-cost threshold; do not equate three favorable seeds with independent confirmation'}
    shared.write(out/'metrics/aggregate.json',aggregate)
    columns={'prefill':('prefill','full_context_prefill','ttft'),'decode':('tpot','decode_tokens_per_second'),'memory':('kv_at_full_context','peak_generation_allocated_bytes','peak_full_prefill_allocated_bytes','peak_allocated_bytes')}
    for kind in ('prefill','decode','memory','compute'):
        data={}
        for result in results:
            key=str(result['seed'])+'/'+result['arm'];measured=result['benchmark']
            if kind=='compute':data[key]=measured['compute']
            else:
                fields=('context_limit','prompt_tokens','batch','generated_tokens')+columns[kind]
                data[key]=[{k:v for k,v in case.items() if k in fields} for case in measured['cases']]
        shared.write(out/'benchmarks'/(kind+'.json'),data)
    shared.write(out/'tests/trained-parity.json',{str(r['seed'])+'/'+r['arm']:r['benchmark']['trained_parity'] for r in results})
    report=out/'report';report.mkdir(exist_ok=True)
    lines+=['','## Cost and inference','','GPU-resident prompts; context is total sequence limit. See raw benchmark JSON for samples and scope.','','| Seed | Model | Parameters | Forward matrix FLOPs/token | Training targets/s |','|---|---|---:|---:|---:|']
    for r in results:
        lines.append('| '+str(r['seed'])+' | '+r['arm']+' | '+str(r['benchmark']['parameters'])+' | '+str(r['benchmark']['compute']['forward_flops_per_token'])+' | '+str(r['training_performance']['session_targets_per_second'])+' |')
    lines+=['','| Seed | Model | Context | Batch | Prefill P50 ms | TTFT P95 ms | TPOT P50/P95 ms | Decode tok/s | Peak bytes | KV logical/retained bytes |','|---|---|---:|---:|---:|---:|---|---:|---:|---|']
    for r in results:
        for case in r['benchmark']['cases']:
            lines.append('| '+str(r['seed'])+' | '+r['arm']+' | '+str(case['context_limit'])+' | '+str(case['batch'])+' | '+str(case['prefill']['p50_seconds']*1000)+' | '+str(case['ttft']['p95_seconds']*1000)+' | '+str(case['tpot']['p50_seconds']*1000)+' / '+str(case['tpot']['p95_seconds']*1000)+' | '+str(case['decode_tokens_per_second'])+' | '+str(case['peak_allocated_bytes'])+' | '+str(case['kv_at_full_context']['logical_total'])+' / '+str(case['kv_at_full_context']['retained_storage_bytes'])+' |')
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n\n'+json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n')
    with (report/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','model','validation_nll','ppl','training_targets'])
        for r in results:writer.writerow([r['seed'],r['arm'],r['validation']['mean_nll'],r['validation']['ppl'],2000000])
    return aggregate


def launch(args):
    started=args.started or time.monotonic();p=json.loads((HERE/'CONFIG.json').read_text());baseline=json.loads((HERE/'baseline-freeze.json').read_text());hard=started+p['experiment']['session_seconds']-60
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();out=args.resume_run or args.output/('clm-depth-kv-001-'+source[:12]);out.mkdir(parents=True,exist_ok=True)
    decision={'format':'minicells.clm-workspace-depth-kv-001.v1','run_id':out.name+'-'+uuid.uuid4().hex[:8],'source_commit':source,'status':'INCOMPLETE','pilot_training_started':False,'scientific_confirmation':False,'independent_test_accessed':False,'hypotheses':{'H1':'NOT_EVALUATED','H0':'NOT_EVALUATED'}}
    run_identity=None;training=[];evaluation=[]
    try:
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        devices=inventory()
        check=subprocess.run([sys.executable,'-m','pytest','-q',str(HERE/'tests')],cwd=ROOT,env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'},capture_output=True,text=True,timeout=120)
        shared.write(out/'tests/cpu.json',{'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr})
        if check.returncode:raise RuntimeError('core CPU checks failed')
        from scripts.research.publish_experiment_results import validate_origin,load_github_token
        validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        docs,data=assets.prepare(args.cache,p)
        if data!=baseline['data']:raise ValueError('baseline data/tokenizer identity mismatch')
        groups,actual=plan(docs['train'],p,2000000);smoke,_=plan(docs['train'],p,p['experiment']['smoke_targets'])
        plans={'train':shared.sha([[d['doc_id'] for d in g] for g in groups]),'smoke':shared.sha([[d['doc_id'] for d in g] for g in smoke])}
        if actual!=2000000 or len(groups)!=1068 or plans['train']!=baseline['plans']['train']:raise ValueError('baseline exact exposure plan/update count mismatch')
        if len(docs['validation'])!=32 or sum(len(d['ids'])-1 for d in docs['validation'])!=5997:raise ValueError('baseline validation sample count mismatch')
        configs={a:copy.deepcopy(baseline['configs']['C2' if a=='C-DKV' else a]) for a in p['experiment']['arms']}
        dkv_cost=compute('C-DKV',p['model']);configs['C-DKV']['flops']={'total':dkv_cost['forward_flops_per_token'],'parts':dkv_cost['parts'],'convention':dkv_cost['convention']}
        immutable={'source':source,'baseline_source':baseline['source'],'baseline_result_commit':baseline['result_commit'],'protocol':p,'data':data,'configs':configs,'plans':plans};run_identity=shared.sha(immutable)
        if not (out/'manifest.json').exists():restore(out,run_identity,min(hard,time.monotonic()+45),artifact_dir='artifacts/experiments/'+NAME)
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['identity']!=run_identity:raise ValueError('resume source/config/data identity mismatch')
        else:shared.write(out/'manifest.json',{'identity':run_identity,**immutable,'devices':devices})
        if (out/'recovery.json').exists():
            import shutil
            bundle=json.loads((out/'recovery.json').read_text())
            if bundle['identity']!={'run_identity':run_identity}:raise ValueError('recovery bundle identity mismatch')
            for logical,digest in bundle['files'].items():
                if not logical.endswith('/checkpoint.pt'):continue
                path=out/logical
                if not path.exists() or shared.digest(path)!=digest:
                    snapshot=out/bundle['objects'][logical]
                    if not snapshot.exists() or shared.digest(snapshot)!=digest:raise ValueError('no verified checkpoint snapshot: '+logical)
                    path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_suffix('.restoring');shutil.copyfile(snapshot,pending);pending.replace(path)
        shared.write(out/'protocol.json',p)
        if (out/'decision.json').exists() and json.loads((out/'decision.json').read_text())['status']=='COMPLETE':return
        # Existing correctness logic covers all arms; it checks depth1 cache too.
        gates=correctness(configs,p,min(hard,started+480));shared.write(out/'tests/gpu-correctness.json',gates)
        if gates['status']!='PASS':raise RuntimeError('GPU causal/cache/decode checks failed')
        for name in ('causal_correctness','cache_correctness','training_inference_parity','regression'):
            shared.write(out/'tests'/(name+'.json'),{'status':'PASS','cpu_evidence':'cpu.json','gpu_evidence':'gpu-correctness.json','scope':'engineering only; includes depth1 cache; no scientific conclusion'})
        spec={'protocol':p,'output':str(out),'cache':str(args.cache),'data':data,'configs':configs,'source':source,'plans':plans};spec_path=out/'worker-spec.json';shared.write(spec_path,spec)
        smoke_rows=run_workers(out,spec_path,'smoke',[('17-'+a,i) for i,a in enumerate(p['experiment']['arms'])],min(hard,started+900),run_identity)
        rates={row['result']['arm']:row['result']['performance']['session_targets_per_second'] for row in smoke_rows}
        jobs=[(str(seed)+'-'+arm,2000000/rates[arm]) for seed in p['experiment']['seeds'] for arm in p['experiment']['arms']]
        remaining={job:2000000 for job,_ in jobs}
        for job,_ in jobs:
            report=out/'train'/job/'result.json';checkpoint=out/'train'/job/'checkpoint.pt'
            if report.exists() and json.loads(report.read_text()).get('committed_targets')==2000000:remaining[job]=0
            elif checkpoint.exists():
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False);remaining[job]=2000000-saved['progress']['targets'];del saved
        forecast=predict({job:rates[job.split('-',1)[1]] for job,_ in jobs},targets=remaining);forecast['remaining_targets']=remaining
        shared.write(out/'budget-freeze.json',{'targets_per_model':2000000,'seeds':p['experiment']['seeds'],'total_targets':24000000,'schedule':forecast,'precision':'fp32','source_commit':source,'session_seconds':7200})
        predicted_train_end=time.monotonic()+1.2*forecast['lpt_seconds']
        if predicted_train_end>started+p['experiment']['execution_seconds'] or predicted_train_end+600>hard:raise RuntimeError('TIME_BUDGET_BLOCKED: no target reduction; measured remaining-cohort estimate exceeds session')
        decision['pilot_training_started']=True
        training=run_workers(out,spec_path,'train',jobs,started+p['experiment']['execution_seconds'],run_identity)
        evaluation=run_workers(out,spec_path,'evaluate',jobs,started+p['experiment']['evaluation_seconds'],run_identity)
        aggregate=analyze(out,[row['result'] for row in evaluation],p)
        decision.update(status='COMPLETE',training_status='COMPLETE',evaluation_status='COMPLETE',training_targets=sum(row['result']['committed_targets'] for row in training),research_review='PENDING',hypotheses={'H1':'REVIEW_REQUIRED','H0':'REVIEW_REQUIRED'})
    except Exception as error:decision['error']=type(error).__name__+': '+str(error)
    finally:
        # Completed package is published unchanged on retry.
        if not ((out/'decision.json').exists() and json.loads((out/'decision.json').read_text())['status']=='COMPLETE'):
            decision['elapsed_seconds']=time.monotonic()-started;shared.write(out/'decision.json',decision);shared.write(out/'protocol.json',p)
            (out/'RESEARCH-REPORT.zh-CN.md').write_text('# CLM-Workspace-DepthKV-001\n\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n\nDetailed metrics/benchmarks/report are emitted only when actually completed.\n')
            if run_identity:
                for path in sorted(out.rglob('*.json')):
                    if '_recovery_objects' not in path.parts and not path.name.startswith('recovery'):commit_bundle(out,run_identity,path)
        try:
            with shared.bounded(max(.1,hard-time.monotonic())):publish(out,NAME)
        except Exception as error:shared.write(out/'publication-failure.json',{'error':str(error),'output_preserved':True})
        print({'status':json.loads((out/'decision.json').read_text())['status'],'report':str(out/'decision.json'),**({'error':decision['error']} if 'error' in decision else {})},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/kaggle/working/clm-depth-kv'));parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/stateful-routing-data'));parser.add_argument('--started',type=float);parser.add_argument('--resume-run',type=Path);parser.add_argument('--worker');parser.add_argument('--phase',choices=['smoke','train','evaluate']);parser.add_argument('--spec',type=Path);parser.add_argument('--queue',type=Path);parser.add_argument('--deadline',type=float);parser.add_argument('--publish-only',type=Path);args=parser.parse_args()
    if args.worker:worker(args)
    elif args.publish_only:publish(args.publish_only,NAME)
    else:launch(args)
