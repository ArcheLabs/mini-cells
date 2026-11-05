"""Fresh subprocess, physical GPU masked before torch import; no DDP or fork."""
import os
import hashlib,random
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,json,time,copy,traceback
from pathlib import Path
import torch
from dependencies import BASE,shared
from models import Model
from precision import select
from training import train,identity
from main_runner import plan
from evaluate import evaluate,profile
from pilot_queue import claim,finish


def execute(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4':raise RuntimeError('worker must see exactly one assigned Tesla T4')
    spec=json.loads(args.spec.read_text());p=spec['protocol'];phase=args.phase;deadline=args.deadline;out=args.output
    policy=select(p['precision'].get('requested','auto'),'cuda:0')
    if policy!=spec['precision_policy']:raise ValueError('worker precision differs from frozen dual-GPU policy')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    docs=json.loads((Path(spec['data_cache'])/'train.json').read_text());validation=json.loads((Path(spec['data_cache'])/'validation.json').read_text())
    groups,actual=plan(docs,p,spec['budgets'][phase])
    if actual!=spec['budgets'][phase]:raise ValueError('exact complete-document target plan unavailable')
    if shared.sha([[d['doc_id'] for d in group] for group in groups])!=spec['plan_sha256'][phase]:raise ValueError('worker document plan differs from parent freeze')
    for doc in docs:doc['_pinned_ids']=torch.tensor(doc['ids'],dtype=torch.long).pin_memory()
    while time.monotonic()<deadline:
        arm=claim(args.queue,args.worker)
        if arm is None:break
        started=time.monotonic();folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
        q=copy.deepcopy(p);q['schedule_target_budget']=actual;q['execution_mode']='pilot-'+phase
        global_seed=int.from_bytes(hashlib.sha256(('17:'+arm+':global').encode()).digest()[:4],'little')
        router_seed=int.from_bytes(hashlib.sha256(('17:'+arm+':router').encode()).digest()[:4],'little')
        q['router_seed']=router_seed
        q['checkpoint_interval_targets']=250000 if phase=='train' else 0;q['pin_memory']=True
        run_identity=identity(q,arm,spec['source_commit'],spec['manifest']);checkpoint=folder/'checkpoint.pt'
        result={'arm':arm,'worker':args.worker,'physical_gpu_mask':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(0),'gpu_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','NOT_EXPOSED')),'precision_policy':policy,'phase':phase,'started':started,'status':'INCOMPLETE'}
        def checkpoint_event(path):
            pending=folder/'checkpoint-event.pending';pending.write_text(json.dumps({'checkpoint':str(path),'mtime_ns':path.stat().st_mtime_ns})+'\n');pending.replace(folder/'checkpoint-event.json')
        try:
            if phase=='train' and checkpoint.exists():
                expected_hash=spec.get('resume_checkpoint_sha256',{}).get(arm)
                if expected_hash and shared.digest(checkpoint)!=expected_hash:raise ValueError('RECOVERY_BLOCKED: checkpoint SHA mismatch')
            model=Model(arm,spec['configs'][arm]['config'],17).cuda()
            random.seed(global_seed);np.random.seed(global_seed);torch.manual_seed(global_seed);torch.cuda.manual_seed_all(global_seed)
            result['rng_substreams']={'initialization_seed':17,'global_seed':global_seed,'router_seed':router_seed,'data_order_seed':17,'data_order':'deterministic ordered documents, no stochastic shuffle'}
            if phase=='evaluate':
                # Read frozen Pilot weights only; never call train.
                checkpoint=Path(spec['training_output'])/arm/'checkpoint.pt'
                saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
                train_q=copy.deepcopy(p);train_q.update(schedule_target_budget=2000000,execution_mode='pilot-train',checkpoint_interval_targets=250000,pin_memory=True,router_seed=router_seed)
                expected=identity(train_q,arm,spec['source_commit'],spec['manifest'])
                if saved['identity']!=expected or saved['precision_policy']!=policy:raise ValueError('evaluation checkpoint identity mismatch')
                model.load_state_dict(saved['model']);result['committed_targets']=saved['progress']['targets'];del saved
            else:
                if phase=='dry-run' and not checkpoint.exists():
                    # Local same-process resume checks training interface, not Kaggle recovery.
                    first=copy.deepcopy(q);first['session_stop_targets']=sum(len(d['ids'])-1 for d in groups[0])
                    train(model,groups,first,deadline,checkpoint,run_identity,checkpoint_hook=checkpoint_event)
                def milestone(committed):
                    previous_generator=model.generator;cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();noise=model.noise
                    try:
                        value=evaluate(model,validation[:4],q,deadline)
                        return {'targets':committed,'status':value['status'],'mean_nll':value['mean_nll'],'ppl':value['ppl'],'seconds':value['seconds'],'subset':[d['doc_id'] for d in validation[:4]]}
                    finally:
                        model.generator=previous_generator;model.train();model.noise=noise;torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                progress,performance=train(model,groups,q,deadline,checkpoint,run_identity,resume=phase!='benchmark',checkpoint_hook=checkpoint_event,validation_hook=milestone if phase=='train' else None)
                result.update(training=progress,performance=performance,committed_targets=progress['targets'],checkpoint_sha256=shared.digest(checkpoint) if checkpoint.exists() else None)
                span=max(1,len(progress['rows'])//10)
                result['loss_declined']=sum(r['lm_loss'] for r in progress['rows'][-span:])<sum(r['lm_loss'] for r in progress['rows'][:span])
                shared.write(folder/'training.json',progress);shared.write(folder/'performance.json',performance)
                if progress['status']!='COMPLETE':raise TimeoutError('phase ended before exact target budget')
                required=['router'] if arm in ('C1','C2','C3') else []
                if arm in ('C2','C3'):required.append('controller')
                if any(not any(row['module_gradients'].get(name,0)>0 for row in progress['rows']) for name in required):raise RuntimeError('required Router/Controller gradient absent')
            if phase in ('dry-run','evaluate'):
                subset=validation[:(p['evaluation']['documents'] if phase=='evaluate' else 4)]
                result['validation']=evaluate(model,subset,p,deadline)
                if phase=='evaluate':
                    diagnostics=validation[:8];counter={}
                    if arm in ('C1','C2','C3'):
                        if time.monotonic()<deadline:
                            try:counter['normal']=evaluate(model,diagnostics,q,deadline)
                            except TimeoutError:counter['normal']={'status':'PENDING','reason':'diagnostic deadline'}
                        for mode in ('random','shuffle','fixed'):
                            if time.monotonic()>=deadline:counter[mode]={'status':'PENDING'};continue
                            try:counter[mode]=evaluate(model,diagnostics,p,deadline,mode,pair=(0,1))
                            except TimeoutError:counter[mode]={'status':'PENDING','reason':'diagnostic deadline'}
                        if arm=='C2' and time.monotonic()<deadline:
                            try:counter['state_off']=evaluate(model,diagnostics,p,deadline,'state_off')
                            except TimeoutError:counter['state_off']={'status':'PENDING','reason':'diagnostic deadline'}
                    result['counterfactual']=counter
                    if time.monotonic()<deadline:result['compute_profile']=profile(model,p,deadline,policy)
            result['status']='COMPLETE';finish(args.queue,arm,'COMPLETE',result)
        except Exception as failure:
            if checkpoint.exists() and 'committed_targets' not in result:
                try:
                    valid=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if valid['identity']==run_identity:result.update(committed_targets=valid['progress']['targets'],completed_updates=valid['progress']['updates'],checkpoint_sha256=shared.digest(checkpoint))
                    del valid
                except Exception as checkpoint_error:result['checkpoint_error']=str(checkpoint_error)
            result.update(error=type(failure).__name__+': '+str(failure),traceback=traceback.format_exc()[-4000:]);finish(args.queue,arm,'INCOMPLETE',result)
        finally:
            result['ended']=time.monotonic();result['wall_seconds']=result['ended']-started;shared.write(folder/'result.json',result)
            # Update queue with final wall clock fields as well.
            finish(args.queue,arm,result['status'],result)
            if 'model' in locals():del model
            torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--worker',required=True);parser.add_argument('--phase',required=True);parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--output',type=Path,required=True);execute(parser.parse_args())
