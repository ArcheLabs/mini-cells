"""Temporary benchmark, bounded curriculum phases and independent restore."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from p3_data import BASE,forward_batch,write,file_hash,digest
from p3_metrics import candidate_loss,statistics,evaluate,stage1_gate
from p3_training import train_stage
from b0_models import construct,state_hash,parameter_report
from b0_integrity import environment
from e0 import save_checkpoint,load_checkpoint
sys.path.insert(0,str(BASE.parent/'b0-p2'))
from p2_training import ObservedAdamW


def benchmark(model,rows,val,p,folder,level):
    folder.mkdir(parents=True,exist_ok=True);initial=state_hash(model);optimizer=ObservedAdamW(model.parameters(),lr=p['training']['lr'],betas=tuple(p['training']['betas']),weight_decay=p['training']['weight_decay'])
    model.eval()
    with torch.no_grad():
        group=rows[:8];baseline=forward_batch(model,group)
        altered=forward_batch(model,[rows[8],*group[1:]])
        isolation_error=float((baseline[1:]-altered[1:]).abs().max())
        if not np.isfinite(isolation_error) or isolation_error>2e-6:raise ValueError('GPU native batch isolation failed')
        changed=copy.deepcopy(group);answer=changed[0]['query_positions'][0];valid=changed[0]['valid_length']
        changed[0]['input_ids'][answer+1:valid]=[264]*(valid-answer-1)
        future=forward_batch(model,changed);causal_error=float((baseline[0,:answer+1]-future[0,:answer+1]).abs().max())
        if not np.isfinite(causal_error) or causal_error>2e-6:raise ValueError('GPU future-answer leakage')
    measured=[];warmup=[]
    for i in range(p['benchmark']['warmup_updates']+p['benchmark']['measured_updates']):
        selected=rows[i*8:(i+1)*8];total=sum(len(r['query_positions']) for r in selected)
        model.train();optimizer.zero_grad(set_to_none=True);fwd=back=0.
        torch.cuda.synchronize();start=time.perf_counter();logits=forward_batch(model,selected)
        loss=torch.stack([candidate_loss(logits[i],row) for i,row in enumerate(selected)]).sum()/total
        for i,row in enumerate(selected):statistics(logits[i,row['query_positions']],row)
        torch.cuda.synchronize();fwd=time.perf_counter()-start
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark')
        start=time.perf_counter();loss.backward();torch.cuda.synchronize();back=time.perf_counter()-start
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['training']['clip'],error_if_nonfinite=True)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite benchmark gradient')
        start=time.perf_counter();optimizer.step();torch.cuda.synchronize();opt=time.perf_counter()-start
        (warmup if i<p['benchmark']['warmup_updates'] else measured).append({'forward':fwd,'backward':back,'optimizer':opt,'total':fwd+back+opt})
    start=time.perf_counter();evaluate(model,val,folder/'validation',full_probabilities=True,draws=p['gate']['bootstrap_draws']);validation=time.perf_counter()-start
    start=time.perf_counter();save_checkpoint({'model':dict(model.state_dict()),'optimizer':optimizer.state_dict()},folder/'temporary.pt');checkpoint=time.perf_counter()-start
    write(folder/'benchmark.json',{'temporary_run_id':'b0-p3-temp-'+folder.parent.name+'-'+str(level)+'-'+str(time.time_ns()),
        'gpu_preflight':{'status':'PASS','batch_isolation_max_abs_error':isolation_error,'causal_max_abs_error':causal_error,'tolerance':2e-6},'warmup':warmup,'steps':measured,'validation_seconds':validation,'checkpoint_seconds':checkpoint,
        'logical_batch_size':8,'storage_length':64 if level==8 else 128,'processed_length':59 if level==8 else 115,
        'formal_weights_reused':False,'temporary_optimizer_updates':len(warmup)+len(measured),'initialization_sha256':initial,'parameters':parameter_report(model),'environment':environment()})


def independent_verify(model,data,p,folder,identity):
    stages=[1,2] if (folder/'checkpoints/stage2-final.pt').exists() else [1];proof={}
    for stage in stages:
        checkpoint=folder/('checkpoints/stage'+str(stage)+'-final.pt');saved=load_checkpoint(checkpoint)
        if saved['identity']!=identity or saved['stage']!=stage:raise ValueError('Independent checkpoint identity differs')
        model.load_state_dict(saved['model']);level=str(p['stages'][str(stage)]['level'])
        rows=json.loads((data/(level+'-validation.json')).read_text());target=folder/'independent-restore'/str(stage)
        report=evaluate(model,rows,target,full_probabilities=True,draws=p['gate']['bootstrap_draws'])
        original=folder/'training'/('stage'+str(stage))/'evaluation'/str(p['stages'][str(stage)]['updates'])
        expected=json.loads((original/'metrics.json').read_text())
        errors={}
        with np.load(original/'probabilities.npz') as a,np.load(target/'probabilities.npz') as b:
            for name in ('targets','candidate_ids','mapping_hashes','active_predictions','value_predictions','full_predictions'):
                if not np.array_equal(a[name],b[name]):raise ValueError('Restored labels/predictions differ:'+name)
            for name in ('full_log_probabilities','value_log_probabilities','candidate_log_probabilities'):
                finite=np.isfinite(a[name])
                if not np.array_equal(finite,np.isfinite(b[name])):raise ValueError('Restored candidate masks differ')
                error=float(np.abs(a[name][finite]-b[name][finite]).max());errors[name]=error
                if error>2e-5:raise ValueError('Restored probabilities differ')
        if abs(report['active_nll']-expected['active_nll'])>2e-5:raise ValueError('Restored NLL differs')
        proof[str(stage)]={'status':'PASS','checkpoint_sha256':file_hash(checkpoint),'probability_log_max_abs_error':errors,'tolerance':2e-5}
    write(folder/'independent-restore/verification.json',{'status':'PASS','stages':proof,'new_optimizer_updates':0,'test_accessed':False})


def work(args):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('BLOCKED_HARDWARE: actual Tesla T4 required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    p=json.loads(args.protocol.read_text());folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    model=construct(args.arm,p).cuda();initial=state_hash(model)
    if args.action.startswith('benchmark'):
        stage=int(args.action[-1]);level=p['stages'][str(stage)]['level'];rows=json.loads((args.data/(str(level)+'-train.json')).read_text())
        val=json.loads((args.data/(str(level)+'-validation.json')).read_text());benchmark(model,rows,val,p,folder,level);return
    identity=json.loads((folder/'run-manifest.json').read_text())
    if identity['initialization_sha256']!=initial:raise ValueError('Fresh initialization differs')
    if args.action=='verify':independent_verify(model,args.data,p,folder,identity);return
    if args.action.startswith('isolated'):
        stage=int(args.action[-1]);level=p['stages'][str(stage)]['level'];saved=load_checkpoint(folder/('checkpoints/stage'+str(stage)+'-final.pt'))
        model.load_state_dict(saved['model']);rows=json.loads((args.data/(str(level)+'-query_isolated.json')).read_text())
        result=evaluate(model,rows,folder/'query-isolated'/str(stage),draws=p['gate']['bootstrap_draws'])
        result['interpretation']='Single Query changes Query position and chunk/Store history; differences do not isolate answer-history causality.'
        write(folder/'query-isolated'/str(stage)/'metrics.json',result);return
    stage=int(args.action[-1]);level=p['stages'][str(stage)]['level']
    if stage==1 and p['stages']['1']['updates']!=128:raise ValueError('Canonical Stage1 requires128 updates')
    if stage==2:
        gate=json.loads((folder.parent/'stage1-gate.json').read_text())
        if gate['status']!='PASS' or set(gate['arms'])!=set(p['models_order']) or any(r['status']!='PASS' for r in gate['arms'].values()):raise PermissionError('All-arm Stage1 gate required')
        if gate['protocol_sha256']!=digest(p):raise ValueError('Stage1 gate protocol differs')
        for arm,proof in gate['arms'].items():
            previous=json.loads((folder.parent/arm/'stage1-result.json').read_text())
            if stage1_gate(previous['validation'],previous['online'],p)['status']!='PASS':raise PermissionError('Recomputed Stage1 gate failed')
            if file_hash(folder.parent/arm/'checkpoints/stage1-final.pt')!=proof['checkpoint_sha256']:raise ValueError('Stage1 gate checkpoint differs')
    rows=json.loads((args.data/(str(level)+'-train.json')).read_text());val=json.loads((args.data/(str(level)+'-validation.json')).read_text())
    result=train_stage(model,rows,val,p,folder/'training'/('stage'+str(stage)),identity,stage,
        continuation=folder/'checkpoints/stage1-final.pt' if stage==2 else None)
    if result['status']!='COMPLETE':raise RuntimeError('Incomplete stage')
    target=folder/('checkpoints/stage'+str(stage)+'-final.pt');target.parent.mkdir(exist_ok=True)
    if not target.exists():os.link(folder/'training'/('stage'+str(stage))/'checkpoint.pt',target)
    write(folder/'initialization.json',{'sha256':initial,'historical_weights_loaded':False,'exclusive_e7_parameters':0,'exclusive_e7_initialization':'NOT_APPLICABLE'})
    write(folder/'environment.json',environment());write(folder/'parameters.json',parameter_report(model))
    write(folder/('stage'+str(stage)+'-result.json'),result)


def launch(args):
    script="import os,sys,time,signal; p=int(sys.argv[1]); end=time.monotonic()+float(sys.argv[2]);\nwhile os.getppid()==p and time.monotonic()<end: time.sleep(.02)\nif os.getppid()==p: os.kill(p,signal.SIGKILL)"
    watcher=subprocess.Popen([sys.executable,'-c',script,str(os.getpid()),str(args.remaining)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:work(args)
    finally:watcher.terminate();watcher.wait(timeout=5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',required=True);p.add_argument('--arm',required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remaining',type=float,required=True);launch(p.parse_args())
