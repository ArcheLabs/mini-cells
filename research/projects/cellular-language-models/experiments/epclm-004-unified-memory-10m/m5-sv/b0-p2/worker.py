"""One bounded GPU phase; the controller owns cumulative accounting."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from p2_dataset import BASE
from p2_training import fit,ObservedAdamW,pure_checkpoint
from p2_metrics import evaluate
from b0_models import construct,state_hash,parameter_report
from b0_integrity import environment
from data import batch
from e0 import write,masked_loss,rng,Sampler,load_checkpoint


def benchmark(model,train,val,diag,p,folder):
    folder.mkdir(parents=True,exist_ok=True);initial=state_hash(model);optimizer=ObservedAdamW(model.parameters(),lr=p['training']['lr'],
        betas=tuple(p['training']['betas']),weight_decay=p['training']['weight_decay'])
    timings=[];warmup=[]
    warmup_steps=p['p2']['benchmark_warmup_updates']
    for i in range(p['p2']['benchmark_updates']+warmup_steps):
        b=batch([train[i]],'cuda');model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        start=time.perf_counter();loss=masked_loss(model(b.input_ids),b.labels);torch.cuda.synchronize();forward=time.perf_counter()-start
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
        start=time.perf_counter();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['training']['clip'],error_if_nonfinite=True)
        torch.cuda.synchronize();backward=time.perf_counter()-start
        start=time.perf_counter();optimizer.step();torch.cuda.synchronize();update=time.perf_counter()-start
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite benchmark')
        (warmup if i<warmup_steps else timings).append({'forward':forward,'backward':backward,'optimizer':update,'total':forward+backward+update})
    start=time.perf_counter();evaluate(model,val,folder/'validation',draws=p['metrics']['bootstrap_draws']);validation=time.perf_counter()-start
    start=time.perf_counter();evaluate(model,diag,folder/'diagnostic',draws=p['metrics']['bootstrap_draws']);diagnostic=time.perf_counter()-start
    sampler=Sampler(train,p['seeds']['sampler'],1)
    start=time.perf_counter();pure_checkpoint({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng(True),
        'sampler':sampler.snapshot(),'ledger':{'committed_updates':0},'identity':{'phase':'temporary-benchmark'}},folder/'benchmark.pt')
    checkpoint=time.perf_counter()-start
    result={'status':'PASS','temporary_run_id':'b0-p2-benchmark-'+folder.name,'optimizer_updates_not_in_formal_ledger':len(timings)+len(warmup),'warmup_steps':warmup,
        'steps':timings,'validation_seconds':validation,'diagnostic_seconds':diagnostic,'checkpoint_seconds':checkpoint,
        'environment':environment(),'formal_weights_reused':False,'initialization_sha256':initial,'parameters':parameter_report(model)}
    write(folder/'benchmark.json',result)


def verify(folder,model,train,val,diag,p,identity):
    saved=load_checkpoint(folder/'checkpoints/final.pt')
    if saved['identity']!=identity:raise ValueError('Restored identity differs')
    steps=p['training']['updates'];ledger=saved['ledger']
    if any(ledger[k]!=v for k,v in {'committed_updates':steps,'input_tokens':steps*512,'supervised_target_tokens':steps*8,'query_target_tokens':steps*8}.items()):
        raise ValueError('Restored ledger differs')
    model.load_state_dict(saved['model']);hash_restored=state_hash(model)
    result={}
    for kind,rows in [('training',diag),('validation',val)]:
        output=evaluate(model,rows,folder/'independent-restored'/kind,draws=p['metrics']['bootstrap_draws'])
        original=json.loads((folder/'training/evaluation'/str(steps)/kind/'metrics.json').read_text())
        import numpy as np
        with np.load(folder/'independent-restored'/kind/'probabilities.npz') as a,np.load(folder/'training/evaluation'/str(steps)/kind/'probabilities.npz') as b:
            error=float(np.abs(a['probabilities']-b['probabilities']).max())
            if not np.array_equal(a['targets'],b['targets']) or not np.array_equal(a['episode_ids'],b['episode_ids']):raise ValueError('Restored probability identity differs')
        if output['correct']!=original['correct'] or abs(output['nll']-original['nll'])>2e-5 or error>2e-5:
            raise ValueError('Restored predictions differ')
        result[kind]={'status':'PASS','correct':output['correct'],'nll_abs_error':abs(output['nll']-original['nll']),
            'probability_max_abs_error':error,'tolerance':2e-5}
    write(folder/'independent-restored/verification.json',{'status':'PASS','restored_state_sha256':hash_restored,
        'new_optimizer_updates':0,'checks':result,'test_accessed':False})


def launch(args):
    #A child watchdog notices parent exit and cannot kill a later reused PID.
    script="import os,sys,time,signal; p=int(sys.argv[1]); end=time.monotonic()+float(sys.argv[2]);\nwhile os.getppid()==p and time.monotonic()<end: time.sleep(.02)\nif os.getppid()==p: os.kill(p,signal.SIGKILL)"
    watchdog=subprocess.Popen([sys.executable,'-c',script,str(os.getpid()),str(args.remaining)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('BLOCKED_HARDWARE: Tesla T4 required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        p=json.loads(args.protocol.read_text());data=args.data
        train=json.loads((data/'training.json').read_text());val=json.loads((data/'validation.json').read_text())
        diag=train[:p['p2']['training_diagnostic_episodes']]
        model=construct(args.arm,p).cuda();initial=state_hash(model)
        previous=json.loads((BASE/'configs/prior-pilot-evidence.json').read_text())[args.arm]
        if initial!=previous['manifest']['initialization_sha256']:raise ValueError('STOP: fresh initialization differs')
        if args.action=='benchmark':benchmark(model,train,val,diag,p,args.output);return
        identity=json.loads((args.output/'run-manifest.json').read_text())
        if identity['initialization_sha256']!=initial:raise ValueError('STOP: initial identity differs')
        if args.action=='verify':verify(args.output,model,train,val,diag,p,identity);return
        result=fit(model,train,val,diag,p,args.output/'training',identity,time.monotonic()+args.remaining-5)
        if result['status']!='COMPLETE':raise TimeoutError('TIME_BUDGET_EXCEEDED: valid prefix retained')
        saved=load_checkpoint(args.output/'training/checkpoint.pt')
        sequence=[r['input_sha256'] for r in saved['history']]
        from data import digest
        sampler=Sampler(train,p['seeds']['sampler'],1);expected=[];ledger={'committed_updates':0}
        for _ in range(p['training']['updates']):
            expected.append(digest([sampler.propose()[0]['input_ids']]))
            sampler.resolve(True,ledger,{})
        if sequence!=expected:raise ValueError('P2 training order differs')
        checkpoints=args.output/'checkpoints';checkpoints.mkdir(exist_ok=True)
        #Hard links preserve one immutable final payload without duplicating disk allocation.
        for name in ('latest.pt','final.pt'):
            target=checkpoints/name
            if not target.exists():os.link(args.output/'training/checkpoint.pt',target)
        write(args.output/'initialization.json',{'state_sha256':initial,'old_weights_loaded':False,
            'exclusive_e7_trainable_parameters':0,'exclusive_e7_initialization':'NOT_APPLICABLE: Store and EMA are parameter-free'})
        write(args.output/'environment.json',environment());write(args.output/'parameters.json',parameter_report(model))
        write(args.output/'result.json',result)
        record=json.loads((args.output/'training/metrics.json').read_text())
        write(args.output/'metrics/training.json',record['history'])
        write(args.output/'metrics/validation.json',result['curve'])
        all_points=[{'update':0,'training':result['initial_evaluation']['training_diagnostic'],'validation':result['initial_evaluation']}]+result['curve']
        distribution={str(point['update']):{kind:{k:point[kind][k] for k in ('prediction_counts','top20_predicted_tokens',
            'prediction_entropy','dominant_token_rate','unique_predicted_tokens','gold_rank_histogram')} for kind in ('training','validation')} for point in all_points}
        write(args.output/'metrics/prediction-distribution.json',distribution)
        memory={str(path.relative_to(args.output)):json.loads(path.read_text()) for path in (args.output/'training/evaluation').glob('*/*/memory-diagnostics.json')}
        write(args.output/'metrics/memory-diagnostics.json',memory)
        logs=args.output/'logs';logs.mkdir(exist_ok=True)
        (logs/'training.log').write_text((args.output/'training/updates.jsonl').read_text())
        (logs/'validation.log').write_text(json.dumps(result['curve']))
    finally:
        watchdog.terminate();watchdog.wait(timeout=5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['benchmark','train','verify'],required=True)
    p.add_argument('--arm',required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--remaining',type=float,required=True)
    launch(p.parse_args())
