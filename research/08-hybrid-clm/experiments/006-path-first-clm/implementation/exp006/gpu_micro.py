"""Bounded engineering-only GPU execution, with synthetic integer input and no optimizer."""
from __future__ import annotations
import copy,json,platform,time
from pathlib import Path
import torch
from .baseline import make_model
from .training import objective
from .profiling import parity,benchmark
from .accounting import ComputeCounter,summarize
from .reuse import write_json,canonical_sha256


def save_evidence(output,result):
    path=Path(output)
    if path.exists():raise FileExistsError("micro evidence is immutable")
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,result)


def run(c,output,device='cuda:0',source=None):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('GPU micro-preflight requires available CUDA; CPU is not a GPU pass')
    if c['model']['width']>512 or c['model']['cell_count']>32 or c['model']['steps']>6:
        raise ValueError('micro-preflight is bounded to the existing candidate scales')
    if output and Path(output).exists():raise FileExistsError('micro evidence is immutable')
    micro=copy.deepcopy(c);micro['evaluation'].update(decode_prefix=4,decode_tokens=3,decode_repetitions=3)
    started=time.perf_counter();rows={}
    for arm in ('R0','R1','R2','M0'):
        print('GPU micro-preflight '+arm+': one forward/backward, no optimizer',flush=True)
        model=make_model(c['model'],arm,606).to(device)
        x=torch.arange(16,device=device).reshape(2,8)%c['model']['vocab_size'];y=(x+1)%c['model']['vocab_size']
        with ComputeCounter(model) as count:
            out=model(x,policy='sampled',generator=torch.Generator(device=device).manual_seed(606))
        loss,parts=objective(out,y,c['training']);loss.backward();torch.cuda.synchronize()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError('nonfinite GPU micro gradients')
        training_compute=summarize(model,out,count)
        model.zero_grad(set_to_none=True);model.eval()
        check=parity(model,x,policy='sampled',generator=torch.Generator(device=device).manual_seed(607))
        if not check['passed']:raise AssertionError('GPU sampled-replay decode parity failed')
        timing=benchmark(model,x[:1],micro)
        rows[arm]={'finite_forward_backward':True,'objective_parts':parts,'decode_parity':check,
                   'training_compute':training_compute,'decode_micro_profile':timing}
        print(json.dumps({'arm':arm,'micro_p50_ms':timing['warm_p50_ms'],'parity':check['passed']}),flush=True)
        del model,out,loss,x,y;torch.cuda.empty_cache()
    result={'status':'GPU_MICRO_ENGINEERING_PASSED','scientific_decision':False,'source_commit':source,
            'config_sha256':canonical_sha256(c),'corpus_read':False,'confirmation_read':False,'optimizer_steps':0,
            'fixture_seed':606,'shape':[2,8],'timing_repetitions':3,'elapsed_seconds':time.perf_counter()-started,
            'gpu':torch.cuda.get_device_name(device),'cuda_capability':list(torch.cuda.get_device_capability(device)),
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda},'arms':rows,
            'scope':'untrained synthetic engineering workload; local device/shared desktop environment; not T4 validation, learning evidence, sparse-budget attainment or production latency'}
    if output:save_evidence(output,result)
    return result
