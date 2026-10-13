"""Gate 0 engineering only, no corpus access and no language scientific claim."""
from __future__ import annotations
import copy,json,tempfile
from pathlib import Path
import numpy as np
import torch
from .baseline import make_model
from .training import objective
from .profiling import parity
from .accounting import ComputeCounter,summarize
from .checkpoint import save,restore

def tiny_config(c):
    c=copy.deepcopy(c);c['model'].update(width=16,heads=2,cell_count=3,cell_hidden=24,vocab_size=32,max_sequence_length=16,steps=2,router_hidden=8,step_width=4,edge_rank=2,moe_layers=2,moe_experts=2,moe_hidden=24)
    c['training'].update(batch_size=2,sequence_length=8,steps=2,tokens=32,checkpoint_every=1,evaluate_every=1);c['evaluation'].update(windows=2,bootstrap_replicates=100,decode_prefix=4,decode_tokens=3,decode_repetitions=2);return c

def run(c):
    torch.set_num_threads(1);small=tiny_config(c);reports={};parameters={}
    for arm in ('R0','R1','R2','M0'):
        full=make_model(c['model'],arm,606);parameters[arm]=full.parameter_breakdown();del full
        model=make_model(small['model'],arm,606);x=torch.arange(16).remainder(32).reshape(2,8);y=(x+1)%32
        with ComputeCounter(model) as counter:out=model(x,policy='sampled',generator=torch.Generator().manual_seed(606))
        loss,parts=objective(out,y,small['training']);loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise AssertionError('nonfinite Gate 0')
        if arm!='M0' and not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.controller.parameters()):raise AssertionError('no policy gradient')
        cpu_compute=summarize(model,out,counter);replayed=parity(model,x,policy='sampled',generator=torch.Generator().manual_seed(608))
        if not replayed['passed']:raise AssertionError('decode mismatch')
        future=x.clone();future[:,5:]=(future[:,5:]+7)%32
        a=model(x);b=model(future)
        if not torch.allclose(a.logits[:,:5],b.logits[:,:5],atol=1e-5,rtol=1e-4):raise AssertionError('future leakage')
        reports[arm]={'finite_forward_backward':True,'causal':True,'decode':replayed,'compute':cpu_compute}
    values=[parameters[a]['total'] for a in parameters];ratio=max(values)/min(values)
    if ratio>1.01:raise AssertionError('capacity difference exceeds 1%')
    return {'status':'ENGINEERING_READY','scientific_decision':False,'gate':'CPU_GATE_0','parameter_counts':parameters,'capacity_max_min_ratio':ratio,'arms':reports,'scope':'unit-sized engineering validation, no corpus, no GPU; full acceptance additionally requires pytest suite'}
