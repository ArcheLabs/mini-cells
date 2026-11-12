"""Audit unchanged main dependencies, canonical counts and actual Chunk behavior."""
from pathlib import Path
import json
import torch
from b0_models import construct,parameter_report,state_hash,BMojo,EXPERIMENT
from data import file_hash,episode,MQARConfig
from e0 import ROOT

BASE=Path(__file__).resolve().parents[1]


def verify_dependencies():
    lock=json.loads((BASE/'configs/dependency-lock.json').read_text())
    for name,expected in lock['files'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('frozen main dependency differs: '+name)
    return {'status':'PASS','baseline_main':lock['baseline_main'],'files_verified':len(lock['files'])}


def parameter_audit(protocol):
    records={};common=None
    for arm in protocol['arms']:
        model=construct(arm,protocol);report=parameter_report(model)
        report['initial_state_sha256']=state_hash(model)
        report['forward_owner']=type(model).__module__+'.'+type(model).__name__+'.forward'
        records[arm]=report
        if arm!='E0':
            if common is None:common=report['initial_state_sha256']
            elif common!=report['initial_state_sha256']:raise ValueError('E6/E7 initialization mismatch')
    baseline=records['E0']['static_parameters']
    for r in records.values():
        r['relative_parameter_difference_vs_e0']=r['static_parameters']/baseline-1
        r['difference_exceeds_five_percent']=abs(r['relative_parameter_difference_vs_e0'])>.05
    return records


@torch.no_grad()
def chunk_audit(model, ids, tolerance=3e-5):
    model.eval();full=model(ids);state=None;parts=[]
    for block in ids.split(model.chunk,dim=1):
        value,state,_=model.forward_state(block,state,trace=False);parts.append(value)
    aligned=float((full-torch.cat(parts,1)).abs().max())
    model.reset_runtime();first,state,_=model.forward_state(ids[:,:-3],trace=False)
    last,_,_=model.forward_state(ids[:,-3:],state,trace=False)
    nonaligned=float((full-torch.cat((first,last),1)).abs().max())
    return {'aligned_chunk_size':model.chunk,'aligned_max_abs_error':aligned,'aligned_tolerance':tolerance,
        'aligned_status':'PASS' if aligned<=tolerance else 'FAIL_REPORTED_NO_ALGORITHM_PATCH',
        'nonaligned_split':[ids.shape[1]-3,3],'nonaligned_max_abs_error':nonaligned,
        'primary_path':'one complete forward call in both training and evaluation',
        'nonaligned_used_for_learnability':False,'state_summary':'preceding-chunk actual SSM state only',
        'gradient_scope':'full Episode graph; EMA/selection detached as frozen; final runtime clone is observational',
        'write_order':'all readers complete before the current chunk is admitted'}
