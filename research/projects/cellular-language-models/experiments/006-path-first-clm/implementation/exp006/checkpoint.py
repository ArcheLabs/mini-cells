"""Existing atomic checkpoint implementation plus explicit path-sampler state."""
from __future__ import annotations
import re
import numpy as np
from .reuse import save_checkpoint,restore_checkpoint,file_sha256,canonical_sha256

def identity(source,c,p,assets,arm,seed,role):
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('full source SHA required')
    if arm not in ('R0','R1','R2','M0') or role not in ('development','confirmation'):raise ValueError('invalid arm/role')
    if not re.fullmatch('[0-9a-f]{64}',assets):raise ValueError('full data manifest hash required')
    return {'experiment_id':'006-path-first-clm','source_commit':source,'config_sha256':canonical_sha256(c),'protocol_sha256':canonical_sha256(p),'data_sha256':assets,'arm':arm,'seed':seed,'role':role}

def save(path,model,optimizer,step,run_id,data_rng,path_rng,extra=None):
    save_checkpoint(path,model=model,optimizer=optimizer,step=step,scaler=None,identity=run_id,generator=data_rng,
                    extra={**(extra or {}),'path_rng':path_rng.get_state(),'numpy_global_rng':np.random.get_state()})

def restore(path,model,optimizer,run_id,data_rng,path_rng):
    step,extra=restore_checkpoint(path,model=model,optimizer=optimizer,scaler=None,expected_identity=run_id,generator=data_rng)
    path_rng.set_state(extra.pop('path_rng').cpu());np.random.set_state(extra.pop('numpy_global_rng'));return step,extra
