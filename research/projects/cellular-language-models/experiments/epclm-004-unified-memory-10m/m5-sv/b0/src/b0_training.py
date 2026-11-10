"""Thin private M2 adapter: preserve updates/rollback, enrich observation only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import math
import time
import torch
from b0_models import EXPERIMENT,parameter_report
from b0_metrics import evaluate
from data import digest,file_hash
from e0 import write,load_checkpoint


class MutationAdamW(torch.optim.AdamW):
    """Standard AdamW with persisted full-parameter mutation observations."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.b0_mutations={}

    def step(self,closure=None):
        params=[p for g in self.param_groups for p in g['params']]
        before=[p.detach().clone() for p in params]
        value=super().step(closure)
        change=float(torch.stack([(p.detach()-old).float().square().sum() for p,old in zip(params,before)]).sum().sqrt())
        if not math.isfinite(change) or change<=0:raise FloatingPointError('no finite real parameter update')
        step=max(int(s['step']) for s in self.state.values())
        self.b0_mutations[step]=change
        return value

    def state_dict(self):
        state=super().state_dict();state['b0_parameter_changes']=self.b0_mutations.copy();return state

    def load_state_dict(self,state):
        self.b0_mutations={int(k):float(v) for k,v in state.get('b0_parameter_changes',{}).items()}
        super().load_state_dict(state)


def private_trainer(model, folder, protocol):
    spec=importlib.util.spec_from_file_location('_b0_private_m2_training',EXPERIMENT/'m2/training.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Only this private module receives the optimizer observer; global Torch and
    # the original M2 module remain unchanged. The optimizer math is inherited.
    module.torch=SimpleNamespace(optim=SimpleNamespace(AdamW=MutationAdamW),cuda=torch.cuda,nn=torch.nn,
                                 isfinite=torch.isfinite,stack=torch.stack)
    module.parameters=parameter_report
    initial_pending=not (folder/'checkpoint.pt').exists()
    def measured_evaluation(model,rows,batch_size):
        nonlocal initial_pending
        name='initial.npz' if initial_pending else 'latest.npz';initial_pending=False
        return evaluate(model,rows,batch_size,save_probabilities=folder/'probabilities'/name,
            bootstrap_seed=protocol['seeds']['statistics'],bootstrap_draws=protocol['metrics']['bootstrap_draws'])
    module.evaluate=measured_evaluation
    return module


def fit_stage(model,rows,validation,protocol,folder,identity,stage,deadline=float('inf'),stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if stage not in ('overfit','pilot'):raise ValueError('unknown B0 stage')
    if any(r.get('split')=='test' for r in [*rows,*validation]):raise PermissionError('B0 must not consume Test')
    config={**protocol['training'],'seed':protocol['seeds']['sampler']}
    expected_identity={**identity,'stage':stage,'train_sha256':digest(rows),'validation_sha256':digest(validation),
        'optimizer_observation':'standard-AdamW-full-parameter-change-v1'}
    curve_path=folder/'curve.json';curve=json.loads(curve_path.read_text()) if curve_path.exists() else []
    boundaries=list(range(protocol['overfit']['evaluation_every'],config['updates']+1,protocol['overfit']['evaluation_every'])) if stage=='overfit' else protocol['pilot']['validation_updates']
    if stop_after is not None:boundaries=[n for n in boundaries if n<=stop_after]
    if not boundaries:raise ValueError('no approved validation boundary')
    for target in boundaries:
        if any(r['update']==target for r in curve):continue
        metrics_path=folder/'metrics.json'
        if metrics_path.exists() and json.loads(metrics_path.read_text())['ledger']['committed_updates']>target:
            raise ValueError('missing historical curve boundary; refusing to invent it')
        if time.monotonic()+180>=deadline:return {'status':'INCOMPLETE_RESOURCE_LIMIT','curve':curve}
        trainer=private_trainer(model,folder,protocol);began=time.perf_counter()
        record=trainer.fit(model,rows,validation,config,folder,expected_identity,deadline-120,stop_after=target)
        saved=load_checkpoint(folder/'checkpoint.pt')
        mutations=saved['optimizer']['b0_parameter_changes']
        for r in record['history']:r['parameter_change_norm']=mutations[r['update']]
        record['parameter_change_scope']='L2 over all unique optimizer parameters; observer GPU copies are included in peak memory/compute timing'
        actual=record['ledger']['committed_updates']
        artifact=record['validation'].get('full_query_probability_artifact')
        if artifact and Path(artifact['path']).name=='latest.npz':
            destination=folder/'probabilities'/f'update-{actual}.npz';Path(artifact['path']).replace(destination)
            artifact['path']=str(destination)
        resources_path=folder/'resources.json';resources=json.loads(resources_path.read_text()) if resources_path.exists() else {'fit_wall_seconds':0.,'training_peak_allocated_bytes':None}
        peak=record['training_peak_allocated_bytes'];prior=resources['training_peak_allocated_bytes']
        resources.update(fit_wall_seconds=resources['fit_wall_seconds']+time.perf_counter()-began,
            training_peak_allocated_bytes=max(peak or 0,prior or 0) if peak is not None else None,
            training_tokens_per_second=record['ledger']['input_tokens']/sum(r['compute_seconds'] for r in record['history']),
            checkpoint=record['checkpoint'],checkpoint_write=record['checkpoint_policy'],training_flops=None,
            flops_status='NOT_MEASURED_NO_VALIDATED_COMPLETE_COUNTER',**parameter_report(model))
        write(resources_path,resources);write(metrics_path,record)
        (folder/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in record['history']))
        if actual<target:return {'status':'INCOMPLETE_RESOURCE_LIMIT','curve':curve,'ledger':record['ledger']}
        curve.append({'update':actual,'ledger':record['ledger'],'evaluation':record['validation'],
            'checkpoint_sha256_at_boundary':record['checkpoint']['sha256']});write(curve_path,curve)
    saved=load_checkpoint(folder/'checkpoint.pt');record=json.loads((folder/'metrics.json').read_text())
    if saved['identity']!=expected_identity or saved['ledger']!=record['ledger'] or file_hash(folder/'checkpoint.pt')!=record['checkpoint']['sha256']:raise ValueError('B0 checkpoint identity/ledger/hash differs')
    model.load_state_dict(saved['model'])
    if stage=='overfit':
        best=streak=0
        for point in curve:
            streak=streak+1 if point['evaluation']['correct']>=protocol['overfit']['minimum_correct'] else 0;best=max(best,streak)
        status='OVERFIT_ESTABLISHED' if best>=protocol['overfit']['consecutive_evaluations'] else 'OVERFIT_NOT_ESTABLISHED'
    else:status='PILOT_COMPLETE_REQUIRES_RESEARCH_REVIEW'
    return {'status':status if saved['ledger']['committed_updates']==config['updates'] else 'INCOMPLETE_RESOURCE_LIMIT',
            'curve':curve,'ledger':saved['ledger'],'checkpoint_sha256':file_hash(folder/'checkpoint.pt')}
