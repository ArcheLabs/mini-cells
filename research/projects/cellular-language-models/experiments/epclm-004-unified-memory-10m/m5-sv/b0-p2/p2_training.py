"""Private M2 instrumentation; preserve optimizer, loss, sampler and rollback math."""
import csv
import json
import time
from pathlib import Path
import torch
from p2_dataset import BASE
from p2_metrics import evaluate
from b0_training import MutationAdamW,private_trainer
from b0_models import parameter_report
from e0 import load_checkpoint,save_checkpoint,write
from data import file_hash


class ObservedAdamW(MutationAdamW):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.p2_observations={}

    def step(self,closure=None):
        value=super().step(closure);step=max(int(s['step']) for s in self.state.values())
        params=[p for group in self.param_groups for p in group['params']]
        self.p2_observations[step]={'parameter_norm':float(torch.stack([p.detach().float().square().sum() for p in params]).sum().sqrt()),
            'learning_rate':[group['lr'] for group in self.param_groups],
            'gpu_allocated_bytes':torch.cuda.memory_allocated() if params[0].is_cuda else None,
            'gpu_reserved_bytes':torch.cuda.memory_reserved() if params[0].is_cuda else None}
        return value

    def state_dict(self):
        state=super().state_dict();state['p2_observations']=self.p2_observations.copy();return state

    def load_state_dict(self,state):
        self.p2_observations={int(k):v for k,v in state.get('p2_observations',{}).items()};super().load_state_dict(state)


def pure_checkpoint(state,path):
    #Drop PyTorch _metadata carrying the inherited bound _version method.
    #Tensor values and aliases are unchanged; historical checkpoint files are untouched.
    state={**state,'model':dict(state['model']),
        'p2_epoch_cursor':{'completed_epochs':state['ledger']['committed_updates']//len(state['sampler']['order']),'sampler_cursor':state['sampler']['cursor']}}
    save_checkpoint(state,path)


def boundaries(updates):return sorted({n for n in (256,512,768,1024,updates) if 0<n<=updates})


def fit(model,train,validation,diagnostic,protocol,folder,identity,deadline=float('inf'),stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if any(r['split']=='test' for bank in (train,validation,diagnostic) for r in bank):raise PermissionError('Test is forbidden')
    if identity.get('phase')!='b0-p2':raise ValueError('P2 identity required')
    curve=json.loads((folder/'curve.json').read_text()) if (folder/'curve.json').exists() else []
    initial_pending=not (folder/'checkpoint.pt').exists()
    targets=boundaries(protocol['training']['updates'])
    if stop_after is not None:targets=[n for n in targets if n<=stop_after]
    if not targets:raise ValueError('No requested P2 boundary')
    for target in targets:
        if any(r['update']==target for r in curve):continue
        existing=load_checkpoint(folder/'checkpoint.pt') if (folder/'checkpoint.pt').exists() else None
        if existing and (existing['identity']!=identity or existing['ledger']['committed_updates']>target):raise ValueError('Checkpoint identity/curve mismatch')
        module=private_trainer(model,folder,protocol);module.torch.optim.AdamW=ObservedAdamW;module.save_checkpoint=pure_checkpoint
        def measurement(model,rows,batch_size):
            nonlocal initial_pending
            step=0 if initial_pending else target;initial_pending=False
            val=evaluate(model,validation,folder/'evaluation'/str(step)/'validation',draws=protocol['metrics']['bootstrap_draws'])
            train_metrics=evaluate(model,diagnostic,folder/'evaluation'/str(step)/'training',draws=protocol['metrics']['bootstrap_draws'])
            return {**val,'training_diagnostic':train_metrics,'evaluation_update':step}
        module.evaluate=measurement
        began=time.perf_counter();record=module.fit(model,train,validation,
            {**protocol['training'],'seed':protocol['seeds']['sampler']},folder,identity,deadline,stop_after=target)
        saved=load_checkpoint(folder/'checkpoint.pt')
        for row in record['history']:
            step=row['update'];row.update(saved['optimizer']['p2_observations'][step],
                parameter_change_norm=saved['optimizer']['b0_parameter_changes'][step],
                epoch=(step-1)//len(train)+1,epoch_cursor=(step-1)%len(train)+1,
                input_tokens_seen=step*protocol['data']['length'],targets_seen=step*protocol['data']['queries'])
        write(folder/'metrics.json',record)
        (folder/'updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in record['history']))
        resources=json.loads((folder/'resources.json').read_text()) if (folder/'resources.json').exists() else {'fit_wall_seconds':0.}
        peak=record['training_peak_allocated_bytes']
        resources.update(fit_wall_seconds=resources['fit_wall_seconds']+time.perf_counter()-began,
            training_peak_allocated_bytes=max(resources.get('training_peak_allocated_bytes',0) or 0,peak or 0),
            checkpoint_write_policy=record['checkpoint_policy'],checkpoint=record['checkpoint'],
            checkpoint_writes_total=resources.get('checkpoint_writes_total',0)+record['checkpoint_policy']['writes_this_process'],
            checkpoint_write_seconds_total=resources.get('checkpoint_write_seconds_total',0.)+record['checkpoint_policy']['write_seconds_this_process'],
            checkpoint_timing_scope='Completed fit calls; interrupted-call timing may be unavailable; cumulative phase wall remains charged',
            compute_tokens_per_second=record['ledger']['input_tokens']/sum(r['compute_seconds'] for r in record['history']),
            observer_cost_included=True,**parameter_report(model))
        write(folder/'resources.json',resources)
        if record['ledger']['committed_updates']!=target:return {'status':'INCOMPLETE','ledger':record['ledger']}
        point={'update':target,'validation':record['validation'],
            'training':record['validation']['training_diagnostic'],'checkpoint_sha256':file_hash(folder/'checkpoint.pt')}
        curve.append(point);write(folder/'curve.json',curve)
    saved=load_checkpoint(folder/'checkpoint.pt')
    if saved['identity']!=identity:raise ValueError('P2 final identity differs')
    model.load_state_dict(saved['model'])
    for kind in ('training','validation'):
        with (folder/(kind+'-curves.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['update','exact_match','nll','type_nll','conditional_nll'])
            first=saved['initial_evaluation'];initial=first if kind=='validation' else first['training_diagnostic']
            for step,r in [(0,initial)]+[(p['update'],p[kind]) for p in curve]:
                writer.writerow([step,r['full_vocab_exact_match'],r['nll'],r['type_nll'],r['value_conditional_nll']])
    return {'status':'COMPLETE' if saved['ledger']['committed_updates']==protocol['training']['updates'] else 'INCOMPLETE',
        'ledger':saved['ledger'],'curve':curve,'initial_evaluation':saved['initial_evaluation'],
        'checkpoint_sha256':file_hash(folder/'checkpoint.pt')}
