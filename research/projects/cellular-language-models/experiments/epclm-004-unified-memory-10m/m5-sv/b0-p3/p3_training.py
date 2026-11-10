"""P3-specific logical batches and curriculum over existing atomic primitives."""
import json
import os
import sys
import time
from pathlib import Path
import torch
from p3_data import BASE,forward_batch,StageSampler,digest,file_hash,write
from p3_metrics import candidate_loss,statistics,evaluate
sys.path.insert(0,str(BASE.parent/'b0-p2'))
from p2_training import ObservedAdamW
from e0 import save_checkpoint,load_checkpoint,rng,restore_rng
from b0_models import state_hash


def epoch_schedule(p,stage):return p['stages'][str(stage)]['validation_updates']


def train_stage(model,train,val,p,folder,identity,stage,*,continuation=None,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'checkpoint.pt'
    level=p['stages'][str(stage)]['level'];limit=p['stages'][str(stage)]['updates']
    device=next(model.parameters()).device
    optimizer=ObservedAdamW(model.parameters(),lr=p['training']['lr'],betas=tuple(p['training']['betas']),weight_decay=p['training']['weight_decay'])
    sampler=StageSampler(train,p['master_seed'],level)
    names=('input_tokens','valid_input_tokens','padding_tokens','processed_tensor_tokens','supervised_target_tokens','query_target_tokens','episodes','committed_updates')
    ledger={k:0 for k in names};history=[];online={};step=0;start={k:0 for k in names}
    def restore(saved,enter=False):
        nonlocal ledger,history,online,step,start
        if saved['identity']!=identity:raise ValueError('Source/config/data/model identity differs')
        if saved['stage']!=stage and not (enter and stage==2 and saved['stage']==1):raise ValueError('Invalid curriculum checkpoint')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        ledger=saved['ledger'];history=saved['history'];online=saved['online']
        if enter:step=0;start=ledger.copy()
        else:step=saved['stage_step'];start=saved['stage_start_ledger'];sampler.restore(saved['sampler'])
    def save():
        save_checkpoint({'format':'minicells.b0-p3.checkpoint.v1','identity':identity,'stage':stage,'stage_step':step,
            'global_step':ledger['committed_updates'],'model':dict(model.state_dict()),'optimizer':optimizer.state_dict(),
            'rng':rng(device.type=='cuda'),'sampler':sampler.snapshot(),'stage_start_ledger':start,
            'ledger':ledger,'history':history,'online':online,'scheduler_state':None},path)
    if path.exists():restore(load_checkpoint(path))
    else:
        if stage==2:
            if continuation is None:raise ValueError('Stage2 requires its own Stage1 final checkpoint')
            restore(load_checkpoint(continuation),enter=True)
        online.setdefault(str(stage),{'correct':0,'queries':0,'numerically_finite':True});save()
    def validation():
        target=folder/'evaluation'/str(step);existing=target/'evaluation-identity.json'
        stamp={'model_sha256':state_hash(model),'validation_sha256':digest(val),'identity_sha256':digest(identity)}
        if existing.exists():
            if json.loads(existing.read_text())!=stamp:raise ValueError('Saved evaluation model/data differs')
            proof=json.loads((target/'probability-manifest.json').read_text())
            if file_hash(target/'probabilities.npz')!=proof['sha256']:raise ValueError('Saved probability array changed')
            return json.loads((target/'metrics.json').read_text())
        r=evaluate(model,val,target,full_probabilities=step==limit,draws=p['gate']['bootstrap_draws'])
        write(existing,stamp);return r
    if step==limit and (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text())
        if result['status']=='COMPLETE' and result['checkpoint_sha256']==file_hash(path):
            validation();return result
    if step==0:validation()
    target=limit if stop_after is None else min(limit,stop_after)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    began=time.perf_counter()
    def log_event(event):
        with (folder/'training.log').open('a') as stream:
            stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
    if step<target:log_event({'event':'START_OR_RESUME','stage':stage,'durable_global_step':ledger['committed_updates'],'checkpoint_sha256':file_hash(path)})
    while step<target:
        try:
            rows=sampler.propose();total=sum(len(r['query_positions']) for r in rows)
            if len(rows)!=8 or any(r['split']!='train' for r in rows):raise ValueError('Logical batch must have8 independent Training Episodes')
            model.train();optimizer.zero_grad(set_to_none=True);reports=[]
            if device.type=='cuda':torch.cuda.synchronize(device)
            tick=time.perf_counter()
            logits=forward_batch(model,rows)
            loss=torch.stack([candidate_loss(logits[i],row) for i,row in enumerate(rows)]).sum()/total
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite candidate loss')
            for i,row in enumerate(rows):
                report,_=statistics(logits[i,row['query_positions']],row);reports.append(report)
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None for g in gradients) or not all(bool(g.isfinite().all()) for g in gradients):raise FloatingPointError('Missing/nonfinite gradient')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['training']['clip'],error_if_nonfinite=True)
            nontrivial=any(r['candidate_count']>1 for r in reports)
            if nontrivial and float(norm)==0 and float(loss.detach())>1e-7:raise FloatingPointError('Zero gradient despite nonzero candidate loss')
            optimizer.step()
            if not all(bool(param.isfinite().all()) for param in model.parameters()):raise FloatingPointError('Nonfinite updated parameters')
            if device.type=='cuda':torch.cuda.synchronize(device)
            accounts={'input_tokens':sum(r['storage_length'] for r in rows),'valid_input_tokens':sum(r['valid_length'] for r in rows),
                'padding_tokens':sum(r['storage_length']-r['valid_length'] for r in rows),'processed_tensor_tokens':sum(r['valid_length'] for r in rows),
                'supervised_target_tokens':total,'query_target_tokens':total,'episodes':8}
            sampler.resolve(True,ledger,accounts);step+=1;global_step=ledger['committed_updates']
            correct=sum(r['active_correct'] for r in reports);online[str(stage)]['correct']+=correct;online[str(stage)]['queries']+=total
            history.append({'global_step':global_step,'stage':stage,'stage_step':step,'epoch':sampler.epoch+1,
                'active_nll':sum(r['active_nll']*r['queries'] for r in reports)/total,'active_exact_match':correct/total,
                'random_accuracy':sum(r['random_active_accuracy']*r['queries'] for r in reports)/total,
                'random_nll':sum(r['random_active_nll']*r['queries'] for r in reports)/total,
                'gold_rank':sum(sum(r['active_gold_rank']) for r in reports)/total,
                'gold_probability':sum(r['active_gold_probability']*r['queries'] for r in reports)/total,
                'value_mass':sum(r['value_mass']*r['queries'] for r in reports)/total,
                'candidate_mass':sum(r['candidate_mass']*r['queries'] for r in reports)/total,
                'gradient_norm':float(norm),'zero_gradient':float(norm)==0,'parameter_change_norm':optimizer.b0_mutations[global_step],
                'compute_seconds':time.perf_counter()-tick,'input_tokens_seen':ledger['input_tokens'],
                'valid_input_tokens_seen':ledger['valid_input_tokens'],'supervised_targets_seen':ledger['supervised_target_tokens'],
                'batch_mapping_hashes':[r['mapping_hash'] for r in rows],'batch_episode_ids':[r['id'] for r in rows],
                **optimizer.p2_observations[global_step]})
            log_event({'event':'UPDATE_COMMITTED_IN_MEMORY',**history[-1]})
            if step%64==0 or step in epoch_schedule(p,stage) or step==target:save()
            if step in epoch_schedule(p,stage):validation()
        except BaseException:
            #Restore Model/Optimizer/RNG/Sampler/Ledger together; no partial update survives.
            log_event({'event':'ROLLBACK','in_memory_global_step':ledger['committed_updates'],'checkpoint_sha256':file_hash(path)})
            restore(load_checkpoint(path));optimizer.zero_grad(set_to_none=True);raise
    saved=load_checkpoint(path)
    if saved['stage_step']!=step:save();saved=load_checkpoint(path)
    final=validation() if step==limit else None
    (folder/'committed-updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    result={'status':'COMPLETE' if step==limit else 'PARTIAL','stage':stage,'stage_step':step,'global_step':ledger['committed_updates'],
        'ledger':ledger,'stage_ledger':{k:ledger[k]-start[k] for k in names},'online':online[str(stage)],
        'validation':final,'checkpoint_sha256':file_hash(path),
        'resources':{'fit_wall_seconds_this_call':time.perf_counter()-began,'training_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}}
    write(folder/'result.json',result);return result
