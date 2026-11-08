"""M2 thin training adapter: reuse M1 loss/sampler/atomic checkpoint primitives."""
from pathlib import Path
import time
import hashlib
import torch
from e0 import (Sampler,masked_loss,evaluate,parameters,write,save_checkpoint,
                load_checkpoint,rng,restore_rng)
from data import batch,digest


def fit(model,rows,validation,config,out,identity,deadline=float('inf'),stop_after=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'checkpoint.pt'
    device=next(model.parameters()).device
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=(.9,.95),weight_decay=.1)
    sampler=Sampler(rows,config['seed'],config['batch'])
    ledger={k:0 for k in ('input_tokens','processed_tensor_tokens','padding_tokens',
                         'supervised_target_tokens','query_target_tokens','committed_updates')}
    history=[];initial=None;last_saved=-1;last_save_time=time.monotonic()
    saves={'writes_this_process':0,'write_seconds_this_process':0.,'policy':config['checkpoint']}
    def save():
        nonlocal last_saved,last_save_time
        started=time.perf_counter()
        save_checkpoint({'format':'minicells.epclm004.m2-checkpoint.v1','identity':identity,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng(device.type=='cuda'),
            'sampler':sampler.snapshot(),'ledger':ledger,'history':history,'initial_evaluation':initial},path)
        saves['writes_this_process']+=1;saves['write_seconds_this_process']+=time.perf_counter()-started
        last_saved=ledger['committed_updates'];last_save_time=time.monotonic()
    def restore():
        nonlocal ledger,history,initial,last_saved
        saved=load_checkpoint(path)
        if saved['format']!='minicells.epclm004.m2-checkpoint.v1' or saved['identity']!=identity:
            raise ValueError('checkpoint source/config/architecture/data/tokenizer mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        sampler.pending=None;sampler.restore(saved['sampler']);restore_rng(saved['rng'])
        ledger=saved['ledger'];history=saved['history'];initial=saved['initial_evaluation']
        last_saved=ledger['committed_updates']
    def result(status,error=None,final=None):
        record={'status':status,'error':error,'ledger':ledger,'initial_evaluation':initial,'validation':final,
                'history':history,'parameters':parameters(model),'checkpoint_policy':saves,
                'checkpoint':{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
                              'durable_updates':last_saved},'hypotheses':'NOT_EVALUATED'}
        write(out/'metrics.json',record)
        (out/'metrics.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in history))
        return record
    if path.exists():restore()
    else:initial=evaluate(model,validation,config['batch']);save()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    limit=min(config['updates'],stop_after) if stop_after is not None else config['updates']
    while ledger['committed_updates']<limit:
        if time.monotonic()>=deadline:break
        before_rng=rng(device.type=='cuda');stepped=False
        try:
            selected=sampler.propose();b=batch(selected,device)
            model.train();optimizer.zero_grad(set_to_none=True)
            if device.type=='cuda':torch.cuda.synchronize(device)
            started=time.perf_counter();logits=model(b.input_ids);loss=masked_loss(logits,b.labels)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();grads=[p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None for g in grads) or not torch.stack([g.isfinite().all() for g in grads]).all():
                raise FloatingPointError('nonfinite/missing gradient; no clipping or update')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            before=model.token_embedding.weight.detach().clone() if ledger['committed_updates']<2 else None
            stepped=True;optimizer.step()
            if not torch.stack([p.isfinite().all() for p in model.parameters()]).all():
                raise FloatingPointError('nonfinite updated parameters')
            changed=float((model.token_embedding.weight.detach()-before).norm()) if before is not None else None
            if before is not None and not changed>0:raise FloatingPointError('no real parameter update')
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.perf_counter()-started
            count=int((b.labels!=-100).sum());queries=int(b.query_mask.sum())
            sampler.resolve(True,ledger,{'input_tokens':b.input_tokens,'processed_tensor_tokens':b.input_ids.numel(),
                'padding_tokens':b.input_ids.numel()-b.input_tokens,'supervised_target_tokens':count,'query_target_tokens':queries})
            history.append({'update':ledger['committed_updates'],'loss':float(loss.detach()),'gradient_norm':float(norm),
                'parameter_change_norm':changed,'compute_seconds':seconds,'input_sha256':digest(b.input_ids.cpu().tolist()),
                'supervised_targets':count,'query_targets':queries})
            policy=config['checkpoint']
            if ledger['committed_updates']%policy['every_updates']==0 or time.monotonic()-last_save_time>=policy['every_seconds']:
                save()
        except BaseException as error:
            if stepped:
                # A partially mutated optimizer is unsafe. Discard all updates
                # since the last durable boundary, including their ledger rows.
                restore()
                rollback='LAST_DURABLE_CHECKPOINT'
            else:
                if sampler.pending is not None:sampler.resolve(False,ledger)
                restore_rng(before_rng);optimizer.zero_grad(set_to_none=True)
                # Earlier successful updates remain valid: flush them on exit.
                try:
                    if last_saved!=ledger['committed_updates']:save()
                except BaseException:restore()
                rollback='PRESTEP_REJECT_AND_FLUSH_VALID_PREFIX'
            result('INCOMPLETE',{'type':type(error).__name__,'message':str(error),'rollback':rollback})
            raise
    # Final/timeout/intentional-stop boundary: always persist the latest valid
    # prefix, without rewriting an identical just-saved full checkpoint.
    try:
        if last_saved!=ledger['committed_updates']:save()
    except BaseException as error:
        restore();result('INCOMPLETE',{'type':type(error).__name__,'message':str(error),'rollback':'LAST_DURABLE_CHECKPOINT'})
        raise
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    final=evaluate(model,validation,config['batch'])
    record=result('COMPLETE' if ledger['committed_updates']==config['updates'] else 'INCOMPLETE',final=final)
    record['training_peak_allocated_bytes']=peak
    write(out/'metrics.json',record)
    return record
