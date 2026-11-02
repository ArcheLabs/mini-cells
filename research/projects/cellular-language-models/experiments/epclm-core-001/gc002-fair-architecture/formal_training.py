"""Frozen T1 full-sequence training path and atomic resume/accounting state."""
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from cached_transformer import CachedTransformer,Config
from exposure import ExposureLedger
from flops_audit import ArithmeticCounter
from native_clm_runtime import lr_multiplier
from training_path import FullSequenceSampler,teacher_forced_sequences
from trainer import save_checkpoint,load_checkpoint,rng,restore_rng


def _cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_cpu(v) for v in value)
    return value


def _equal(left,right):
    if torch.is_tensor(left):return torch.equal(left.cpu(),right.cpu())
    if isinstance(left,np.ndarray):return np.array_equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(_equal(a,b) for a,b in zip(left,right))
    return left==right


class FormalTrainer:
    def __init__(self,documents,identity,protocol,device='cuda',target_tokens=None):
        self.documents=documents;self.identity=identity;self.protocol=protocol;self.device=device
        self.config=Config(**{k:protocol['model'][k] for k in ('vocab','dim','layers','heads','kv_heads','ffn','window')})
        train=protocol['training'];opt=protocol['optimizer'];schedule=protocol['scheduler']
        self.seed=protocol['seed'];self.batch=train['batch_lanes'];self.chunk=train['fresh_prediction_targets_per_lane']
        self.target_tokens=train['prediction_token_budget'] if target_tokens is None else target_tokens
        self.budget_updates=math.ceil(self.target_tokens/(self.batch*self.chunk));self.total_updates=schedule['total_updates']
        if self.target_tokens==train['prediction_token_budget'] and self.budget_updates!=self.total_updates:raise ValueError('frozen scheduler update count differs from formal token budget')
        torch.manual_seed(self.seed)
        self.model=CachedTransformer(self.config).to(device=device,dtype=torch.float32).train()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=opt['learning_rate'],betas=tuple(opt['betas']),
            weight_decay=opt['weight_decay'],foreach=opt['foreach'])
        warmup=schedule['warmup_updates'];total=schedule['total_updates']
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:lr_multiplier(min(step+1,total),total,warmup))
        self.sampler=FullSequenceSampler(documents,self.seed,self.batch,self.chunk,train['context_reuse_per_sequence_max'])
        self.ledger=ExposureLedger(sum(len(row['tokens'])-1 for row in documents),identity)
        self.metrics=[];self.flops_updates=[];self.flops_aggregate={};self.resume_result={'status':'NOT_RUN'};self.poisoned=False;self.last_counter=None
        self.training_seconds=0.;self.checkpoint_seconds=0.

    def _flops_commit(self,counter,update,tokens):
        records=[counter.records[key] for key in sorted(counter.records)]
        digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        row={'update':update,'prediction_tokens':tokens,'phase_flops':dict(counter.totals),
            'registered_operator_records':len(records),'operator_records_sha256':digest,
            'unsupported_operators':dict(counter.unknown)}
        self.flops_updates.append(row)
        for record in records:
            key=(record['phase'],record['operator'],tuple(record['input_shapes']+['->']+record['output_shapes']),
                tuple(record['input_dtypes']+['->']+record['output_dtypes']))
            aggregate=self.flops_aggregate.setdefault(key,{**record,'execution_count':0,'estimated_flops':0,'logical_output_bytes':0})
            aggregate['execution_count']+=record['execution_count'];aggregate['estimated_flops']+=record['estimated_flops']
            aggregate['logical_output_bytes']+=record['memory_output_bytes']
        return row

    def update(self,profile=False):
        if self.poisoned:raise RuntimeError('trainer is poisoned after a failed optimizer transaction')
        proposal=self.sampler.propose(self.target_tokens-self.ledger.consumed)
        self.optimizer.zero_grad(set_to_none=True);counter=ArithmeticCounter();started=time.perf_counter()
        learning_rate_used=self.optimizer.param_groups[0]['lr']
        if self.device=='cuda':torch.cuda.synchronize()
        profiler=None
        activities=[torch.profiler.ProfilerActivity.CPU]
        if self.device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            if profile:
                profiler=torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_flops=True)
                profiler.__enter__()
            with sdpa_kernel(SDPBackend.MATH),counter:
                loss,accounting=teacher_forced_sequences(self.model,proposal['segments'])
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite FP32 loss')
                counter.phase='backward';loss.backward()
                gradients=[p.grad for p in self.model.parameters() if p.grad is not None]
                if not gradients or not bool(torch.stack([g.isfinite().all() for g in gradients]).all()):raise FloatingPointError('missing or nonfinite FP32 gradient')
                counter.phase='gradient_clipping'
                grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.protocol['optimizer']['gradient_clip_norm'],foreach=False,error_if_nonfinite=True)
                counter.phase='optimizer';self.optimizer.step()
            self.last_counter={'phase_flops':dict(counter.totals),'unsupported_operators':dict(counter.unknown),
                'operator_inventory':dict(counter.inventory),'operator_records':[counter.records[key] for key in sorted(counter.records)]}
            if counter.unknown:
                self.poisoned=True;raise ValueError('unsupported floating-point operators in live update: '+json.dumps(dict(counter.unknown),sort_keys=True))
            if not bool(torch.stack([p.isfinite().all() for p in self.model.parameters()]).all()):
                self.poisoned=True;raise FloatingPointError('nonfinite parameters after optimizer step; prior atomic checkpoint retained')
            tokens=accounting['prediction_tokens']
            if tokens!=len(proposal['events']):raise ValueError('exposure events differ from masked loss targets')
            self.scheduler.step();self.sampler.resolve(True,self.ledger)
            flop_row=self._flops_commit(counter,self.ledger.updates,tokens)
            if self.device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            row={'update':self.ledger.updates,'prediction_tokens':self.ledger.consumed,'update_prediction_tokens':tokens,
                'loss':float(loss.detach()),'gradient_norm_before_clip':float(grad_norm),'input_tokens':accounting['input_tokens'],
                'context_reuse_tokens':accounting['context_reuse_tokens'],'padding_tokens':accounting['padding_tokens'],
                'eos_targets':proposal['accounting']['eos_targets'],'seconds':elapsed,'tokens_per_second':tokens/max(elapsed,1e-12),
                'phase_flops':flop_row['phase_flops'],'registered_operator_records':flop_row['registered_operator_records'],
                'unsupported_operators':flop_row['unsupported_operators'],'data_order_hash':self.ledger.order_hash,
                'learning_rate_used':learning_rate_used}
            self.metrics.append(row);self.training_seconds+=elapsed
            if profiler is not None:profiler.__exit__(None,None,None)
            return row,profiler
        except BaseException as error:
            if self.last_counter is None or self.last_counter.get('phase_flops')!=dict(counter.totals):
                self.last_counter={'phase_flops':dict(counter.totals),'unsupported_operators':dict(counter.unknown),
                    'operator_inventory':dict(counter.inventory),'operator_records':[counter.records[key] for key in sorted(counter.records)]}
            if profiler is not None:
                try:profiler.__exit__(type(error),error,error.__traceback__)
                except Exception:pass
            if self.sampler.pending is not None:self.sampler.resolve(False,self.ledger)
            self.optimizer.zero_grad(set_to_none=True);self.poisoned=True
            raise

    def save(self,path):
        if self.poisoned:raise RuntimeError('refusing to checkpoint a failed optimizer transaction')
        started=time.perf_counter()
        value={'format':'gc002.t1-formal-state.v1','identity':self.identity,'config':self.config.__dict__,
            'protocol_sha256':self.identity['protocol'],'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),'sampler':self.sampler.snapshot(),'ledger':self.ledger.snapshot(),
            'flops_updates':self.flops_updates,'flops_aggregate':self.flops_aggregate,'metrics':self.metrics,
            'rng':rng(self.device=='cuda'),'resume_result':self.resume_result,
            'training_seconds':self.training_seconds,'checkpoint_seconds':self.checkpoint_seconds}
        save_checkpoint(value,path);self.checkpoint_seconds+=time.perf_counter()-started

    @classmethod
    def load(cls,path,documents,identity,protocol,device='cuda',target_tokens=None):
        saved=load_checkpoint(path)
        config={k:protocol['model'][k] for k in ('vocab','dim','layers','heads','kv_heads','ffn','window')}
        if saved.get('format')!='gc002.t1-formal-state.v1' or saved.get('identity')!=identity or saved.get('protocol_sha256')!=identity['protocol'] or saved.get('config')!=config:
            raise ValueError('checkpoint source/protocol/data/seed/model identity mismatch')
        result=cls(documents,identity,protocol,device,target_tokens)
        result.model.load_state_dict(saved['model'],strict=True);result.optimizer.load_state_dict(saved['optimizer']);result.scheduler.load_state_dict(saved['scheduler'])
        result.sampler.restore(saved['sampler']);result.ledger=ExposureLedger.restore(saved['ledger'],identity)
        result.flops_updates=saved['flops_updates'];result.flops_aggregate=saved['flops_aggregate'];result.metrics=saved['metrics'];result.resume_result=saved['resume_result']
        result.training_seconds=saved.get('training_seconds',sum(row['seconds'] for row in result.metrics))
        result.checkpoint_seconds=saved.get('checkpoint_seconds',0.)
        if len(result.metrics)!=result.ledger.updates or len(result.flops_updates)!=result.ledger.updates or sum(r['prediction_tokens'] for r in result.metrics)!=result.ledger.consumed:
            raise ValueError('checkpoint metric, FLOPs and exposure positions disagree')
        restore_rng(saved['rng']);return result

    def next_signature(self):
        proposal=self.sampler.propose(self.target_tokens-self.ledger.consumed)
        try:
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):loss,_=teacher_forced_sequences(self.model,proposal['segments'])
            return hashlib.sha256(json.dumps(proposal['events'],separators=(',',':')).encode()).hexdigest(),float(loss)
        finally:self.sampler.resolve(False,self.ledger)


def verify_resume(engine,path):
    engine.save(path);control_rng=rng(engine.device=='cuda')
    resumed=FormalTrainer.load(path,engine.documents,engine.identity,engine.protocol,engine.device,engine.target_tokens)
    reference_row,_=engine.update()
    reference_model=_cpu(engine.model.state_dict());reference_optimizer=_cpu(engine.optimizer.state_dict())
    reference_scheduler=engine.scheduler.state_dict();reference_ledger=engine.ledger.snapshot()
    reference_sampler=engine.sampler.snapshot();reference_flops=_cpu(engine.flops_updates)
    reference_flops_aggregate=_cpu(engine.flops_aggregate);reference_rng=rng(engine.device=='cuda')
    restore_rng(control_rng);resumed_row,_=resumed.update();resumed_rng=rng(engine.device=='cuda')
    timing={'seconds','tokens_per_second'}
    same=({k:v for k,v in reference_row.items() if k not in timing}=={k:v for k,v in resumed_row.items() if k not in timing} and _equal(reference_model,resumed.model.state_dict()) and
        _equal(reference_optimizer,resumed.optimizer.state_dict()) and reference_scheduler==resumed.scheduler.state_dict() and
        reference_ledger==resumed.ledger.snapshot() and reference_sampler==resumed.sampler.snapshot() and
        _equal(reference_flops,resumed.flops_updates) and _equal(reference_flops_aggregate,resumed.flops_aggregate) and
        _equal(reference_rng,resumed_rng) and
        [{k:v for k,v in row.items() if k not in timing} for row in engine.metrics]==[{k:v for k,v in row.items() if k not in timing} for row in resumed.metrics])
    if not same:raise ValueError('interrupted and resumed full-sequence continuations differ')
    resumed.resume_result={'status':'PASS','verified_after_update':reference_row['update']-1,
        'next_prediction_order_sha256':reference_row['data_order_hash'],'model_optimizer_scheduler_rng_sampler_exposure_flops':'BITWISE_EQUAL',
        'verification_updates_on_each_branch':1,'verification_tokens_not_added_to_formal_budget':True}
    resumed.save(path)
    return resumed
