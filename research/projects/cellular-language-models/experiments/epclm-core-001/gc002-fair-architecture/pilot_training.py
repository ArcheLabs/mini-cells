"""Approved 50-update T1-PM pilot; transactional state and atomic checksummed checkpoints."""
import hashlib
import json
import time
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from cached_transformer import CachedTransformer,Config
from exposure import ExposureLedger
from training_path import ChunkSampler,teacher_forced_lanes
from flops_audit import ArithmeticCounter
from trainer import save_checkpoint,load_checkpoint,rng,restore_rng
from native_clm_runtime import lr_multiplier


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
    return value


def tree_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


class PilotEngine:
    def __init__(self,documents,config,identity,device,seed,batch,unroll,steps=50):
        torch.manual_seed(seed);self.documents=documents;self.identity=identity;self.device=device
        self.config=config;self.seed=seed;self.batch=batch;self.unroll=unroll;self.steps=steps
        self.model=CachedTransformer(config).to(device=device,dtype=torch.float32).train()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,foreach=False)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:lr_multiplier(max(1,step),steps,min(5,max(1,steps//10))))
        self.sampler=ChunkSampler(documents,seed,batch)
        self.ledger=ExposureLedger(sum(len(d['tokens'])-1 for d in documents),identity)
        self.caches=None;self.curve=[];self.resume_result={'status':'PENDING'};self.training_seconds=0.;self.checkpoint_seconds=0.

    def update(self):
        proposal=self.sampler.propose(self.batch*self.unroll,self.unroll)
        self.optimizer.zero_grad(set_to_none=True);counter=ArithmeticCounter();started=time.monotonic()
        try:
            with sdpa_kernel(SDPBackend.MATH),counter:
                loss,next_caches,tokens=teacher_forced_lanes(self.model,proposal['segments'],self.caches)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite FP32 loss')
                counter.phase='backward';loss.backward()
                counter.phase='gradient_clipping'
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,foreach=False,error_if_nonfinite=True)
                counter.phase='optimizer';self.optimizer.step()
            # One combined synchronization, not a Python per-parameter synchronization.
            if not bool(torch.stack([p.isfinite().all() for p in self.model.parameters()]).all()):raise FloatingPointError('nonfinite parameters after optimizer step')
            if tokens!=self.batch*self.unroll:raise ValueError('pilot update token count differs')
            self.scheduler.step();self.sampler.resolve(True,self.ledger);self.caches=next_caches
        except BaseException:
            if self.sampler.pending is not None:self.sampler.resolve(False,self.ledger)
            self.optimizer.zero_grad(set_to_none=True)
            # Never overwrite the previous atomic checkpoint with a failed attempt.
            raise
        elapsed=time.monotonic()-started;self.training_seconds+=elapsed
        tensors=[v for c in self.caches if c is not None for pair in c.layers for v in pair]
        state_norm=float(torch.stack([v.square().sum() for v in tensors]).sum().sqrt())
        row={'update':self.ledger.updates,'tokens':self.ledger.consumed,'update_tokens':tokens,'loss':float(loss.detach()),
             'gradient_norm_before_clip':float(norm),'state_norm':state_norm,'state_norm_scope':'combined compact per-lane KV tensor L2; not Cellular state norm',
             'seconds':elapsed,'tokens_per_second':tokens/elapsed,'registered_arithmetic':dict(counter.totals),
             'unsupported_operators':dict(counter.unknown),'data_order_hash':self.ledger.order_hash,'lr_next_update':self.optimizer.param_groups[0]['lr']}
        self.curve.append(row);return row

    def save(self,path):
        started=time.monotonic()
        value={'format':'gc002.pilot-state.v1','identity':self.identity,'config':self.config.__dict__,
               'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),
               'sampler':self.sampler.snapshot(),'ledger':self.ledger.snapshot(),'rng':rng(self.device=='cuda'),
               'caches':[self.model.snapshot(c) if c is not None else None for c in (self.caches or [None]*self.batch)],
               'curve':self.curve,'resume_result':self.resume_result,'training_seconds':self.training_seconds,'checkpoint_seconds':self.checkpoint_seconds}
        save_checkpoint(value,path);self.checkpoint_seconds+=time.monotonic()-started

    @classmethod
    def load(cls,path,documents,config,identity,device,seed,batch,unroll,steps=50):
        value=load_checkpoint(path)
        if value['format']!='gc002.pilot-state.v1' or value['identity']!=identity or value['config']!=config.__dict__:raise ValueError('pilot checkpoint source/protocol/data/config differs')
        engine=cls(documents,config,identity,device,seed,batch,unroll,steps)
        engine.model.load_state_dict(value['model'],strict=True);engine.optimizer.load_state_dict(value['optimizer']);engine.scheduler.load_state_dict(value['scheduler'])
        engine.sampler.restore(value['sampler']);engine.ledger=ExposureLedger.restore(value['ledger'],identity)
        engine.caches=[engine.model.restore(c) if c is not None else None for c in value['caches']]
        engine.curve=value['curve'];engine.resume_result=value['resume_result'];engine.training_seconds=value['training_seconds'];engine.checkpoint_seconds=value['checkpoint_seconds']
        if len(engine.curve)!=engine.ledger.updates or engine.ledger.consumed!=engine.ledger.updates*batch*unroll:raise ValueError('pilot counters/checkpoint inconsistent')
        restore_rng(value['rng']);return engine

    def next_signature(self):
        proposal=self.sampler.propose(self.batch*self.unroll,self.unroll)
        try:
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):loss,_,_=teacher_forced_lanes(self.model,proposal['segments'],self.caches)
            return hashlib.sha256(json.dumps(proposal['events'],separators=(',',':')).encode()).hexdigest(),float(loss)
        finally:self.sampler.resolve(False,self.ledger)


def verify_resume(engine,path):
    engine.save(path);expected=engine.next_signature()
    optimizer=cpu_tree(engine.optimizer.state_dict());scheduler=engine.scheduler.state_dict();ledger=engine.ledger.report()
    restored=PilotEngine.load(path,engine.documents,engine.config,engine.identity,engine.device,engine.seed,engine.batch,engine.unroll,engine.steps)
    observed=restored.next_signature()
    if expected[0]!=observed[0] or abs(expected[1]-observed[1])>1e-6 or not tree_equal(optimizer,restored.optimizer.state_dict()) or scheduler!=restored.scheduler.state_dict() or ledger!=restored.ledger.report():raise ValueError('pilot checkpoint continuation mismatch')
    restored.resume_result={'status':'PASS','verified_at_update':restored.ledger.updates,'identical_next_input_sha256':observed[0],
                            'next_loss_absolute_error':abs(expected[1]-observed[1]),'optimizer_scheduler_and_ledger':'EXACTLY_RESTORED',
                            'verification_optimizer_updates':0}
    restored.save(path);return restored
