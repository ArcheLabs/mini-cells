"""Focused CPU fault injection for overflow transactions; no GPU benchmark."""
import copy
from contextlib import nullcontext
import time
import torch
import pytest
from model import Core,Config
from trainer import train,load_checkpoint,NumericalFailure


class FakeScaler:
    always_overflow=False
    optimizer_steps=0

    def __init__(self,*args,init_scale=1024.,**kwargs):
        self.value=init_scale; self.calls=0; self.found=False
    def is_enabled(self): return True
    def scale(self,loss): return loss
    def get_scale(self): return self.value
    def get_backoff_factor(self): return .5
    def unscale_(self,optimizer):
        self.calls+=1
        parameters=[p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if self.always_overflow or self.calls==1:
            parameters[0].grad.fill_(float('inf'))
        self.found=any(not torch.isfinite(p.grad).all() for p in parameters)
    def step(self,optimizer):
        if not self.found:
            type(self).optimizer_steps+=1
            optimizer.step()
    def update(self,new_scale=None):
        if new_scale is not None: self.value=new_scale
        elif self.found: self.value*=.5
        self.found=False
    def state_dict(self): return {'scale':self.value}
    def load_state_dict(self,state): self.value=state['scale']


def test_overflow_transaction(tmp_path,monkeypatch):
    torch.set_num_threads(1); torch.manual_seed(7)
    original=Core(Config(vocab=16,cells=4,dim=8))
    reference=copy.deepcopy(original); candidate=copy.deepcopy(original)
    documents=[{'id':'a','tokens':[1,2,3,4,5]},{'id':'b','tokens':[5,4,3,2,1]}]
    kwargs=dict(documents=documents,identity={'test':'overflow'},target_tokens=8,
                deadline=time.monotonic()+120,seed=17,batch=2,unroll=2)
    fp32=train(reference,path=tmp_path/'reference.pt',precision='fp32',**kwargs)
    assert fp32['updates']==2
    assert fp32['effective_precision']['autocast_enabled'] is False
    assert fp32['effective_precision']['grad_scaler_enabled'] is False
    assert fp32['effective_precision']['model_dtype']==fp32['effective_precision']['state_dtype']=='torch.float32'
    assert fp32['effective_precision']['optimizer_dtypes']==['torch.float32']
    assert any(not torch.equal(a,b) for a,b in zip(reference.parameters(),original.parameters()))
    with pytest.raises(ValueError,match='precision mismatch'):
        train(copy.deepcopy(original),path=tmp_path/'reference.pt',precision='fp16',**kwargs)
    monkeypatch.setattr(torch.amp,'GradScaler',FakeScaler)
    monkeypatch.setattr(torch,'autocast',lambda **kwargs:nullcontext())
    clip=torch.nn.utils.clip_grad_norm_
    def guarded_clip(parameters,*args,**kwargs):
        parameters=list(parameters)
        assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
        return clip(parameters,*args,**kwargs)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',guarded_clip)
    FakeScaler.optimizer_steps=0
    result=train(candidate,path=tmp_path/'candidate.pt',**kwargs)
    assert FakeScaler.optimizer_steps==2
    assert result['updates']==2 and result['tokens']==8
    event=result['overflow_diagnostics'][0]
    assert event['committed_tokens']==0 and event['update']==1
    assert event['grad_scaler_scale']==1024 and event['next_scale']==512
    assert event['classification']=='MIXED_PRECISION_PATH'
    assert event['fp32']['input_sha256']==event['input_sha256']
    assert event['fp32']['optimizer_updates']==0
    for a,b in zip(reference.parameters(),candidate.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    expected=load_checkpoint(tmp_path/'reference.pt'); actual=load_checkpoint(tmp_path/'candidate.pt')
    for key in ('order','cursor','lanes','positions','tokens','updates','scheduler'):
        assert actual[key]==expected[key]
    torch.testing.assert_close(actual['state'],expected['state'],atol=0,rtol=0)
    torch.testing.assert_close(actual['rng']['torch'],expected['rng']['torch'],atol=0,rtol=0)

    monkeypatch.setattr(FakeScaler,'always_overflow',True)
    FakeScaler.optimizer_steps=0
    with pytest.raises(NumericalFailure) as failure:
        train(copy.deepcopy(original),path=tmp_path/'limit.pt',**kwargs)
    assert len(failure.value.diagnostics)==3 and FakeScaler.optimizer_steps==0
    assert failure.value.diagnostics[-1]['reused_fp32_diagnostic']
    saved=load_checkpoint(tmp_path/'limit.pt')
    assert saved['updates']==saved['tokens']==0
    assert saved['cursor']==0 and saved['positions']==[0,0] and saved['lanes']==[None,None]
    assert saved['optimizer']['state']=={} and saved['scaler']['scale']==128
    assert torch.count_nonzero(saved['state'])==0

    unstable=copy.deepcopy(original)
    unstable.embedding.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
    with pytest.raises(NumericalFailure) as failure:
        train(unstable,path=tmp_path/'unstable.pt',**kwargs)
    assert failure.value.diagnostics[-1]['classification']=='MODEL_OR_TBPTT_INSTABILITY'
    assert len(failure.value.diagnostics)==1
