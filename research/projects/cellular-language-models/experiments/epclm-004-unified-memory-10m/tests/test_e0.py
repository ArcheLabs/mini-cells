import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from torch.nn import functional as F
from data import MQARConfig,episode,batch
from e0 import E0,Config,masked_loss,fit,load_checkpoint,Sampler


def tiny():
    return E0(Config(vocab=536,dim=16,layers=1,heads=2,kv_heads=2,ffn=32,window=1))


def rows(split,n=16):
    c=MQARConfig(bindings=2,length=24)
    return [episode(c,split,i) for i in range(n)]


def test_causal_full_sequence_masked_loss_gradients_and_parameter_count():
    torch.manual_seed(17);m=tiny();x=torch.randint(8,264,(2,12));y=x.clone();y[:,7:]=300
    a=m(x);b=m(y)
    torch.testing.assert_close(a[:,:7],b[:,:7],atol=1e-6,rtol=1e-6)
    # A token beyond Config.window=1 must still have a differentiable path to the final output.
    m.zero_grad();m(x)[:,-1,264].sum().backward()
    assert m.token_embedding.weight.grad[x[0,0]].norm()>0
    labels=torch.full((2,12),-100);labels[0,-1]=264;labels[1,-2:]=265
    loss=masked_loss(a,labels)
    expected=F.cross_entropy(torch.stack([a[0,-1],a[1,-2],a[1,-1]]),torch.tensor([264,265,265]))
    torch.testing.assert_close(loss,expected)
    logits=a.detach().requires_grad_(True);masked_loss(logits,labels).backward()
    assert torch.count_nonzero(logits.grad[labels==-100])==0
    with pytest.raises(ValueError):masked_loss(a,torch.full_like(labels,-100))
    with pytest.raises(ValueError):m.cached(x)
    c=Config(vocab=8192,dim=256,layers=10,heads=4,kv_heads=4,ffn=704,window=16384)
    full=E0(c)
    assert sum(p.numel() for p in full.parameters())==10130688
    assert full.lm_head.weight is full.token_embedding.weight
    # Separate lanes cannot exchange information.
    ids=batch(rows('train',2)).input_ids
    torch.testing.assert_close(m(ids)[0],m(ids[:1])[0])


def test_real_updates_exact_resume_sampler_rollback_and_corruption(tmp_path):
    config={'seed':17,'batch':4,'lr':.001,'updates':4};identity={'fixture':'m1-v1'}
    train=rows('train');dev=rows('validation',8)
    torch.manual_seed(17);whole=tiny();result=fit(whole,train,dev,config,tmp_path/'whole',identity)
    torch.manual_seed(17);part=tiny();first=fit(part,train,dev,config,tmp_path/'part',identity,stop_after=2)
    assert first['status']=='INCOMPLETE'
    resumed=tiny();second=fit(resumed,train,dev,config,tmp_path/'part',identity)
    for a,b in zip(whole.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert result['ledger']==second['ledger']=={'input_tokens':384,'processed_tensor_tokens':384,'padding_tokens':0,
        'supervised_target_tokens':16,'query_target_tokens':16,'committed_updates':4}
    assert all(r['parameter_change_norm']>0 for r in second['history'][:2])
    assert all(torch.isfinite(torch.tensor(r['gradient_norm'])) for r in second['history'])
    saved=load_checkpoint(tmp_path/'part/checkpoint.pt')
    assert all(k in saved for k in ('model','optimizer','rng','sampler','ledger'))
    assert saved['optimizer']['state']
    sampler=Sampler(train,17,4);one=sampler.propose();sampler.resolve(False,{})
    assert sampler.propose()==one
    with pytest.raises(ValueError):fit(tiny(),train,dev,config,tmp_path/'part',{'fixture':'other'})
    raw=(tmp_path/'part/checkpoint.pt').read_bytes()
    (tmp_path/'part/checkpoint.pt').write_bytes(raw[:len(raw)//2])
    with pytest.raises(Exception):load_checkpoint(tmp_path/'part/checkpoint.pt')


def test_nonfinite_attempt_rolls_back_mutated_optimizer_and_ledger(tmp_path,monkeypatch):
    config={'seed':17,'batch':4,'lr':.001,'updates':2};model=tiny()
    original=torch.optim.AdamW.step
    def broken(optimizer,*args,**kwargs):
        original(optimizer,*args,**kwargs)
        with torch.no_grad():optimizer.param_groups[0]['params'][0].fill_(float('nan'))
    monkeypatch.setattr(torch.optim.AdamW,'step',broken)
    with pytest.raises(FloatingPointError):fit(model,rows('train'),rows('validation',4),config,tmp_path,{'fixture':'rollback'})
    checkpoint=load_checkpoint(tmp_path/'checkpoint.pt')
    assert checkpoint['ledger']['committed_updates']==0
    assert all(p.isfinite().all() for p in model.parameters())
    for k,v in model.state_dict().items():torch.testing.assert_close(v,checkpoint['model'][k],rtol=0,atol=0)
