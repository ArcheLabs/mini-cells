"""Only new gate-critical regressions; no GPU/remote claims from CPU fixtures."""
import sys,copy,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from dependencies import BASE,frozen
from models import Model
from costs import flops
from precision import select
from main_runner import plan
from training import train
from matching_gate import freeze

torch.set_num_threads(1)
C={'dim':8,'heads':2,'ffn':12,'encoder_ffn':12,'layers':3,'vocab':16,'cells':8,'state_dim':6,'window':256,'chunk':4,'steps':2,'top_k':2}


def test_dead_controller_removal_predictive_and_gradient_equivalence():
    old=frozen('historical_smoke_reference',BASE/'reports/pre-main-gate/frozen_smoke_models.py')
    ids=torch.tensor([[2,4,5,6,7,3]])
    for arm in ('C0','C1','C2','C3','T0-P','T0-F'):
        before=old.Model(arm,C);after=Model(arm,C);after.load_state_dict(before.state_dict());a=[];b=[];old_calls=[];new_calls=[]
        def register(model,records,calls):
            hooks=[]
            if hasattr(model,'router'):hooks.append(model.router.register_forward_hook(lambda m,i,o:records.append(o.detach().clone())))
            if hasattr(model,'state_router'):hooks.append(model.state_router.register_forward_hook(lambda m,i,o:records.append(o.detach().clone())))
            if hasattr(model,'controller'):hooks.append(model.controller.register_forward_hook(lambda m,i,o:calls.append(o.detach().clone())))
            return hooks
        hooks=register(before,a,old_calls)+register(after,b,new_calls)
        x,_=before(ids);y,_=after(ids);lx=x.square().mean();ly=y.square().mean();lx.backward();ly.backward()
        assert torch.equal(x,y) and torch.equal(lx,ly)
        assert all(torch.equal(v,w) for v,w in zip(a,b))
        for name,value in before.named_parameters():
            other=dict(after.named_parameters())[name]
            assert (value.grad is None)==(other.grad is None)
            if value.grad is not None:assert torch.equal(value.grad,other.grad),name
        if arm in ('C2','C3'):assert len(old_calls)==2 and len(new_calls)==1 and torch.equal(old_calls[0],new_calls[0])
        for hook in hooks:hook.remove()
    full=json.loads((BASE/'protocol.json').read_text())['model']
    assert flops('C2',full)['total']==8679424


def test_native_precision_policy_does_not_confuse_compatibility(monkeypatch):
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda *a:(7,5));monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'Tesla T4')
    p=select('auto','cuda');assert p['selected_precision']=='fp16' and p['scaler_enabled'] and not p['bf16_native_hardware_supported']
    with pytest.raises(ValueError):select('bf16','cuda')
    assert select('bf16','cuda',True)['diagnostic_non_native_bf16']
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda *a:(8,0))
    assert select('auto','cuda')['selected_precision']=='bf16'


def test_main_seed_scheduler_isolation_exact_plan_and_control_definitions(tmp_path):
    p=json.loads((BASE/'protocol.json').read_text());p['model']=C;p['groups']['documents']=1
    documents=[{'doc_id':'a','ids':[2,4,5,3]},{'doc_id':'b','ids':[2,5,3]}]
    groups,n=plan(documents,p,12);assert n==12 and sum(len(d['ids'])-1 for g in groups for d in g)==12
    a=Model('C2',C,17);b=Model('C2',C,29);assert not torch.equal(a.token.weight,b.token.weight)
    p['schedule_target_budget']=12;p['session_stop_targets']=len(groups[0][0]['ids'])-1
    first,_=train(a,groups,p,time.monotonic()+20,tmp_path/'resume.pt','exact-main-dry')
    assert first['targets']==p['session_stop_targets'] and first['status']=='INCOMPLETE'
    del p['session_stop_targets'];resumed,_=train(a,groups,p,time.monotonic()+20,tmp_path/'resume.pt','exact-main-dry')
    continuous=Model('C2',C,17);reference,_=train(continuous,groups,p,time.monotonic()+20,tmp_path/'full.pt','continuous')
    assert resumed['targets']==reference['targets']==12
    assert [r['lr'] for r in resumed['rows']]==[r['lr'] for r in reference['rows']]
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in continuous.state_dict().items())
    pool=Model('C0-PoolBalanced',C);assert not hasattr(pool,'router')
    output,_=pool(torch.tensor([[2,4,5,6,7,8,9,10]]));assert output.shape==(1,8,16) and all(v>0 for v in pool.call_counts)
    matched=freeze(C);assert matched['H2']==matched['H3']=='CONTROL_DESIGN_FROZEN'
