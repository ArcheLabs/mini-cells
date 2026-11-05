"""Only the three prescribed small CPU checks; no training or benchmark."""
import json
from pathlib import Path
import pytest
import torch
from model import Core,Config
from frozen_model import DiagnosticCore
from runtime import weight_hash
from communication_propagation import replay,check_cone,IntegrityError


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        model=DiagnosticCore(Config(vocab=16,cells=8,dim=8)).eval()
        for p in model.parameters():p.requires_grad_(False)
        state=torch.randn(1,8,8)
    plan=json.loads((Path(__file__).resolve().parents[1]/'communication-propagation-protocol.json').read_text())
    return model,state,plan


@torch.inference_mode()
def test_no_op_equivalence(fixture,monkeypatch):
    model,state,plan=fixture
    def forbidden(*args,**kwargs):raise AssertionError('training/optimizer forbidden')
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    monkeypatch.setattr(torch.Tensor,'backward',forbidden)
    original=Core(model.config).eval();original.load_state_dict(model.state_dict())
    before=state.clone();weights=weight_hash(model)
    expected,expected_logits=original.step(state,[2])
    injected=model.inject(state.clone(),[2]);trace=[]
    actual=model.evolve(injected,intervention=None,on_update=lambda k,s:trace.append(s.clone()))
    assert torch.equal(expected,actual) and torch.equal(expected_logits,model.read(actual))
    normal=replay(model,state,[2,3,4],[3,4,5],plan,float('inf'))
    sham=replay(model,state.clone(),[2,3,4],[3,4,5],plan,float('inf'),'SHAM',normal=normal)
    assert all(p['state_difference_norm']==0 and p['logit_max_abs_difference']==0 and p['delta_nll']==0 for p in sham['points'])
    assert torch.equal(state,before) and weight_hash(model)==weights
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@torch.inference_mode()
def test_single_edge_isolation(fixture,monkeypatch):
    model,state,plan=fixture;receiver=1
    pulse={'type':'EDGE_DROP_ONCE','receiver':receiver,'source':0,'internal_step':0}
    normal=model.evolve(state,1)
    altered=model.evolve(state,1,intervention=pulse)
    difference=(altered-normal).norm(dim=-1)[0]
    assert difference[receiver]>0 and torch.count_nonzero(difference)==1
    normalized=model.norm(state)
    left=model.left(normalized.roll(1,1)).clone();left[:,receiver]=0
    message=model.center(normalized)+left+model.right(normalized.roll(-1,1))
    expected=state+model.config.alpha*model.update(torch.cat((normalized,message),-1))
    assert torch.equal(altered,expected)
    assert torch.equal(model.evolve(state,4,intervention=pulse),model.evolve(altered,3))
    calls=[];inject=model.inject
    def counted(*args):calls.append(1);return inject(*args)
    normal_path=replay(model,state,[2,3],[3,4],plan,float('inf'))
    monkeypatch.setattr(model,'inject',counted)
    replay(model,state,[2,3],[3,4],plan,float('inf'),'LOCAL',receiver,normal_path)
    assert len(calls)==2  # Once per actual event, never again for internal observations.


@torch.inference_mode()
def test_causal_cone(fixture):
    model,state,plan=fixture
    normal=replay(model,state,[2,3,4,5],[3,4,5,6],plan,float('inf'))
    pulse=replay(model,state.clone(),[2,3,4,5],[3,4,5,6],plan,float('inf'),'LOCAL',1,normal)
    assert len(pulse['spatial_trace'])==16
    for row,reference in zip(pulse['spatial_trace'],normal['states']):
        delta=torch.tensor(row['absolute_state_delta'])
        check_cone(delta,reference.norm(dim=-1)[0],1,row['event_offset'],row['internal_step'],plan['tolerances'])
    assert all((step-1)>=min((int(cell)-1)%8,(1-int(cell))%8) for cell,step in pulse['first_arrival_completed_update'].items())
    impossible=torch.zeros(8);impossible[5]=1
    with pytest.raises(IntegrityError,match='causal cone'):
        check_cone(impossible,torch.ones(8),1,0,0,plan['tolerances'])
