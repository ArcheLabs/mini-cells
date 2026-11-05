"""Four targeted synthetic contracts, never full language training or a benchmark."""
from dataclasses import asdict
import torch
import pytest
from repair_models import build,select_rank,flat_parameters,architecture,ARMS
from runtime import Runtime,weight_hash
from trainer import save_checkpoint
from generative_core import trained_model
from frozen_model import BaselineIntegrityError


def test_event_access_and_independence():
    torch.set_num_threads(1);torch.manual_seed(713)
    independent=build('IC-ALL-SELF');broadcast=build('C1-BROADCAST');broadcast.load_state_dict(independent.state_dict())
    with torch.no_grad():independent.embedding.weight[7]=torch.arange(128,dtype=torch.float32)
    broadcast.load_state_dict(independent.state_dict())
    initial=independent.init_state(1);injected=independent.inject(initial,[7]);other=broadcast.inject(initial,[7])
    assert torch.equal(injected,other) and torch.equal(initial,torch.zeros_like(initial))
    for i in range(16):assert torch.equal(injected[0,i],torch.arange(128,dtype=torch.float32).roll(8*i))
    assert len(torch.unique(injected[0],dim=0))==16
    state=independent.evolve(injected,1);changed=injected.clone();changed[:,4]+=1
    difference=(independent.evolve(changed,1)-state).abs().sum(-1)[0]
    assert difference[4]>0 and torch.count_nonzero(difference)==1
    assert int((state.norm(dim=-1)>1e-6).sum())==16


def test_broadcast_ring_is_synchronous():
    torch.manual_seed(713);model=build('C1-BROADCAST');state=torch.randn_like(model.init_state(1));before=state.clone()
    norm=model.norm(state);message=model.center(norm)+model.left(norm.roll(1,1))+model.right(norm.roll(-1,1))
    expected=state+.1*model.update(torch.cat((norm,message),-1))
    assert torch.equal(model.evolve(state,1),expected) and torch.equal(before,state)
    changed=state.clone();changed[:,15]+=1
    delta=(model.evolve(changed,1)-model.evolve(state,1)).norm(dim=-1)[0]
    assert delta[0]>0 and delta[14]>0 and delta[15]>0 and torch.count_nonzero(delta)==3


def test_active_parameters_and_flat_matching():
    result=select_rank();assert result['rank']==43 and result['parameters']==964736 and result['within_tolerance']
    assert select_rank()==result and flat_parameters(43)==964736
    for variant in ARMS:
        torch.manual_seed(713);model=build(variant);state=model.init_state(1)
        for token in (7,9):state,logits=model.step(state,[token])
        torch.nn.functional.cross_entropy(logits,torch.tensor([11])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0 for p in model.parameters())
        assert model.init_state(1).numel()==2048 and all(p.requires_grad for p in model.parameters())
        count=architecture(model)['active_trainable_parameters']
        assert count==(964736 if variant=='F0-PM-BROADCAST' else 966784)
        if variant=='F0-PM-BROADCAST':
            assert not any(name.startswith(('left','center','right','update')) for name,_ in model.named_parameters())
            assert torch.all(model.up.weight.grad.norm(dim=1)>0)
            before=state.detach();perturbed=before.clone();perturbed[:,0,0]+=1
            delta=(model.evolve(perturbed,1)-model.evolve(before,1)).abs().flatten()
            # A single FP32 finite difference may round individual coordinates to zero.
            # Every output row already has nonzero gradient above; mixing crosses all blocks.
            assert torch.all(delta.reshape(16,128).norm(dim=1)>0)


def test_snapshot_and_checkpoint_identity(tmp_path):
    model=build('IC-ALL-SELF').eval();state=model.init_state(1)
    for token in (7,9):state,_=model.step(state,[token])
    runtime=Runtime(model);snapshot=runtime.snapshot(state)
    assert torch.equal(runtime.restore(snapshot),state)
    other=build('C1-BROADCAST');other.load_state_dict(model.state_dict())
    with pytest.raises(ValueError):Runtime(other).restore(snapshot)
    identity={'source':'repair-source','protocol':'repair-protocol','model':asdict(model.config),'seed':173018021}
    path=tmp_path/'model.pt';save_checkpoint({'identity':identity,'precision':'fp32','tokens':250000,'updates':977,'model':model.state_dict(),'curve':[]},path)
    with pytest.raises(BaselineIntegrityError):
        trained_model(model,[],{**identity,'source':'other-source'},path,{'tokens_per_trained_baseline':250000},float('inf'),lambda:None)


def test_blind_review_and_nested_deadline(tmp_path):
    import csv,json,signal
    from pathlib import Path
    from repair_diagnostics import blind_review
    from repair_io import timed_budget
    base=Path(__file__).resolve().parents[1];plan=json.loads((base/'protocol.json').read_text())
    sample={'prompt_index':0,'prompt':'A story','mode':{'name':'greedy'},'text':'A short result.'}
    models={'MODEL-A':{'status':'COMPLETE','samples':[sample]},'MODEL-B':{'status':'INCOMPLETE','samples':[sample]}}
    result=blind_review(tmp_path,models,plan)
    assert result['status']=='PENDING' and result['reviewers']==0 and result['samples']==1
    text=(tmp_path/'blind-review.csv').read_text();assert 'MODEL-A' not in text and 'MODEL-B' not in text
    rows=list(csv.DictReader((tmp_path/'blind-review.csv').open()));row=rows[0];row['reviewer_id']='human-fixture'
    for dimension in plan['human_review']['dimensions']:row[dimension]='4'
    row['comments']='Supplied fixture comment'
    with (tmp_path/'human-review-scored.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    reviewed=blind_review(tmp_path,models,plan)
    assert reviewed['reviewers']==1 and reviewed['reviews']==1 and reviewed['inter_reviewer_agreement']=='NOT_AVAILABLE'
    assert (tmp_path/'blind-review.csv').read_text()==text
    assert 'Supplied fixture comment' in (tmp_path/'private/human-reviews-preserved.json').read_text()
    with timed_budget(5):
        with timed_budget(1):assert 0<signal.getitimer(signal.ITIMER_REAL)[0]<=1
        assert signal.getitimer(signal.ITIMER_REAL)[0]>3
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
