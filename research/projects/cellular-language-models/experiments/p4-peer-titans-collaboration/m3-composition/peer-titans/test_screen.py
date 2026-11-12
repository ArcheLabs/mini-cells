"""Tests for the Stage 3-specific data, scoring and failure contracts."""
import json
import torch
import pytest
import screen


def test_nested_and_deterministic():
    small=screen.dataset(7310100,8)
    large=screen.dataset(7310100,16)
    assert all(torch.equal(x,y) for x,y in zip(small,screen.dataset(7310100,8)))
    for i in (0,1):
        assert torch.equal(small[i],large[i][:,:8])
    for i in (2,3,4):
        assert torch.equal(small[i],torch.stack([row[row<8] for row in large[i]]))
    assert len(torch.unique(large[0].flatten(0,1),dim=0))==32
    assert not torch.equal(large[0],large[1])
    assert not torch.equal(large[0],screen.dataset(7310101,16)[0])


def test_score_joint_and_false_accept():
    candidates=torch.eye(4)
    outputs=torch.stack((candidates[[0,1]],candidates[[2,0]]))
    labels=torch.tensor([[0,1],[2,3]])
    early=torch.tensor([[True,False],[True,False]])
    result=screen.score(outputs,candidates,labels,early)
    assert result['recall']==[1.,.5] and result['joint']==.5
    assert result['incorrect_accept']==[0,1]
    raw=torch.tensor([[[.8,.6],[.79,(1-.79**2)**.5]]])
    assert screen.rejection(raw,torch.tensor([[1.,0.]]))['accepted']==[1]


def test_failure_is_retained(tmp_path,monkeypatch):
    def fail(*_):
        raise RuntimeError('Injected acquisition failure')
    monkeypatch.setattr(screen,'episode',fail)
    with pytest.raises(RuntimeError,match='Injected'):
        screen.writer(tmp_path,True)
    rows=list((tmp_path/'episodes').glob('*.json'))
    assert len(rows)==1 and json.loads(rows[0].read_text())['status']=='FAIL'
    assert json.loads((tmp_path/'manifest.json').read_text())['config']==screen.CONFIG
