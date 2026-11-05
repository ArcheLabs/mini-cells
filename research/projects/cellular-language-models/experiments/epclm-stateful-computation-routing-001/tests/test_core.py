"""Only critical semantic regressions; tiny CPU documents, no benchmarks/training corpus."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import json
import time
import pytest
import torch
from models import Model,detach
from engine import group_update,optimizer,evaluate,paired
from run import audit,common_hash
from dependencies import BASE

torch.set_num_threads(1)
C={'dim':8,'heads':2,'ffn':12,'layers':2,'vocab':16,'cells':4,'state_dim':6,'router_dim':6,'window':8,'chunk':4,'K':2}
D=[{'doc_id':'a','ids':[2,4,5,6,7,8,9,3]},{'doc_id':'b','ids':[2,9,8,3]}]

def config():
    p=json.loads((BASE/'protocol.json').read_text());p['model']=C;p['evaluation']['bootstrap_repetitions']=20;return p


def test_common_initialization_and_audit():
    models=[Model(a,C) for a in ('M0','M1','M2')]
    assert len({common_hash(m) for m in models})==1
    assert models[0].reset('a')['state'] is None
    assert torch.equal(models[1].update.weight_ih,models[2].update.weight_ih)
    assert audit(json.loads((BASE/'protocol.json').read_text()))['audit_status']=='PASS'


@pytest.mark.parametrize('arm',['T0','M0','M1','M2'])
def test_causal_cache_detach_and_document_isolation(arm):
    model=Model(arm,C);model.sample=False;x=torch.tensor([[2,4,5,6,7,8]])
    with torch.no_grad():
        whole,cache,_=model(x,model.reset('a'))
        first,part,_=model(x[:,:3],model.reset('a'));second,part,_=model(x[:,3:],part)
        changed,_,_=model(torch.tensor([[2,4,5,9,9,9]]),model.reset('a'))
        assert torch.allclose(whole,torch.cat((first,second),1),atol=1e-6,rtol=1e-5)
        assert torch.allclose(whole[:,:3],changed[:,:3],atol=1e-6)
    if arm!='T0':assert model.writes==18 and model.calls==36
    output,cache,_=model(x,model.reset('a'));cut=detach(cache)
    def tensors(v):
        if torch.is_tensor(v):return [v]
        if isinstance(v,dict):return sum((tensors(a) for a in v.values()),[])
        if isinstance(v,(tuple,list)):return sum((tensors(a) for a in v),[])
        return []
    assert all(not v.requires_grad for v in tensors(cut))
    assert all(torch.equal(a,b) for a,b in zip(tensors(cache),tensors(cut)))
    streams={'a':cache,'b':model.reset('b')}
    reordered=[streams[i] for i in ('b','a')];assert reordered[0]['doc_id']=='b' and reordered[1]['position']==6
    if arm in ('M1','M2'):assert bool(streams['b']['state'].eq(0).all())


def test_selected_state_paths_and_ablation():
    for arm in ('M1','M2'):
        m=Model(arm,C);a=m.reset('a');b=m.reset('a');b['state'][:,0]+=20
        with torch.no_grad():
            _,_,la=m(torch.tensor([[2]]),a);_,_,lb=m(torch.tensor([[2]]),b)
        if arm=='M1':assert torch.equal(la[0],lb[0])
        else:assert not torch.equal(la[0],lb[0])
    repeated=Model('M1',C);repeated.query.weight.data.zero_();updates=[]
    hook=repeated.update.register_forward_hook(lambda m,args,out:updates.append((args[1].clone(),out.clone())))
    with torch.no_grad():repeated(torch.tensor([[2]]),repeated.reset('a'))
    hook.remove();assert len(updates)==2 and torch.equal(updates[1][0],updates[0][1])
    model=Model('M2',C);model.diagnostics=True
    with torch.no_grad():
        _,cache,_=model(torch.tensor([[2]]),model.reset('a'))
        selected={r['cell'] for r in model.trace};s=cache['state'][0]
        assert all(bool(s[i].eq(0).all()) for i in range(4) if i not in selected)
        assert all(bool(s[i].ne(0).any()) for i in selected)
        model.mode='write_off';_,c,_=model(torch.tensor([[2,4]]),model.reset('a'));assert bool(c['state'].eq(0).all())
        for arm in ('M1','M2'):
            m=Model(arm,C);m.mode='all_off';a=m.reset('a');b=m.reset('a');b['state'][:,0]+=200
            y,_,_=m(torch.tensor([[2,4]]),a);z,_,_=m(torch.tensor([[2,4]]),b)
            assert torch.equal(y,z)


def test_real_group_gradients_modes_and_failure_rollback(tmp_path):
    p=config();model=Model('M2',C);opt=optimizer(model,p);g=torch.Generator().manual_seed(7)
    row=group_update(model,D,opt,p,0.,g,time.monotonic()+30)
    assert row['targets']==10 and row['module_gradient_norms']['query']>0 and row['module_gradient_norms']['update']>0
    normal=evaluate(model,D,p,time.monotonic()+30);sampled=evaluate(model,D,p,time.monotonic()+30,sample=True)
    assert normal['routing']=='argmax' and sampled['routing']=='categorical_fixed_seed'
    assert paired(normal,normal,p)['relative_nll']==0
    before=copy.deepcopy(model.state_dict());rng=g.get_state().clone()
    with pytest.raises(TimeoutError):group_update(model,D,opt,p,row['baseline'],g,time.monotonic()-1)
    assert torch.equal(rng,g.get_state()) and all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path);restored=Model('M2',C);restored.load_state_dict(torch.load(path,weights_only=True))
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
