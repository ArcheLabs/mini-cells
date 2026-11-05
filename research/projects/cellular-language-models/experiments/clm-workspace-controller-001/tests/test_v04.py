import sys,copy,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from torch.nn import functional as F
from dependencies import BASE
from models import Model,Session
from costs import match
from training import train

torch.set_num_threads(1)
C={'dim':8,'heads':2,'ffn':12,'encoder_ffn':12,'layers':3,'vocab':16,'cells':8,'state_dim':6,'window':256,'chunk':4,'steps':2,'top_k':2}
def x(n=8):return torch.arange(n)[None]%12+2

def test_workspace_causal_cache_and_common_initialization():
    for arm in ('C0','C1','C2','C3','T0-P'):
        m=Model(arm,C).eval();ids=x();changed=ids.clone();changed[:,4:]=4
        with torch.no_grad():a,cache=m(ids);b,_=m(changed)
        assert torch.allclose(a[:,:4],b[:,:4],atol=1e-6)
        s=Session(m);cached=torch.cat([s.prefill(part) for part in ids.split(3,1)],1)
        assert torch.allclose(a,cached,atol=3e-6,rtol=3e-5)
        step=Session(m);single=torch.cat([step.prefill(token) for token in ids.split(1,1)],1)
        assert torch.allclose(a,single,atol=3e-6,rtol=3e-5)
        if arm.startswith('C'):
            assert cache['encoder_kv'][0].data_ptr()!=cache['workspace_kv'][0].data_ptr()
            assert cache['workspace_kv'][0].ndim==4
    a=Model('C1',C);b=Model('C2',C)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_shared_kv_repeated_reads_and_no_cell_feedback():
    m=Model('C2',C).eval();ids=x();shapes=[]
    hook=m.query.register_forward_hook(lambda m,args,out:shapes.append(out.clone()))
    with torch.no_grad():a,c=m(ids)
    hook.remove();assert len(shapes)==2 and not torch.equal(shapes[0],shapes[1])
    with torch.no_grad():
        for cell in m.cells:cell.down.weight.add_(2)
        b,d=m(ids)
    assert torch.equal(c['workspace_kv'][0],d['workspace_kv'][0]) and torch.equal(c['workspace_kv'][1],d['workspace_kv'][1])
    assert not torch.equal(a,b)


def test_sparse_dispatch_dense_reference_and_expert_independence():
    m=Model('C1',C);v=torch.randn(2,3,8);indices=torch.tensor([[[0,1],[0,2],[2,3]],[[1,3],[2,0],[3,1]]]);weights=torch.randn(2,3,2).softmax(-1)
    seen=[];hooks=[cell.register_forward_hook(lambda m,args,out,i=i:seen.append(i)) for i,cell in enumerate(m.cells)]
    actual=m.dispatch(v,indices,weights)
    for h in hooks:h.remove()
    assert set(seen)=={0,1,2,3}
    reference=torch.stack([cell(v) for cell in m.cells],-2).gather(-2,indices[:,:,:,None].expand(-1,-1,-1,8))
    assert torch.allclose(actual,(reference*weights[:,:,:,None]).sum(-2),atol=1e-6)
    actual.square().sum().backward();assert m.cells[7].up.weight.grad is None
    assert len({cell.up.weight.data_ptr() for cell in m.cells})==8
    assert not torch.equal(m.cells[0].up.weight,m.cells[1].up.weight)


def test_router_gates_controller_gradient_and_state_off():
    m=Model('C2',C);m.diagnostics=True;logits,_=m(x())
    F.cross_entropy(logits.reshape(-1,16),x().reshape(-1)).backward()
    assert m.router.weight.grad.abs().sum()>0 and m.controller.weight_ih.grad.abs().sum()>0
    assert len(m.records)==2 and m.records[0]['state_norm']==0 and m.records[1]['state_norm']>0
    m.eval();m.mode='state_off';a,_=m(x());stateless=Model('C1',C);stateless.load_state_dict({k:v for k,v in m.state_dict().items() if k in stateless.state_dict()})
    b,_=stateless(x());assert torch.equal(a,b)
    assert m(x())[1].get('controller_state') is None


def test_token_state_isolation_and_repeated_read_ablation():
    m=Model('C2',C).eval();single=m(x())[0];batch=m(x().expand(2,-1))[0]
    assert torch.allclose(single,batch[:1],atol=2e-6)
    changed=x().expand(2,-1).clone();changed[1]=4;other=m(changed)[0];assert torch.equal(batch[0],other[0])
    c3=Model('C3',C);c3.load_state_dict(m.state_dict());calls=[];h=c3.query.register_forward_hook(lambda *args:calls.append(1));c3(x());h.remove();assert len(calls)==1
    assert set(m.state_dict())==set(c3.state_dict())


def test_counterfactual_paths_are_executable_and_reproducible():
    m=Model('C2',C).eval();m.diagnostics=True;m.mode='fixed';m.fixed_pair=(1,3);m(x())
    assert all(set(pair)=={1,3} for record in m.records for row in record['selected'] for pair in row)
    assert m.records[1]['state_norm']>0
    m.mode='random';m.generator=torch.Generator().manual_seed(4);a=m(x())[0]
    m.generator.manual_seed(4);b=m(x())[0];assert torch.equal(a,b)
    from evaluate import evaluate
    from run import launch
    p=json.loads((BASE/'protocol.json').read_text());p['model']=C
    documents=[{'doc_id':'fixture','ids':x()[0].tolist()}]
    normal=evaluate(m,documents,p,time.monotonic()+20)
    shuffled=evaluate(m,documents,p,time.monotonic()+20,'shuffle')
    assert normal['expert_activation_counts']==shuffled['expert_activation_counts']


def test_checkpoint_resume_and_group_accounting(tmp_path):
    p=json.loads((BASE/'protocol.json').read_text());p['model']=C;p['smoke']['seed']=17
    groups=[[{'doc_id':'a','ids':[2,4,5,6,7,8,3]}],[{'doc_id':'b','ids':[2,9,8,7,6,5,3]}]]
    model=Model('C2',C);path=tmp_path/'checkpoint.pt'
    progress,perf=train(model,groups,p,time.monotonic()+30,path,'fixture')
    assert progress['status']=='COMPLETE' and progress['targets']==12 and progress['updates']==2
    restored=Model('C2',C);again,_=train(restored,groups,p,time.monotonic()+30,path,'fixture')
    assert again['targets']==12 and again['updates']==2
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):train(restored,groups,p,time.monotonic()+30,path,'different-source')
    interrupted=Model('C2',C);partial=tmp_path/'partial.pt'
    def interrupt(checkpoint):raise TimeoutError('simulated interruption after committed checkpoint')
    first,_=train(interrupted,groups,p,time.monotonic()+30,partial,'partial',checkpoint_hook=interrupt)
    assert first['status']=='INCOMPLETE' and first['updates']==1
    resumed=Model('C2',C);done,_=train(resumed,groups,p,time.monotonic()+30,partial,'partial')
    assert done['updates']==2 and all(torch.equal(v,resumed.state_dict()[k]) for k,v in model.state_dict().items())
    from recovery import commit_bundle
    from dependencies import shared
    out=tmp_path/'bundle';out.mkdir();(out/'manifest.json').write_text('{}');asset=out/'model.pt';asset.write_bytes(b'old')
    commit_bundle(out,'identity',asset);bundle=json.loads((out/'recovery.json').read_text());physical=out/bundle['objects']['model.pt']
    pending=out/'pending';pending.write_bytes(b'new');pending.replace(asset)
    assert physical.read_bytes()==b'old' and shared.digest(physical)==bundle['files']['model.pt']


def test_matching_freezes_two_distinct_budget_comparisons():
    c={**C,'dim':16,'heads':2,'ffn':32,'encoder_ffn':32,'state_dim':16,'vocab':32}
    matched=match(c)
    for arm in ('T0-P','T0-F'):
        assert matched[arm]['parameters']==sum(p.numel() for p in Model(arm,matched[arm]['config']).parameters())
        assert matched[arm]['relative_matching_error']>=0
