import sys,copy,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models import Model,Session,detach
from engine import group_update,optimizer
from dependencies import BASE

torch.set_num_threads(1)
C={'dim':8,'heads':2,'ffn':12,'layers':2,'vocab':16,'cells':4,'state_dim':6,'router_dim':6,'window':128,'chunk':4,'K':2}
def model(arm='M2'):return Model(arm,C).eval()
def ids(n=10):return torch.arange(n)[None]%12+2

def test_chunk_parallel_causality():
    for size in (32,64,128):
        c={**C,'chunk':size};m=Model('M2',c);x=ids(size+2);changed=x.clone();changed[:,size//2:]=4
        with torch.no_grad():a,_,_=m(x,m.reset('a'));b,_,_=m(changed,m.reset('a'))
        assert torch.allclose(a[:,:size//2],b[:,:size//2],atol=2e-6)

def test_workspace_input_kv():
    m=model();x=ids(3)
    with torch.no_grad():_,cache,_=m(x,m.reset('a'));_,k,v=m.workspace.project(m.work_norm(m.token(x)),0)
    assert torch.equal(cache['kv'][0],k) and torch.equal(cache['kv'][1],v)

def test_workspace_parallel_equivalence():
    m=model('M0');x=ids(8)
    with torch.no_grad():a,_,_=m(x,m.reset('a'));s=Session(m);b=torch.cat([s.prefill(x[:,i:i+1]) for i in range(8)],1)
    assert torch.allclose(a,b,atol=2e-6,rtol=2e-5)

def test_workspace_streaming_equivalence():
    for arm in ('T0','M0','M1','M2'):
        m=model(arm);x=ids(10)
        with torch.no_grad():a,_,_=m(x,m.reset('a'))
        s=Session(m);b=torch.cat([s.prefill(t) for t in x.split(3,1)],1)
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-5)

def test_no_final_output_kv_recursion():
    m=model();x=ids(3)
    with torch.no_grad():_,a,_=m(x,m.reset('a'));m.output.weight.add_(20);_,b,_=m(x,m.reset('a'))
    assert torch.equal(a['kv'][0],b['kv'][0]) and torch.equal(a['kv'][1],b['kv'][1])

def test_state_snapshot_consistency():
    m=model();s=Session(m);s.prefill(ids(3));assert s.cache['state'].eq(0).all();assert s.cache['pending_count'].sum()==6

def test_no_within_chunk_state_leak():
    m=model();x=ids(4);a=m.reset('a');b=m.reset('a');b['pending_sum']=torch.ones(1,4,6)*100;b['pending_count']=torch.ones(1,4)
    with torch.no_grad():y,_,_=m(x,a);z,_,_=m(x,b)
    assert torch.equal(y,z)

def test_state_aggregate_per_cell():
    m=model();s=Session(m);s.prefill(ids(3));counts=s.cache['pending_count'];sums=s.cache['pending_sum'];previous=s.cache['state']
    with torch.no_grad():expected=m.update((sums/counts.clamp_min(1)[:,:,None]).reshape(-1,6),previous.reshape(-1,6)).reshape_as(previous)
    s.finish();assert torch.allclose(s.cache['state'],torch.where((counts>0)[:,:,None],expected,previous))

def test_inactive_state_identity():
    m=model('M1');m.query.weight.data.zero_();s=Session(m);s.prefill(ids(4));assert s.cache['state'][:,1:].eq(0).all();assert s.cache['state'][:,0].ne(0).any()

def test_cross_chunk_state_gradient():
    for arm in ('M1','M2'):
        m=model(arm);x=ids(8);a,c,_=m(x[:,:4],m.reset('a'));b,c,_=m(x[:,4:],c)
        b.square().sum().backward();assert m.writer.weight.grad.abs().sum()>0
        m.zero_grad();a,c,_=m(x[:,:4],m.reset('a'));b,_,_=m(x[:,4:],detach(c));b.square().sum().backward()
        assert m.writer.weight.grad is None

def test_router_gradient():
    p=json.loads((BASE/'protocol.json').read_text());p['model']=C;m=model();g=torch.Generator().manual_seed(7)
    r=group_update(m,[{'doc_id':'a','ids':ids(10)[0].tolist()}],optimizer(m,p),p,[0.,0.],g,time.monotonic()+20)
    assert r['module_gradient_norms']['query']>0 and r['module_gradient_norms']['writer']>0
    assert r['writes_with_next_chunk_gradient']==1

def test_sparse_cellcore_execution():
    m=model();sizes=[];hook=m.core.register_forward_hook(lambda m,a,o:sizes.append(a[0].shape))
    with torch.no_grad():m(ids(4),m.reset('a'))
    hook.remove();assert sizes==[torch.Size([1,4,8])]*2 and m.calls==8

def test_chunk_partition_consistency():
    m=model();x=ids(10);a=Session(m);b=Session(m)
    y=a.prefill(x);z=torch.cat([b.prefill(t) for t in x.split(2,1)],1)
    assert torch.allclose(y,z,atol=2e-6);assert torch.allclose(a.cache['state'],b.cache['state'],atol=2e-6)

def test_document_state_isolation():
    m=model();a=Session(m,'a');b=Session(m,'b');a.prefill(ids(4));assert b.cache['state'].eq(0).all() and b.cache['kv'] is None

def test_padding_write_mask():
    m=model();x=ids(4);mask=torch.tensor([[True,True,False,False]])
    with torch.no_grad():_,c,_=m(x,m.reset('a'),mask=mask)
    assert m.calls==4
    # Trailing masked inputs cannot change state proposals from valid positions.
    changed=x.clone();changed[:,2:]=9
    with torch.no_grad():_,d,_=m(changed,m.reset('a'),mask=mask)
    assert torch.allclose(c['state'],d['state'],atol=1e-6)

def test_snapshot_restore():
    m=model();a=Session(m);a.prefill(ids(3));saved=a.snapshot();x=ids(5)
    b=Session(m).restore(saved);assert torch.allclose(a.prefill(x),b.prefill(x),atol=2e-6)
    assert torch.allclose(a.cache['state'],b.cache['state'],atol=2e-6)
