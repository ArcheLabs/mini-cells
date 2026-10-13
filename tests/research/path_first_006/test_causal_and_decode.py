import pytest,torch
from exp006.baseline import make_model
from exp006.profiling import parity
@pytest.mark.parametrize('arm',['R0','R1','R2','M0'])
def test_causal_and_incremental(arm,c):
    model=make_model(c['model'],arm,33).eval();x=torch.arange(16).reshape(2,8);y=x.clone();y[:,4:]=(y[:,4:]+9)%32
    kw={} if arm=='M0' else {'replay':torch.full((2,8,2),3,dtype=torch.long)}
    assert torch.allclose(model(x,**kw).logits[:,:4],model(y,**kw).logits[:,:4],atol=1e-5,rtol=1e-4)
    assert parity(model,x,policy='sampled',generator=torch.Generator().manual_seed(101))['passed']

def test_memory_independent_of_paths(c):
    model=make_model(c['model'],'R2').eval();x=torch.arange(8)[None];a=model(x,replay=torch.zeros(1,8,2,dtype=torch.long));b=model(x,replay=torch.full((1,8,2),3,dtype=torch.long))
    assert torch.equal(a.memory.key,b.memory.key) and torch.equal(a.memory.value,b.memory.value)
    assert not torch.equal(a.logits,b.logits)
