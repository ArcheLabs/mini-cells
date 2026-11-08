import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baselines import build


def test_e1_same_initialization_window_boundary_causality_and_cache():
    c=dict(vocab=536,dim=16,layers=1,heads=2,kv_heads=2,ffn=32,window=4)
    e0=build('E0',c,17);e1=build('E1',c,17)
    for name,p in e0.state_dict().items():torch.testing.assert_close(p,e1.state_dict()[name],rtol=0,atol=0)
    x=torch.tensor([[8,9,10,11,12,13,14,15]])
    torch.testing.assert_close(e0(x[:,:4]),e1(x[:,:4]),atol=1e-6,rtol=1e-6)
    y=x.clone();y[:,0]=20
    torch.testing.assert_close(e1(x)[:,4:],e1(y)[:,4:],atol=0,rtol=0)
    y=x.clone();y[:,5:]=20
    torch.testing.assert_close(e1(x)[:,:5],e1(y)[:,:5],atol=0,rtol=0)
    e1.eval();prefill,state=e1.cached(x[:,:3]);later,state=e1.cached(x[:,3:],state)
    torch.testing.assert_close(torch.cat([prefill,later],1),e1(x),atol=1e-6,rtol=1e-6)
    assert all(k.shape[2]==4 for k,v in state.layers)
    assert sum(p.numel() for p in e0.parameters())==sum(p.numel() for p in e1.parameters())
