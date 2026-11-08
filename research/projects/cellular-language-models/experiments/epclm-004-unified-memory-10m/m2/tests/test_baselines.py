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


def test_reference_scan_closed_form_and_all_baselines_same_task_cache_gradient():
    import math
    from vendor.selective_scan_ref import selective_scan_ref
    from baselines import cached_forward,runtime_memory
    from e0 import masked_loss
    from data import MQARConfig,episode,batch
    u=torch.ones(1,1,5,requires_grad=True)
    output=selective_scan_ref(u,torch.ones_like(u),-torch.ones(1,2),torch.ones(1,2),torch.ones(1,2),D=torch.ones(1))
    expected=torch.tensor([1+2*(1-math.exp(-(i+1)))/(1-math.exp(-1)) for i in range(5)])
    torch.testing.assert_close(output[0,0],expected)
    output.sum().backward();assert u.grad.isfinite().all() and u.grad.min()>0
    config=dict(vocab=536,dim=16,layers=2,heads=2,kv_heads=2,ffn=32,window=4)
    rows=[episode(MQARConfig(bindings=2,length=24),'train',i) for i in range(2)]
    b=batch(rows);shared=build('E0',config,17).state_dict()
    first_mamba=None
    for arm in ('E0','E1','E2','E3'):
        m=build(arm,{**config,'window':16384} if arm=='E0' else config,17,state_size=4,shared_config=config)
        torch.testing.assert_close(m.token_embedding.weight,shared['token_embedding.weight'],rtol=0,atol=0)
        if arm in ('E2','E3'):
            if first_mamba is None:first_mamba={k:v.clone() for k,v in m.blocks[0].attn.state_dict().items()}
            else:
                for k,v in m.blocks[0].attn.state_dict().items():torch.testing.assert_close(v,first_mamba[k],rtol=0,atol=0)
        original=m(b.input_ids);future=b.input_ids.clone();future[:,12:]=8
        torch.testing.assert_close(original[:,:12],m(future)[:,:12],rtol=1e-6,atol=1e-6)
        loss=masked_loss(original,b.labels);loss.backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
        m.eval();first,state=cached_forward(m,b.input_ids[:,:5]);before_state=runtime_memory(state,2)
        rest,next_state=cached_forward(m,b.input_ids[:,5:],state)
        torch.testing.assert_close(torch.cat([first,rest],1),m(b.input_ids),rtol=2e-5,atol=2e-5)
        assert runtime_memory(state,2)==before_state
        reset,_=cached_forward(m,b.input_ids[:,:5]);torch.testing.assert_close(first,reset)
        torch.testing.assert_close(m(b.input_ids[:1])[0],m(b.input_ids)[0],rtol=1e-5,atol=1e-6)
        memory=runtime_memory(next_state,2)
        if arm=='E2':assert memory['kv_payload_bytes']==0 and memory['mamba_payload_bytes']>0
        if arm=='E3':assert memory['kv_payload_bytes']>0 and memory['mamba_payload_bytes']>0
