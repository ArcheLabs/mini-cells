"""Three focused gates, reused locally and by the GPU entry point."""
import torch
from torch import nn
from torch.nn import functional as F
from mamba3 import Mamba3Config
from model import Model,Workspace,cache_bytes


def check(device='cpu'):
    cfg=Mamba3Config(d_model=32,n_layer=2,d_state=8,headdim=8,chunk_size=4,vocab_size=64)
    base=Model(cfg,seed=18001,device=device);wa=Model(cfg,window=4,seed=18001,device=device)
    assert all(torch.equal(v,wa.base.state_dict()[k]) for k,v in base.base.state_dict().items())
    ids=torch.arange(48,device=device).reshape(2,24)%64
    errors={}
    def equal(name,a,b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),name+' nonfinite'
        torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
        errors[name]=float((a-b).abs().max())
    # 1. Preserve the exact Reference block, including its native SwiGLU residual.
    with torch.no_grad():
        equal('base_native',base(ids)[0],base.base(ids)[0])
        equal('adapter_bypass',wa(ids,adapter=False)[0],base(ids)[0])
    out,_=wa(ids);loss=F.cross_entropy(out[:,:-1].reshape(-1,64),ids[:,1:].reshape(-1));loss.backward()
    norms={}
    for name,p in wa.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name+' missing/nonfinite gradient'
        if name.startswith('workspace.'):
            norm=float(p.grad.double().norm());assert norm>0,name+' zero gradient';norms[name]=norm
    # 2. Full vs step across several windows, both caches, no cross-batch state leakage.
    with torch.no_grad():
        full,final=wa(ids);first,cache=wa(ids[:,:4]);outputs=[first]
        for t in range(4,24):
            y,cache=wa(ids[:,t:t+1],cache);outputs.append(y)
            for pair in cache['workspace']:
                assert pair[0].shape[2]<=4 and pair[1].shape[2]<=4
        equal('full_decode',full,torch.cat(outputs,1))
        for kind in ('mamba','workspace'):
            for i,(a,b) in enumerate(zip(final[kind],cache[kind])):
                for j,(x,y) in enumerate(zip(a,b)):equal(f'{kind}_{i}_{j}',x,y)
        perturbed=ids.clone();perturbed[:,12:]=(perturbed[:,12:]+17)%64
        equal('causality',full[:,:12],wa(perturbed)[0][:,:12])
        equal('batch_isolation',full,torch.cat([wa(ids[i:i+1])[0] for i in range(2)]))
        prefix,saved=wa(ids[:,:4]);equal('reset',prefix,wa(ids[:,:4])[0])
        a,_=wa(ids[:,4:5],saved);b,_=wa(ids[:,4:5],saved);equal('cache_not_mutated',a,b)
        swapped={'mamba':[type(layer)(*(x.flip(0) for x in layer)) for layer in saved['mamba']],
                 'workspace':[tuple(x.flip(0) for x in pair) for pair in saved['workspace']]}
        equal('batch_reorder',a.flip(0),wa(ids.flip(0)[:,4:5],swapped)[0])
        sizes=cache_bytes(final)
        bound=2*cfg.n_layer*ids.shape[0]*4*cfg.d_model*4
        assert sizes['workspace']['logical_bytes']==bound
        assert sizes['workspace']['retained_storage_bytes']==bound
    # 3. An analytic moving-mean oracle checks current-token inclusion and window expiry.
    ws=Workspace(4,1,3,device);ws.norm=nn.Identity()
    with torch.no_grad():
        ws.qkv.weight.zero_();ws.qkv.weight[8:12].copy_(torch.eye(4,device=device));ws.out.weight.copy_(torch.eye(4,device=device));ws.gate.fill_(.5)
        u=torch.arange(28,device=device,dtype=torch.float32).reshape(1,7,4)
        actual,_=ws(u)
        expected=torch.stack([u[:,max(0,t-2):t+1].mean(1) for t in range(7)],1)
        equal('causal_window_oracle',actual,u+torch.tanh(ws.gate)*expected)
    return {'status':'PASS','errors':errors,'workspace_gradient_norms':norms,'synthetic_loss':float(loss.detach()),
            'cache':sizes,'workspace_bound_bytes':bound,'checks':3,'optimizer_updates':0,'synthetic_prediction_targets':46}
