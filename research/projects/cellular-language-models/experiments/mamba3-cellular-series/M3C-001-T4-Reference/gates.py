"""Synthetic FP32 checks of the pinned third-party implementation."""
import dataclasses
import torch
from torch.nn import functional as F
from mamba3 import Mamba3Config, Mamba3LMHeadModel


def check_mode(mimo,device):
    torch.manual_seed(18001)
    cfg=Mamba3Config(d_model=32,n_layer=1,d_state=8,expand=2,headdim=8,
                     chunk_size=4,vocab_size=64,use_mimo=mimo,mimo_rank=4)
    model=Mamba3LMHeadModel(cfg,device=device).float().eval()
    # Use upstream create_toy_model initialization; constructor has empty A_log/D/dt_bias.
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'A_log' in name:param.uniform_(-4,-1)
            elif 'D' in name and param.dim()==1:param.fill_(1)
            elif 'dt_bias' in name:param.uniform_(0.001,0.1)
            elif 'B_bias' in name or 'C_bias' in name:pass
            elif 'mimo' in name:pass
            elif param.dim()>=2:param.normal_(std=0.02)
    ids=torch.randint(0,64,(2,12),device=device)
    errors={}
    def equal(name,a,b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),name+' nonfinite'
        torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
        errors[name]=float((a-b).abs().max())
    def clone(cache):return [type(s)(*(v.clone() for v in s)) for s in cache]
    with torch.no_grad():
        full,final=model(ids)
        prefix,cache=model(ids[:,:4]);saved=clone(cache);outputs=[prefix]
        for t in range(4,12):
            y,cache=model(ids[:,t:t+1],cache);outputs.append(y)
        equal('full_cached_logits',full,torch.cat(outputs,dim=1))
        for i,(a,b) in enumerate(zip(final,cache)):
            for name,x,y in zip(a._fields,a,b):equal('layer'+str(i)+'_'+name,x,y)
        a,_=model(ids[:,4:5],clone(saved));b,_=model(ids[:,4:5],clone(saved));equal('snapshot_restore',a,b)
        equal('reset',prefix,model(ids[:,:4])[0])
        equal('batch_isolation',full,torch.cat([model(ids[i:i+1])[0] for i in range(2)]))
        changed=ids.clone();changed[:,8:]=(changed[:,8:]+7)%64
        equal('causality',full[:,:8],model(changed)[0][:,:8])
        assert [x.shape for s in saved for x in s]==[x.shape for s in final for x in s]
    model.zero_grad(set_to_none=True)
    logits,_=model(ids);loss=F.cross_entropy(logits[:,:-1].reshape(-1,64),ids[:,1:].reshape(-1));loss.backward()
    grads={name:p.grad for name,p in model.named_parameters() if p.requires_grad}
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads.values()),'missing/nonfinite gradient'
    norm=sum(float(g.double().square().sum()) for g in grads.values())**.5
    assert torch.isfinite(loss) and norm>0, f'loss={float(loss.detach())}, gradient_norm={norm}'
    return {'status':'PASS','config':dataclasses.asdict(cfg),'precision':'fp32','loss':float(loss.detach()),
            'gradient_norm':norm,'gradient_tensors_checked':len(grads),'all_gradients_finite':True,'parameters':sum(p.numel() for p in model.parameters()),'errors':errors,'optimizer_updates':0,'synthetic_prediction_targets':22,
            'multi_token_cached_continuation':'NOT_SUPPORTED_STEP_REQUIRES_ONE_TOKEN',
            'cache':[{name:{'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size()} for name,x in zip(s._fields,s)} for s in final]}
