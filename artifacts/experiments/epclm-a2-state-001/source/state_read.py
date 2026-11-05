"""State projections reuse each original Stage's Q/K/V and output weights."""
import math
import torch
from torch.nn import functional as F
from bootstrap import shared


def projected(model,stage,session,index):
    cache=session['state_memory']
    # Keep State normalization/backward in FP32; cast only at the read matmuls.
    with torch.autocast(device_type=session['persistent'].device.type,enabled=False):
        if 'bridge' not in cache:cache['bridge']=model.persistent.bridge(session['persistent'].float())
        if index not in cache:
            d=model.dim;z=stage.n1(cache['bridge']);weight=stage.attn.qkv.weight
            cache[index]=(F.linear(z,weight[d:2*d]),F.linear(z,weight[2*d:]))
    return cache[index]


def record(model,path,session,index,probabilities,context,scores=None):
    if model.collect_gradients and torch.is_grad_enabled():
        if context.requires_grad:context.retain_grad();model.path_tensors[path].append(context)
        if scores is not None and scores.requires_grad:scores.retain_grad();model.path_tensors['attention_scores'].append(scores)
    if model.record_diagnostics:
        with torch.no_grad():
            p=probabilities.float();conditional=p/p.sum(-1,keepdim=True).clamp_min(1e-30)
            model.observations.append({'path':path,'stage':index,'state_revision':session['state_revision'],
                'state_pointer':session['persistent'].data_ptr(),'entropy':-(conditional*conditional.clamp_min(1e-30).log()).sum(-1).mean(),
                'per_cell_mass':p.reshape(-1,16).mean(0),'memory_mass':p.sum(-1).mean(),
                'gate':model.persistent.gate.detach().sigmoid()})


def attention(model,stage,x,session,index,start):
    if not (model.read_attention and model.read_enabled):
        from bootstrap import old_runtime
        return old_runtime.attend_chunk(stage.attn,x,session['layers'],f'stage:{index}/iteration:0',start)
    a=stage.attn;b,t,d=x.shape;heads=a.heads;width=a.head_dim
    q,k,v=a.qkv(x).view(b,t,3,heads,width).unbind(2);q,k,v=(z.transpose(1,2) for z in (q,k,v))
    positions=torch.arange(start,start+t,device=x.device);raw_q=q
    q=shared.rope_at(q,positions[None,None,:,None]);k=shared.rope_at(k,positions[None,None,:,None])
    key=f'stage:{index}/iteration:0';prior=session['layers'].get(key);old=0 if prior is None else prior['k'].shape[2]
    if prior is not None:k=torch.cat((prior['k'],k),2);v=torch.cat((prior['v'],v),2)
    key_positions=torch.arange(start-old,start+t,device=x.device)
    allowed=(key_positions[None,:]<=positions[:,None]) & (key_positions[None,:]>=positions[:,None]-a.window+1)
    ks,vs=projected(model,stage,session,index);ks=ks.view(b,16,heads,width).transpose(1,2);vs=vs.view(b,16,heads,width).transpose(1,2)
    token_scores=(q@k.transpose(-1,-2))/math.sqrt(width)
    token_scores=token_scores.masked_fill(~allowed[None,None],float('-inf'))
    memory_scores=(raw_q@ks.transpose(-1,-2))/math.sqrt(width)
    weights=torch.cat((token_scores,memory_scores),-1).float().softmax(-1).to(v.dtype)
    memory=weights[:,:,:,-16:]@vs*model.persistent.gate.sigmoid()
    record(model,'attention',session,index,weights[:,:,:,-16:],memory,memory_scores)
    delta=weights[:,:,:,:k.shape[2]]@v+memory
    # Only Token K/V enter the bounded KV cache. State slots remain a separate bank.
    session['layers'][key]={'k':k[:,:,-a.window:],'v':v[:,:,-a.window:],'tag':key,'frontier':start+t}
    return a.out(delta.transpose(1,2).contiguous().view(b,t,d))


def ffn_context(model,stage,x,session,index):
    if not (model.read_ffn and model.read_enabled):return x
    d=model.dim;k,v=projected(model,stage,session,index)
    q=F.linear(x,stage.attn.qkv.weight[:d]);weights=((q@k.transpose(-1,-2))/math.sqrt(d)).float().softmax(-1).to(v.dtype)
    context=weights@v;record(model,'ffn',session,index,weights,context)
    return x+model.persistent.gate.sigmoid()*stage.attn.out(context)
