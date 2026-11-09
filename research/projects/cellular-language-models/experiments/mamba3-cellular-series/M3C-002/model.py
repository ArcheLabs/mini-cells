"""Layer-local bounded Workspace outside the unmodified Reference blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from mamba3 import Mamba3LMHeadModel, RMSNorm


class Workspace(nn.Module):
    def __init__(self, dim, heads, window, device):
        super().__init__()
        if dim % heads or window < 1:raise ValueError('invalid Workspace heads/window')
        self.heads,self.window=heads,window
        self.norm=RMSNorm(dim,device=device)
        self.qkv=nn.Linear(dim,3*dim,bias=False,device=device)
        self.out=nn.Linear(dim,dim,bias=False,device=device)
        self.gate=nn.Parameter(torch.tensor(0.01,device=device))

    def forward(self, u, cache=None):
        b,t,d=u.shape
        q,k,v=self.qkv(self.norm(u)).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        q,k,v=(x.transpose(1,2) for x in (q,k,v))
        past=0 if cache is None else cache[0].shape[2]
        if cache is not None:k,v=torch.cat((cache[0],k),2),torch.cat((cache[1],v),2)
        row=torch.arange(t,device=u.device)[:,None]+past
        col=torch.arange(k.shape[2],device=u.device)[None,:]
        mask=(col<=row)&(col>row-self.window) # Includes current token; exactly W entries at saturation.
        r=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0)
        r=self.out(r.transpose(1,2).reshape(b,t,d))
        # Clone the slice so the cache never retains the full prefill backing allocation.
        new_cache=(k[:,:,-self.window:].clone(),v[:,:,-self.window:].clone())
        return u+torch.tanh(self.gate)*r,new_cache


class Model(nn.Module):
    def __init__(self, config, window=0, heads=4, seed=18001, device='cpu'):
        super().__init__()
        # Each arm receives identical Reference weights, independent of adapter RNG consumption.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.base=Mamba3LMHeadModel(config,device='cpu').float()
            with torch.no_grad():
                for name,p in self.base.named_parameters():
                    if 'A_log' in name:p.uniform_(-4,-1)
                    elif 'D' in name and p.dim()==1:p.fill_(1)
                    elif 'dt_bias' in name:p.uniform_(.001,.1)
                    elif 'B_bias' in name or 'C_bias' in name:pass
                    elif 'mimo' in name:pass
                    elif p.dim()>=2:p.normal_(std=.02)
            torch.manual_seed(seed+101)
            self.workspace=nn.ModuleList([Workspace(config.d_model,heads,window,'cpu') for _ in range(config.n_layer)]) if window else nn.ModuleList()
        self.to(device)
        # The Reference uses its stored device when allocating SSD tensors.
        self.base.device=device
        for layer in self.base.backbone.layers:layer.mixer.device=device

    def forward(self, ids, cache=None, *, adapter=True):
        t=ids.shape[1]
        if cache is not None and t!=1:raise ValueError('Reference cached continuation requires one token')
        if cache is None:
            states=[None]*self.base.args.n_layer;kv=[None]*self.base.args.n_layer
        else:states=list(cache['mamba']);kv=list(cache['workspace'])
        x=self.base.backbone.embedding(ids);new_states=[];new_kv=[]
        for i,layer in enumerate(self.base.backbone.layers):
            y,state=layer.mixer(layer.mixer_norm(x),states[i]);x=x+y
            x=x+layer.mlp(layer.mlp_norm(x))
            if self.workspace and adapter:x,pair=self.workspace[i](x,kv[i])
            else:pair=None
            new_states.append(state);new_kv.append(pair)
        logits=self.base.lm_head(self.base.backbone.norm_f(x))
        return logits,{'mamba':new_states,'workspace':new_kv}


def cache_bytes(cache):
    result={}
    for name in ('mamba','workspace'):
        tensors=[x for layer in cache[name] if layer is not None for x in layer]
        storage={x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in tensors}
        result[name]={'logical_bytes':sum(x.numel()*x.element_size() for x in tensors),
                      'retained_storage_bytes':sum(storage.values()),
                      'shapes':[list(x.shape) for x in tensors]}
    return result
