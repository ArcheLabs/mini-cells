"""M2 baselines reuse frozen M1 modules; E1 changes only visibility."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e0 import E0, Config, CachedTransformer


class E1(CachedTransformer):
    """Causal SWA: max(0,t-W+1)..t, current token included; W keys at most.

    Inherit the explicit mask, bounded cloned KV and differentiable uncached
    forward. Never use inference-only cached() as a training path.
    """
    pass


from dataclasses import dataclass
from types import SimpleNamespace
from torch import nn
from e0 import rope
from cached_transformer import Block
from native_clm_runtime import RMSNorm, HistoricalScaledCLM
from vendor.samba_mamba1 import Mamba


@dataclass
class Runtime:
    """Inference-only SSM conv+recurrent state and Attention KV remain separate."""
    owner: int
    position: int
    mamba_state: dict
    kv_cache: dict
    weight_version: tuple


class RecurrentBaseline(nn.Module):
    def __init__(self,config,arm,state_size=16,seed=41017):
        super().__init__();self.config=config;self.arm=arm
        self.token_embedding=nn.Embedding(config.vocab,config.dim)
        self.blocks=nn.ModuleList()
        self.kinds=[]
        for i in range(config.layers):
            use_ssm=arm=='E2' or i%2==0
            self.kinds.append('mamba1' if use_ssm else 'swa')
            if use_ssm:
                block=nn.Module();block.n1=RMSNorm(config.dim)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+1000+sum(k=='mamba1' for k in self.kinds)-1)
                    block.attn=Mamba(config.dim,d_state=state_size,layer_idx=i,use_fast_path=False)
                if arm=='E3':
                    template=Block(config);block.n2=template.n2;block.ffn=template.ffn
                    block.ffn.apply(HistoricalScaledCLM._init_weights)
            else:block=Block(config)
            self.blocks.append(block)
        self.final_norm=RMSNorm(config.dim)
        self.lm_head=nn.Linear(config.dim,config.vocab,bias=False)
        self.lm_head.weight=self.token_embedding.weight

    def _version(self):return tuple(p._version for p in self.parameters())

    def _forward(self,ids,state=None):
        c=self.config;b,t=ids.shape;h=self.token_embedding(ids);start=state.position if state else 0
        positions=torch.arange(start,start+t,device=ids.device);kv={};ssm={}
        for i,block in enumerate(self.blocks):
            x=block.n1(h)
            if self.kinds[i]=='mamba1':
                if state is None:y=block.attn(x)
                else:
                    existing=state.mamba_state.get(i)
                    params=SimpleNamespace(seqlen_offset=start,key_value_memory_dict={})
                    if existing is not None:params.key_value_memory_dict[i]=tuple(v.clone() for v in existing)
                    if start:
                        parts=[]
                        for token in x.split(1,dim=1):parts.append(block.attn(token,params));params.seqlen_offset+=1
                        y=torch.cat(parts,1)
                    else:y=block.attn(x,params)
                    ssm[i]=params.key_value_memory_dict[i]
            else:
                a=block.attn;d=c.dim//c.heads
                q=rope(a.q(x).view(b,t,c.heads,d).transpose(1,2),positions)
                k=rope(a.k(x).view(b,t,c.kv_heads,d).transpose(1,2),positions)
                v=a.v(x).view(b,t,c.kv_heads,d).transpose(1,2)
                if state is not None and i in state.kv_cache:
                    oldk,oldv=state.kv_cache[i];k=torch.cat([oldk,k],2);v=torch.cat([oldv,v],2)
                keys=torch.arange(start+t-k.shape[2],start+t,device=ids.device)
                mask=(keys[None,:]<=positions[:,None]) & (keys[None,:]>positions[:,None]-c.window)
                n=c.heads//c.kv_heads
                y=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(n,1),v.repeat_interleave(n,1),attn_mask=mask,dropout_p=0.)
                y=a.o(y.transpose(1,2).contiguous().view(b,t,c.dim))
                if state is not None:kv[i]=(k[:,:,-c.window:].clone(),v[:,:,-c.window:].clone())
            h=h+y
            if hasattr(block,'ffn'):h=h+block.ffn(block.n2(h))
        logits=self.lm_head(self.final_norm(h))
        return logits, Runtime(id(self),start+t,ssm,kv,self._version()) if state is not None else None

    def forward(self,ids):return self._forward(ids)[0]

    @torch.no_grad()
    def cached(self,ids,state=None):
        if self.training:raise ValueError('runtime cache is inference-only')
        if state is None:state=Runtime(id(self),0,{},{},self._version())
        if state.owner!=id(self) or state.weight_version!=self._version():raise ValueError('runtime owner/weights differ')
        if state.mamba_state and next(iter(state.mamba_state.values()))[0].shape[0]!=ids.shape[0]:
            raise ValueError('runtime batch differs')
        return self._forward(ids,state)


def build(arm,config,seed,state_size=16,shared_config=None):
    """Shared names/shapes copied from frozen E0; Mamba special init untouched."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm=='E0':return E0(Config(**config))
        if arm=='E1':return E1(Config(**config))
        if arm not in ('E2','E3'):raise ValueError('unknown baseline')
        model=RecurrentBaseline(Config(**config),arm,state_size,seed)
        torch.manual_seed(seed)
        if shared_config is None:
            import json
            canonical=json.loads((Path(__file__).resolve().parents[1]/'configs/model-e0.json').read_text())
            shared_config=canonical if config['vocab']==8192 and config['dim']==256 else config
        reference=E0(Config(**shared_config))
        shared=reference.state_dict()
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name in shared and p.shape==shared[name].shape:p.copy_(shared[name])
        return model


@torch.no_grad()
def cached_forward(model,ids,state=None):
    if model.training:raise ValueError('cache measurements require eval()')
    if isinstance(model,E0):
        # Frozen base cache supports this declared finite measurement range.
        # E0 training itself has no window; do not call this unbounded serving.
        start=state.next_position if state else 0
        if start+ids.shape[1]>model.config.window:raise ValueError('E0 cache measurement range exceeded')
        return model._forward(ids,cache=state,save_cache=True)
    return model.cached(ids,state)


def runtime_memory(state,batch):
    from dataclasses import is_dataclass,fields
    def tensors(x):
        if torch.is_tensor(x):return [x]
        if isinstance(x,dict):return sum([tensors(v) for v in x.values()],[])
        if isinstance(x,(list,tuple)):return sum([tensors(v) for v in x],[])
        if is_dataclass(x):return sum([tensors(getattr(x,f.name)) for f in fields(x)],[])
        return []
    values=tensors(state)
    storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in values}
    ssm=tensors(state.mamba_state) if isinstance(state,Runtime) else []
    return {'tensor_payload_bytes':sum(v.numel()*v.element_size() for v in values),
            'retained_storage_bytes':sum(storages.values()),'retained_bytes_per_stream':sum(storages.values())//batch,
            'mamba_payload_bytes':sum(v.numel()*v.element_size() for v in ssm),
            'kv_payload_bytes':sum(v.numel()*v.element_size() for v in values)-sum(v.numel()*v.element_size() for v in ssm),
            'scope':'runtime tensor state only; no weights/optimizer/activations/Python metadata'}
