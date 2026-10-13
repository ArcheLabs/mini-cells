"""Independent standard Transformer/MoE trained on the 006 asset contract."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from experiments.common.models import GlobalRouter
from .model import initialize,Rollout
class Block(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['width'];self.heads=c['heads'];self.na=nn.LayerNorm(d);self.nf=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d,bias=False);self.out=nn.Linear(d,d,bias=False);self.ffn=GlobalRouter(d,c['moe_hidden'],c['moe_experts'],c['moe_top_k'])
    def forward(self,h,past=None):
        b,n,d=h.shape;q,k,v=self.qkv(self.na(h)).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        if past is not None:k=torch.cat((past[0],k),-2);v=torch.cat((past[1],v),-2)
        h=h+self.out(F.scaled_dot_product_attention(q,k,v,is_causal=past is None).transpose(1,2).reshape(b,n,d))
        capture={};hook=self.ffn.router.register_forward_hook(lambda _m,_i,o:capture.update(scores=o))
        try:delta,balance=self.ffn(self.nf(h))
        finally:hook.remove()
        return h+delta,(k,v),capture['scores'].topk(self.ffn.top_k,-1).indices,balance
class MoELM(nn.Module):
    def __init__(self,c,seed=0):
        super().__init__();self.config=dict(c);self.arm='M0';d=c['width'];self.embedding=nn.Embedding(c['vocab_size'],d);self.position=nn.Embedding(c['max_sequence_length'],d)
        self.blocks=nn.ModuleList(Block(c) for _ in range(c['moe_layers']));self.final=nn.LayerNorm(d);initialize(self,seed)
    def forward(self,tokens,*,past=None,**kwargs):
        if past is not None and tokens.shape[1]!=1:raise ValueError('cached MoE accepts one new token')
        offset=0 if past is None else past[0][0].shape[-2];pos=torch.arange(offset,offset+tokens.shape[1],device=tokens.device)
        h=self.embedding(tokens)+self.position(pos)[None];caches=[];paths=[];balance=h.new_zeros(())
        for i,block in enumerate(self.blocks):h,cache,path,aux=block(h,None if past is None else past[i]);caches.append(cache);paths.append(path);balance=balance+aux
        paths=torch.stack(paths,dim=-2);return Rollout(F.linear(self.final(h),self.embedding.weight),paths,h.new_empty(0),h.new_empty(0),h.new_empty(0),caches,h,balance/len(self.blocks))
    def prefill(self,tokens,**kwargs):return self(tokens,**kwargs)
    def decode_step(self,token,memory,**kwargs):return self(token.reshape(-1,1),past=memory,**kwargs)
    def parameter_breakdown(self):return {'total':sum(p.numel() for p in self.parameters())}
def make_model(c,arm,seed=0):return MoELM(c,seed) if arm=='M0' else __import__(__package__+'.model',fromlist=['PathFirstLM']).PathFirstLM(c,arm,seed)
