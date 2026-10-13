"""Existing FFN primitive and repeatable context operation; no fixed backbone."""
from __future__ import annotations
import torch
from torch import nn
from experiments.common.models import FeedForward
class ContextRead(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.width=width;self.heads=heads;self.query=nn.Linear(width,width,bias=False);self.output=nn.Linear(width,width,bias=False)
    def forward(self,h,memory,batch_ids,positions):
        n=h.shape[0];dh=self.width//self.heads
        q=self.query(h).reshape(n,self.heads,1,dh)
        k=memory.key[batch_ids].reshape(n,-1,self.heads,dh).transpose(1,2)
        v=memory.value[batch_ids].reshape(n,-1,self.heads,dh).transpose(1,2)
        scores=(q@k.transpose(-1,-2))*dh**-0.5
        allowed=torch.arange(k.shape[-2],device=h.device)[None,:]<=positions[:,None]
        scores=scores.masked_fill(~allowed[:,None,None,:],float('-inf'))
        result=(scores.float().softmax(-1).to(v.dtype)@v).reshape(n,self.width)
        return self.output(result)
