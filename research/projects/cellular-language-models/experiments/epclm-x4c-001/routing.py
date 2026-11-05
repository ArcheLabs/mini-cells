"""Top-2 routing scores all Cells, but invokes the main FFN on selected rows only."""
import torch
from torch import nn


class Router(nn.Module):
    def __init__(self,dim,cells=16,width=128,rank=16,stateful=True):
        super().__init__();self.query=nn.Linear(dim,rank,bias=False)
        self.state_key=nn.Linear(width,rank,bias=False) if stateful else None;self.keys=nn.Parameter(torch.randn(cells,rank)*.02)
        self.cells=cells;self.rank=rank;self.last={};self.activation_counts=None;self.token_queries=0;self.entropy_sum=None

    def forward(self,h,slots=None):
        keys=self.keys[None] if slots is None else self.keys[None]+self.state_key(slots)
        scores=torch.einsum('btd,bnd->btn',self.query(h),keys.expand(h.shape[0],-1,-1))/(self.rank**.5)
        values,indices=scores.topk(2,dim=-1);weights=values.softmax(-1)
        with torch.no_grad():
            p=scores.float().softmax(-1);counts=torch.bincount(indices.reshape(-1),minlength=self.cells)
            entropy=-(p*p.clamp_min(1e-30).log()).sum(-1).mean()
            if self.training:
                amount=entropy*(h.shape[0]*h.shape[1]);self.entropy_sum=amount if self.entropy_sum is None else self.entropy_sum+amount
                self.activation_counts=counts if self.activation_counts is None else self.activation_counts+counts
                self.token_queries+=h.shape[0]*h.shape[1]
            self.last={'counts':counts,'entropy':entropy,'indices':indices.detach()}
        return indices,weights
