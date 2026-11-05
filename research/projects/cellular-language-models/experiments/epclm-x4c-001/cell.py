"""A shared small SwiGLU plus selected Cell-specific low-rank computation."""
import torch
from torch import nn
from torch.nn import functional as F
from common import native
from routing import Router


class SelectiveCellFFN(nn.Module):
    def __init__(self,dim,hidden=128,stateful=True,cells=16,width=128,adapter_rank=16):
        super().__init__();self.stateful=stateful;self.router=Router(dim,cells,width,stateful=stateful)
        self.shared=native.SwiGLU(dim,hidden)
        self.down=nn.Parameter(torch.randn(cells,dim,adapter_rank)*.02)
        self.up=nn.Parameter(torch.randn(cells,adapter_rank,dim)*.02)
        self.state_input=nn.Linear(width,dim,bias=False) if stateful else None
        self.executed_rows=0;self.total_training_rows=0;self.last_active_parameters=0

    def forward(self,h,slots=None):
        if not self.stateful:slots=None
        indices,gates=self.router(h,slots);b,t,d=h.shape
        selected=h[:,:,None,:].expand(b,t,2,d).reshape(-1,d)
        if slots is not None:
            batch=torch.arange(b,device=h.device)[:,None,None].expand_as(indices)
            selected=selected+self.state_input(slots[batch,indices].reshape(-1,slots.shape[-1]))
        # There is no [B,T,16,hidden] main-FFN activation: only B*T*2 rows.
        output=self.shared(selected)
        ids=indices.reshape(-1);low=torch.bmm(selected[:,None],self.down[ids]).squeeze(1)
        output=output+torch.bmm(F.silu(low)[:,None],self.up[ids]).squeeze(1)
        self.executed_rows=selected.shape[0]
        if self.training:self.total_training_rows+=self.executed_rows
        self.last_active_parameters=sum(p.numel() for p in self.shared.parameters())+2*(self.down[0].numel()+self.up[0].numel())
        if self.state_input is not None:self.last_active_parameters+=self.state_input.weight.numel()
        return (output.reshape(b,t,2,d)*gates[:,:,:,None]).sum(2)
