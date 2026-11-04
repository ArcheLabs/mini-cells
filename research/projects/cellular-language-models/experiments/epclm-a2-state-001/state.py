"""Shared learnable slots, bridge, gate and FP32 GRUCell for all interaction arms."""
import torch
from torch import nn


class StateBank(nn.Module):
    def __init__(self,dim=188):
        super().__init__();self.slot=nn.Parameter(torch.randn(16,128)*.02)
        self.bridge=nn.Linear(128,dim,bias=False);self.gate=nn.Parameter(torch.tensor([-2.]))
        self.update=nn.GRUCell(dim,128);self.last={}

    def init(self,batch):return self.slot[None].expand(batch,-1,-1).clone().float()
    def slots(self,state):return state

    def write(self,state,features,mask=None):
        mask=torch.ones(features.shape[:2],device=features.device,dtype=torch.bool) if mask is None else mask
        with torch.autocast(device_type=state.device.type,enabled=False):
            h=features.float();count=mask.sum(1);summary=(h*mask[:,:,None]).sum(1)/count.clamp_min(1)[:,None]
            inputs=summary[:,None].expand(-1,16,-1).reshape(-1,summary.shape[-1]);previous=state.float().reshape(-1,128)
            # Explicit GRU gates for the update-gate diagnostic; same nn.GRUCell parameters.
            ih=torch.nn.functional.linear(inputs,self.update.weight_ih,self.update.bias_ih)
            hh=torch.nn.functional.linear(previous,self.update.weight_hh,self.update.bias_hh)
            ir,iz,inn=ih.chunk(3,-1);hr,hz,hn=hh.chunk(3,-1)
            reset=(ir+hr).sigmoid();gate=(iz+hz).sigmoid();candidate=(inn+reset*hn).tanh()
            updated=(candidate+gate*(previous-candidate)).reshape_as(state)
            updated=torch.where((count>0)[:,None,None],updated,state.float())
            with torch.no_grad():self.last={'update_gate':gate.detach().mean(),'update_magnitude':(updated-state).detach().norm(dim=-1).mean(),
                                           'state_norm':updated.detach().norm(dim=-1).mean()}
        return updated
