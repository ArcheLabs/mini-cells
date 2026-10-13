"""R2 adds only a low-rank predecessor prior to the exact R1 scorer."""
from __future__ import annotations
import torch
from torch import nn
class Controller(nn.Module):
    def __init__(self,c,arm):
        super().__init__();self.arm=arm;self.actions=c['cell_count']+1;self.steps=c['steps']
        self.step=nn.Embedding(self.steps,c['step_width'])
        if arm=='R0':self.static=nn.Parameter(torch.zeros(self.steps,self.actions))
        else:self.scorer=nn.Sequential(nn.Linear(c['width']+c['step_width'],c['router_hidden']),nn.Tanh(),nn.Linear(c['router_hidden'],self.actions))
        if arm=='R2':
            self.source=nn.Embedding(self.actions+1,c['edge_rank']);self.target=nn.Embedding(self.actions,c['edge_rank'])
        self.critic=nn.Sequential(nn.Linear(c['width']+c['step_width'],32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,h,previous,t,edge_mode='normal'):
        step=self.step.weight[t].expand(h.shape[0],-1)
        scores=self.static[t].expand(h.shape[0],-1) if self.arm=='R0' else self.scorer(torch.cat((h,step),-1))
        if self.arm=='R2' and edge_mode!='zero':
            ids=previous if edge_mode=='normal' else (previous+1)%(self.actions+1)
            scores=scores+(self.source(ids).float()@self.target.weight.float().T)*self.source.embedding_dim**-0.5
        value=self.critic(torch.cat((h.detach(),step.detach()),-1)).squeeze(-1) if self.training else h.new_zeros(h.shape[0])
        return scores.float(),value
