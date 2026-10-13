"""Only token/position input determines immutable context K/V."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
@dataclass
class Memory:
    key: torch.Tensor
    value: torch.Tensor
class InputMemory(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width);self.key=nn.Linear(width,width,bias=False);self.value=nn.Linear(width,width,bias=False)
    def forward(self,z,past=None):
        z=self.norm(z);k,v=self.key(z),self.value(z)
        if past is not None:k=torch.cat((past.key,k),1);v=torch.cat((past.value,v),1)
        return Memory(k,v)
