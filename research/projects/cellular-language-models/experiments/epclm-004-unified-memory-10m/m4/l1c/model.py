"""Contextual E5 adapter. Frozen branch and memory implementations are reused."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'fast-screen'))
from adapter import Factorial, MASKS, tensor_digest

A_LENGTH=272
B_LENGTH=240


class Contextual(Factorial):
    def context(self,ids):
        return self.swa(ids)

    def read_memory(self,context,cell):
        _,_,query=self.memory.project(context)
        mode='meta_train' if torch.is_grad_enabled() else 'read_only'
        return self.memory.recall(query,cell,mode)

    def fuse(self,embedding,swa,ssm,memory,disabled):
        scores=self.gate(torch.cat((embedding,swa,ssm,memory),dim=-1))
        mask=torch.tensor([name in disabled for name in self.branch_names],device=scores.device)
        gates=scores.masked_fill(mask,float('-inf')).softmax(-1)
        h=embedding+sum(gates[...,i:i+1]*x for i,x in enumerate((swa,ssm,memory)))
        h=h+self.post_mlp(self.post_norm(h))
        return self.lm_head(self.final_norm(h)),gates

    def write_a(self,ids,diagnose=False):
        if 'memory' in self.disabled and not diagnose:
            return None,None,None
        context=self.context(ids)
        zero=torch.zeros_like(context)
        cell=None
        memory=zero
        if 'memory' not in self.disabled:
            mode='meta_train' if self.training and torch.is_grad_enabled() else 'online_adapt'
            recalled,cell,_=self.memory.stream(context,None,mode,self.chunk)
            if diagnose:
                memory=self.memory_norm(self.memory_output(recalled))
        gates=None
        if diagnose:
            ssm=zero if 'ssm' in self.disabled else self.ssm(ids)
            _,gates=self.fuse(self.token_embedding(ids),context,ssm,memory,self.disabled)
        return cell,gates,context

    def read_b(self,ids,cell,disabled=None,cache=False,swa_state=None,ssm_state=None):
        disabled=self.disabled if disabled is None else disabled
        if cache:
            context,swa_state=self.swa.cached(ids,swa_state)
        else:
            context=self.context(ids)
        zero=torch.zeros_like(context)
        ssm=zero
        if 'ssm' not in disabled:
            if cache:
                ssm,ssm_state=self.ssm.cached(ids,ssm_state)
            else:
                ssm=self.ssm(ids)
        memory=zero
        if 'memory' not in disabled:
            if cell is None:
                raise ValueError('Session B requires the carried Session A Cell')
            memory=self.memory_norm(self.memory_output(self.read_memory(context,cell)))
        logits,gates=self.fuse(self.token_embedding(ids),context,ssm,memory,disabled)
        return logits,gates,swa_state,ssm_state

    def forward(self,ids):
        if ids.ndim!=2 or ids.shape[1]!=A_LENGTH+B_LENGTH:
            raise ValueError('L1-C expects exactly 272+240 input tokens')
        a,b=ids.split((A_LENGTH,B_LENGTH),dim=1)
        cell,_,_=self.write_a(a)
        # The original trainer consumes aligned logits and masks all Session A labels.
        # The A positions here are placeholders and are excluded from supervision.
        logits=self.read_b(b,cell)[0]
        return F.pad(logits,(0,0,A_LENGTH,0))


def cell_hash(cell):
    return None if cell is None else dict(fast=tensor_digest(cell.fast),
                                          momentum=tensor_digest(cell.momentum),step=cell.update_step)
