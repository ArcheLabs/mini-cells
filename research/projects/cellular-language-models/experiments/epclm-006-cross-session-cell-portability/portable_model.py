"""E3 Session backbone with a reset-only canonical Cell interface."""
import json
from pathlib import Path
import sys

import torch
from torch import nn

BASE=Path(__file__).resolve().parent
EP4=BASE.parent/'epclm-004-unified-memory-10m'
sys.path[:0]=[str(EP4/'m4/fast-screen'),str(EP4/'m2'),str(EP4/'m3'),str(EP4)]
from baselines import build
from memory import NeuralMemory
from adapter import tensor_digest

P=json.loads((BASE/'config.json').read_text())
M2=json.loads((EP4/'m2/configs/protocol.json').read_text())
E3=M2['arms']['E3']
SOURCE=P['bindings']*P['segment_tokens']
TARGET=P['target_prefix_tokens']+P['queries']*P['segment_tokens']
POSITIONS=tuple(P['target_prefix_tokens']+i*P['segment_tokens']+P['segment_tokens']-1
                for i in range(P['queries']))


class PortableModel(nn.Module):
    def __init__(self,fixture=False):
        super().__init__()
        config=E3['model']
        state=E3['state_size']
        if fixture:
            config=dict(vocab=1024,dim=16,layers=2,heads=2,kv_heads=2,ffn=32,window=32)
            state=4
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(P['model_seed'])
            self.backbone=build('E3',config,P['model_seed'],state_size=state)
            self.lm_head=self.backbone.lm_head
            self.backbone.lm_head=nn.Identity()
            dim=config['dim']
            memory_dim=4 if fixture else 32
            self.memory=NeuralMemory(dim,memory_dim,8 if fixture else 64)
            self.memory_out=nn.Linear(memory_dim,dim,bias=False)
            self.gate=nn.Linear(dim+memory_dim,1)
        self.model_config=config

    @property
    def token_embedding(self):
        return self.backbone.token_embedding

    def encode_cell_segment(self,ids):
        if ids.ndim!=2 or not 1<=ids.shape[1]<=P['segment_tokens']:
            raise ValueError('Canonical segments must be bounded [batch,time] tensors')
        return self.backbone(ids)[:,-1]

    def encode_packets(self,packets):
        batch,count,length=packets.shape
        return self.encode_cell_segment(packets.flatten(0,1)).reshape(batch,count,-1)

    def acquire(self,segments,cell=None):
        hidden=self.encode_packets(segments)
        k,v,_=self.memory.project(hidden)
        mode='meta_train' if self.training and torch.is_grad_enabled() else 'online_adapt'
        if cell is None:
            cell=self.memory.create(len(segments),mode)
        for keys,values in zip(k.split(P['cell_chunk'],1),v.split(P['cell_chunk'],1)):
            cell,_=self.memory.learn(keys,values,cell,mode)
        return cell

    def query(self,hidden,packets,cell,keep=None):
        if cell is None:
            return self.lm_head(hidden)
        _,_,q=self.memory.project(self.encode_packets(packets))
        mode='meta_train' if torch.is_grad_enabled() else 'read_only'
        memory=self.memory.recall(q,cell,mode)
        delta=self.gate(torch.cat((hidden,memory),dim=-1)).sigmoid()*self.memory_out(memory)
        if keep is not None:
            delta=delta*keep[:,None,None]
        return self.lm_head(hidden+delta)

    def query_packets(self,target):
        return target[:,P['target_prefix_tokens']:].reshape(len(target),P['queries'],P['segment_tokens'])

    def forward(self,ids):
        if ids.ndim!=2 or ids.shape[1]!=SOURCE+TARGET:
            raise ValueError('Invalid episodic input shape')
        source,target=ids.split((SOURCE,TARGET),dim=1)
        segments=source.reshape(len(ids),P['bindings'],P['segment_tokens'])
        cell=self.acquire(segments)
        hidden=self.backbone(target)[:,POSITIONS]
        keep=None
        if self.training:
            # Always include one Cell-enabled lane. For batch four, the
            # remaining lanes have 0.20 dropout, or 0.15 averaged over lanes.
            keep=torch.rand(len(ids),device=ids.device)>=P['cell_dropout']*len(ids)/max(1,len(ids)-1)
            keep[0]=True
        logits=self.query(hidden,self.query_packets(target),cell,keep)
        positions=torch.tensor([SOURCE+p for p in POSITIONS],device=ids.device)
        return logits.new_zeros(len(ids),SOURCE+TARGET,logits.shape[-1]).index_copy(1,positions,logits)


def state_hash(cell):
    return dict(fast=tensor_digest(cell.fast),momentum=tensor_digest(cell.momentum),step=cell.update_step)
