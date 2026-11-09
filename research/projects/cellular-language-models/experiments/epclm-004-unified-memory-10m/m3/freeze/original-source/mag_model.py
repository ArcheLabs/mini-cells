"""Single shared Neural Memory fused with the unchanged M2 E1 SWA."""
from pathlib import Path
import sys
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'m2'))
from baselines import E1, Config
from memory import NeuralMemory, MODES
from data import digest


class MAGLite(E1):
    model_id='mag_lite_single_memory'

    def __init__(self,config,memory_dim=32,hidden=64,chunk=16):
        super().__init__(config)
        self.memory=NeuralMemory(config.dim,memory_dim,hidden)
        self.memory_output=nn.Linear(memory_dim,config.dim,bias=False)
        self.memory_norm=nn.LayerNorm(config.dim)
        self.gate=nn.Linear(config.dim,1)
        self.chunk=chunk;self.memory_mode='meta_train'
        self.last_cell=None;self.events=[]

    def set_mode(self,mode):
        if mode not in MODES:raise ValueError('unknown memory mode')
        self.memory_mode=mode
        self.requires_grad_(mode=='meta_train')
        return self

    def reset_runtime(self):
        # Caller-owned persistent Cells are never erased here.
        self.last_cell=None

    def forward_state(self,ids,cell=None,disable=None):
        if disable not in (None,'memory','swa'):raise ValueError('unknown branch ablation')
        record=torch.is_grad_enabled() and self.training and self.memory_mode=='meta_train'
        h=self.token_embedding(ids)
        if disable=='memory':
            return super().forward(ids),cell,h.new_zeros((*ids.shape,1))
        reads,state,diagnostics=self.memory.stream(h,cell,self.memory_mode,self.chunk)
        memory_logits=self.lm_head(self.memory_norm(self.memory_output(reads)))
        gate=torch.sigmoid(self.gate(h)) if disable!='swa' else h.new_ones((*ids.shape,1))
        # The tied linear head distributes over this scalar gated hidden fusion.
        logits=memory_logits if disable=='swa' else (1-gate)*super().forward(ids)+gate*memory_logits
        self.last_cell=state.detached()
        if record:
            initial=cell.fast if cell else {k:p.detach().unsqueeze(0).expand_as(state.fast[k]) for k,p in self.memory.initial.items()}
            self.events.append({'input_sha256':digest(ids.detach().cpu().tolist()),'inner_updates':len(diagnostics),
                'fast_delta':sum(float((state.fast[k].detach()-initial[k].detach()).square().sum()) for k in initial)**.5,
                'momentum_norm':sum(float(v.detach().square().sum()) for v in state.momentum.values())**.5,
                'gate_min':float(gate.detach().min()),'gate_max':float(gate.detach().max())})
        return logits,state,gate

    def cached(self,*args,**kwargs):
        raise ValueError('E4 does not inherit SWA-only cached decoding; use explicit Cell recall/forward_state')

    def forward(self,ids):return self.forward_state(ids)[0]
