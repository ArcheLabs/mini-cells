"""E5: parallel frozen-M2 SWA/Mamba1 primitives plus the frozen-M3 Memory rule."""
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
from torch import nn
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'m3'),str(Path(__file__).resolve().parents[1]/'m2')]
from baselines import E1, RecurrentBaseline, Config, runtime_memory
from memory import NeuralMemory, MODES, state_bytes
from mag_model import MAGLite
import json
from native_clm_runtime import RMSNorm, SwiGLU
from data import digest


@dataclass
class Runtime:
    owner: int
    versions: tuple
    session: int
    disabled: tuple
    batch: int
    swa: object
    ssm: object


class ThreeWayFusion(nn.Module):
    model_id='mamba1_swa_titans_single_memory'
    branch_names=('swa','ssm','memory')

    def __init__(self,config,ssm_layers=8,state_size=20,memory_dim=32,memory_hidden=64,chunk=16,seed=41017):
        super().__init__();self.config=Config(**config);self.chunk=chunk;self.memory_mode='meta_train'
        # Reuse complete differentiable branch forwards. Identity heads expose their
        # final normalized features, without duplicating attention or selective scan.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);swa=E1(self.config)
            self.token_embedding=swa.token_embedding;self.swa=swa
            self.ssm=RecurrentBaseline(Config(**{**config,'layers':ssm_layers}),'E2',state_size,seed)
            self.ssm.token_embedding=self.token_embedding
            self.swa.lm_head=nn.Identity();self.ssm.lm_head=nn.Identity()
            self.memory=NeuralMemory(self.config.dim,memory_dim,memory_hidden)
            self.memory_output=nn.Linear(memory_dim,self.config.dim,bias=False)
            self.memory_norm=nn.LayerNorm(self.config.dim)
            self.gate=nn.Linear(4*self.config.dim,3)
            self.post_norm=RMSNorm(self.config.dim);self.post_mlp=SwiGLU(self.config.dim,self.config.ffn)
            self.final_norm=RMSNorm(self.config.dim)
            self.lm_head=nn.Linear(self.config.dim,self.config.vocab,bias=False)
            self.lm_head.weight=self.token_embedding.weight
        # Copy shape-compatible untrained E4 parameters; never load trained M3 weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            canonical=json.loads((Path(__file__).resolve().parents[1]/'m3/configs/smoke.json').read_text())['model']
            reference_config=canonical if config['vocab']==8192 and config['dim']==256 else config
            reference=MAGLite(Config(**reference_config),memory_dim,memory_hidden,chunk)
            with torch.no_grad():
                self.token_embedding.weight.copy_(reference.token_embedding.weight)
                shared=reference.state_dict()
                for name,p in self.swa.named_parameters():
                    if name in shared and p.shape==shared[name].shape:p.copy_(shared[name])
                self.memory.load_state_dict(reference.memory.state_dict())
                self.memory_output.load_state_dict(reference.memory_output.state_dict())
                self.memory_norm.load_state_dict(reference.memory_norm.state_dict())
        self.last_cell=None;self.events=[];self.session=0

    def set_mode(self,mode):
        if mode not in MODES:raise ValueError('unknown memory mode')
        self.memory_mode=mode;self.requires_grad_(mode=='meta_train');return self

    def reset_runtime(self):
        self.last_cell=None;self.session+=1

    def forward_state(self,ids,cell=None,disabled=(),runtime=None,cache=False):
        disabled=set(disabled)
        if disabled-set(self.branch_names) or len(disabled)==3:raise ValueError('at least one known branch must remain enabled')
        if runtime is not None and not cache:raise ValueError('runtime is inference-only')
        if cache:
            if self.training or self.memory_mode=='meta_train' or torch.is_grad_enabled():
                raise ValueError('cached chunks require eval + no_grad + online_adapt/read_only')
            if torch.is_inference_mode_enabled() and self.memory_mode=='online_adapt':
                raise ValueError('online Memory updates need no_grad, not inference_mode')
            if ids.shape[1]%self.chunk:raise ValueError('cached calls must align to complete Memory chunks')
            if runtime and (runtime.owner!=id(self) or runtime.session!=self.session or runtime.versions!=tuple(p._version for p in self.parameters())
                            or runtime.disabled!=tuple(sorted(disabled)) or runtime.batch!=ids.shape[0]):
                raise ValueError('runtime owner/session/static weights or ablation differ')
        h=self.token_embedding(ids);zero=torch.zeros_like(h);a=s=zero;new_swa=new_ssm=None
        if 'swa' not in disabled:
            if cache:a,new_swa=self.swa.cached(ids,runtime.swa if runtime else None)
            else:a=self.swa(ids)
        if 'ssm' not in disabled:
            if cache:s,new_ssm=self.ssm.cached(ids,runtime.ssm if runtime else None)
            else:s=self.ssm(ids)
        m=zero;diagnostics=[];state=cell
        if 'memory' not in disabled:
            reads,state,diagnostics=self.memory.stream(h,cell,self.memory_mode,self.chunk)
            m=self.memory_norm(self.memory_output(reads));self.last_cell=state.detached()
        gate_input=torch.cat((h,a,s,m),-1);scores=self.gate(gate_input)
        mask=torch.tensor([name in disabled for name in self.branch_names],device=ids.device)
        weights=scores.masked_fill(mask,float('-inf')).softmax(-1)
        z=(weights[...,0:1]*a+weights[...,1:2]*s+weights[...,2:3]*m)
        u=h+z;output=u+self.post_mlp(self.post_norm(u));logits=self.lm_head(self.final_norm(output))
        if torch.is_grad_enabled() and self.training and self.memory_mode=='meta_train' and 'memory' not in disabled:
            initial=cell.fast if cell else {k:p.detach().unsqueeze(0).expand_as(state.fast[k]) for k,p in self.memory.initial.items()}
            self.events.append({'input_sha256':digest(ids.detach().cpu().tolist()),'inner_updates':len(diagnostics),
                'fast_delta':sum(float((state.fast[k].detach()-initial[k].detach()).square().sum()) for k in initial)**.5 if state else 0.,
                'momentum_norm':sum(float(v.detach().square().sum()) for v in state.momentum.values())**.5 if state else 0.,
                'gate_mean':weights.detach().mean((0,1)).cpu().tolist()})
        new_runtime=Runtime(id(self),tuple(p._version for p in self.parameters()),self.session,tuple(sorted(disabled)),ids.shape[0],new_swa,new_ssm) if cache else None
        return logits,state,weights,new_runtime

    def forward(self,ids):return self.forward_state(ids)[0]

    def resources(self,cell=None,runtime=None):
        named=dict(self.named_parameters());total=sum(p.numel() for p in named.values())
        return {'static_parameters':total,'static_by_module':{name:sum(p.numel() for n,p in named.items() if n.split('.')[0]==name) for name in sorted({n.split('.')[0] for n in named})},
            'cell':state_bytes(cell) if cell else None,
            'swa_runtime':runtime_memory(runtime.swa,runtime.batch) if runtime and runtime.swa else None,
            'ssm_runtime':runtime_memory(runtime.ssm,runtime.batch) if runtime and runtime.ssm else None}
