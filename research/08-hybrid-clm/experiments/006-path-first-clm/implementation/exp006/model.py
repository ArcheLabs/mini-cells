"""One hard, replayable trajectory per position through a globally shared pool."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .memory import InputMemory
from .cells import FeedForward,ContextRead
from .controller import Controller
@dataclass
class Rollout:
    logits: torch.Tensor
    actions: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor
    memory: object
    states: torch.Tensor
    balance: torch.Tensor | None = None
    edge_mode: str = 'normal'
    training_mode: bool = False
    context_events: object = None
    context_mode: str = 'normal'

def initialize(model,seed):
    for name,p in model.named_parameters():
        g=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'exp006:{seed}:{name}'.encode()).digest()[:8],'little')%(2**63))
        with torch.no_grad():
            if p.ndim>=2:p.normal_(std=0.02,generator=g)
            elif name.endswith('weight'):p.fill_(1)
            else:p.zero_()

class PathFirstLM(nn.Module):
    def __init__(self,c,arm,seed=0):
        super().__init__()
        if arm not in ('R0','R1','R2'):raise ValueError('R3 STOP is a separate unregistered experiment')
        if c['memory_mode']!='static_input_readonly':raise ValueError('path-derived memory invalidates per-position reward')
        self.config=dict(c);self.arm=arm;d=c['width']
        self.embedding=nn.Embedding(c['vocab_size'],d);self.position=nn.Embedding(c['max_sequence_length'],d)
        self.init=nn.Linear(d,d,bias=False);self.memory=InputMemory(d);self.norm=nn.LayerNorm(d);self.final=nn.LayerNorm(d)
        self.cells=nn.ModuleList(FeedForward(d,c['cell_hidden']) for _ in range(c['cell_count']))
        self.context=ContextRead(d,c['heads']);self.controller=Controller(c,arm);initialize(self,seed)
    def forward(self,tokens,*,policy='greedy',replay=None,generator=None,past=None,edge_mode='normal',edge_permutation=None,capture_context_states=False,context_mode='normal'):
        if policy not in ('greedy','sampled'):raise ValueError('unknown policy')
        if edge_mode not in ('normal','zero','shuffle','permutation'):raise ValueError('unknown connection intervention')
        if context_mode not in ('normal','zero_replay') or context_mode=='zero_replay' and replay is None:raise ValueError('context intervention requires action replay')
        context_events=[]
        b,n=tokens.shape;offset=0 if past is None else past.key.shape[1]
        if offset+n>self.config['max_sequence_length']:raise ValueError('position budget exceeded')
        positions=torch.arange(offset,offset+n,device=tokens.device)
        z=self.embedding(tokens)+self.position(positions)[None]
        memory=self.memory(z,past);h=self.init(z).reshape(-1,self.config['width'])
        batch_ids=torch.arange(b,device=h.device).repeat_interleave(n);pos=positions.repeat(b)
        previous=torch.full((b*n,),self.config['cell_count']+1,device=h.device,dtype=torch.long)
        acts,lps,ents,vals=[],[],[],[]
        if replay is not None and replay.shape!=(b,n,self.config['steps']):raise ValueError('replay shape mismatch')
        for t in range(self.config['steps']):
            scores,value=self.controller(h,previous,t,edge_mode,edge_permutation);logq=scores.log_softmax(-1);q=logq.exp()
            action=(replay[:,:,t].reshape(-1) if replay is not None else
                    torch.multinomial(q,1,generator=generator).squeeze(-1) if policy=='sampled' else scores.argmax(-1))
            if torch.any(action<0) or torch.any(action>self.config['cell_count']):raise ValueError('invalid action; STOP is disabled')
            normalized=self.norm(h);delta=torch.zeros_like(h)
            # Only selected modules run; a per-operation grouping is an auditable reference.
            for a in range(self.config['cell_count']+1):
                rows=torch.where(action==a)[0]
                if rows.numel()==0:continue
                if a==self.config['cell_count']:
                    if context_mode=='zero_replay':update=torch.zeros_like(normalized[rows])
                    else:update=self.context(normalized[rows],memory,batch_ids[rows],pos[rows])
                    if capture_context_states:context_events.append({'step':t,'rows':rows.detach(),
                        'before':h[rows].detach(),'after':(h[rows]+self.config['alpha']*update.to(h.dtype)).detach()})
                else:update=self.cells[a](normalized[rows])
                delta=delta.index_copy(0,rows,update.to(delta.dtype))
            h=h+self.config['alpha']*delta
            acts.append(action);lps.append(logq.gather(1,action[:,None]).squeeze(1));ents.append(-(q*logq).sum(-1));vals.append(value);previous=action
        logits=F.linear(self.final(h).reshape(b,n,-1),self.embedding.weight)
        stack=lambda x:torch.stack(x,-1).reshape(b,n,-1)
        return Rollout(logits,stack(acts),stack(lps),stack(ents),stack(vals),memory,h.reshape(b,n,-1),edge_mode=edge_mode,training_mode=self.training,context_events=context_events,context_mode=context_mode)
    def prefill(self,tokens,**kwargs):return self(tokens,**kwargs)
    def decode_step(self,token,memory,**kwargs):return self(token.reshape(-1,1),past=memory,**kwargs)
    def parameter_breakdown(self):
        total=sum(p.numel() for p in self.parameters());ctrl=sum(p.numel() for p in self.controller.parameters());cells=sum(p.numel() for p in self.cells.parameters())
        return {'total':total,'controller_and_critic':ctrl,'cells':cells,'shared':total-ctrl-cells}
