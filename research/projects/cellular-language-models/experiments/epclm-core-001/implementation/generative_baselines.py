"""Only the missing center-only baseline; reuse Core F0 and Native T1 unchanged."""
import torch
from torch import nn
from model import Core,Config,check_numerics
from native_clm_runtime import ModernTransformerLM,model_config,T1_NAME


class NoCommunicationCore(Core):
    def __init__(self,config):
        super().__init__(config)
        for projection in (self.left,self.right):
            for p in projection.parameters():p.requires_grad_(False)

    def evolve(self,state,steps=None,communication=True,trace=None):
        if communication is not True or trace is not None:raise ValueError('center-only baseline has a fixed transition')
        steps=self.config.steps if steps is None else steps
        if steps<0:raise ValueError('negative evolution count')
        for _ in range(steps):
            normalized=self.norm(state)
            delta=self.update(torch.cat((normalized,self.center(normalized)),-1))
            state=state+self.config.alpha*delta
            check_numerics(state.detach().abs().amax()<=self.config.max_state_abs,'center-only state outside numerical range')
        return state


class TransformerReference(nn.Module):
    """Existing Native T1, bounded context replay; explicitly no incremental KV cache."""
    def __init__(self,vocab=2048,context=128):
        super().__init__();self.native=ModernTransformerLM(vocab);self.context=context
        self.reference_config={**model_config(T1_NAME,vocab),'context_length':context,'kv_cache':False}
        self.config=Config(vocab=vocab,cells=1,dim=context+1,steps=1,arm='T1')

    @property
    def embedding(self):return self.native.token_embedding

    def init_state(self,batch_size=1):return self.embedding.weight.new_zeros(batch_size,1,self.context+1)

    def reset(self,state,mask=None):
        return torch.zeros_like(state) if mask is None else torch.where(mask[:,None,None],torch.zeros_like(state),state)

    def step(self,state,token_id,evolve_steps=None,communication=True):
        if evolve_steps not in (None,1) or communication is not True:raise ValueError('reference has one transformer forward per token')
        next_state=state.clone();tokens=torch.as_tensor(token_id,device=state.device).reshape(-1)
        if len(tokens)!=len(state):raise ValueError('one event per lane')
        for lane,token in enumerate(tokens):
            length=min(int(state[lane,0,-1]),self.context)
            if length<self.context:next_state[lane,0,length]=token;next_state[lane,0,-1]=length+1
            else:
                next_state[lane,0,:self.context-1]=state[lane,0,1:self.context]
                next_state[lane,0,self.context-1]=token
        return next_state,self.read(next_state)

    def read(self,state):
        # Per-lane lengths prevent padding tokens from becoming hidden context.
        return torch.cat([self.native(row[:int(row[-1])].long()[None])[:,-1] for row in state[:,0]],0)

    def evolve(self,state,steps=None,communication=True,trace=None):
        if steps not in (None,0):raise ValueError('extra event-free evolution is not defined for T1 reference')
        return state


def freeze(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def linear_flops(model):
    c=model.config
    if c.arm=='T1':return {'status':'NOT_MEASURED','context':model.context,'reason':'context-dependent reference cost; actual time reported'}
    # Core dense linear multiply/add estimates; excludes norms, activations and memory.
    if isinstance(model,NoCommunicationCore):recurrent=18*c.cells*c.dim*c.dim
    elif c.arm=='F0':recurrent=4*c.cells*c.dim*c.dim
    else:recurrent=22*c.cells*c.dim*c.dim
    return {'estimated_forward_linear_flops_per_token':c.steps*recurrent+2*c.cells*c.dim*c.dim+2*c.dim*c.vocab,
            'scope':'analytical linear multiply/add only; not measured GPU FLOPs or compute matching'}
