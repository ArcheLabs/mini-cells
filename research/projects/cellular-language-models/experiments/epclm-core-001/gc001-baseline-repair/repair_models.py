"""Matched event access and active-parameter repair; historical Core is unchanged."""
from dataclasses import dataclass
import torch
from torch import nn
from model import Core,Config,check_numerics
from native_clm_runtime import RMSNorm

ARMS=('C1-BROADCAST','IC-ALL-SELF','F0-PM-BROADCAST')


@dataclass(frozen=True)
class RepairConfig(Config):
    variant: str='C1-BROADCAST'
    event_shift: int=8
    rank: int=0


class BroadcastCore(Core):
    def __init__(self,config):
        super().__init__(config)
        indices=(torch.arange(config.dim)[None]-config.event_shift*torch.arange(config.cells)[:,None])%config.dim
        self.register_buffer('_event_coordinates',indices,persistent=False)

    def inject(self,state,token_id):
        embedding=self.embedding(torch.as_tensor(token_id,device=state.device).reshape(-1))
        if len(embedding)!=len(state):raise ValueError('one event per batch lane')
        return state+embedding[:,self._event_coordinates]


class FlatMatched(nn.Module):
    """No allocated dead cellular projections; every parameter is used."""
    def __init__(self,config):
        super().__init__();self.config=config;c=config;n=c.cells*c.dim
        self.embedding=nn.Embedding(c.vocab,c.dim)
        self.norm=RMSNorm(c.dim);self.pool=nn.Linear(n,c.dim,bias=False);self.head=nn.Linear(c.dim,c.vocab,bias=False)
        self.flat_norm=RMSNorm(n);self.down=nn.Linear(n,c.rank,bias=False);self.up=nn.Linear(c.rank,n,bias=False)
        indices=(torch.arange(c.dim)[None]-c.event_shift*torch.arange(c.cells)[:,None])%c.dim
        self.register_buffer('_event_coordinates',indices,persistent=False)

    init_state=Core.init_state
    reset=Core.reset
    inject=BroadcastCore.inject
    read=Core.read
    step=Core.step

    def evolve(self,state,steps=None,communication=True,trace=None):
        steps=self.config.steps if steps is None else steps
        if steps<0 or communication is not True or trace is not None:raise ValueError('invalid flat execution mode')
        for _ in range(steps):
            flat=state.flatten(1)
            state=(flat+self.config.alpha*self.up(torch.nn.functional.silu(self.down(self.flat_norm(flat))))).reshape_as(state)
            check_numerics(state.detach().abs().amax()<=self.config.max_state_abs,'flat matched state outside numerical range')
        return state


def flat_parameters(rank,vocab=2048,cells=16,dim=128):
    n=cells*dim
    return 2*vocab*dim+dim+n*dim+n+2*n*rank


def select_rank(target=966784,vocab=2048,cells=16,dim=128):
    # Integer structural sizing, not a data-dependent hyperparameter search. Ties -> smaller rank.
    rank=min(range(1,cells*dim+1),key=lambda r:(abs(flat_parameters(r,vocab,cells,dim)-target),r))
    actual=flat_parameters(rank,vocab,cells,dim)
    return {'rank':rank,'parameters':actual,'target':target,'relative_error':abs(actual/target-1),'within_tolerance':abs(actual/target-1)<=.01}


def build(variant,rank=None):
    if variant not in ARMS:raise ValueError('unknown repair architecture')
    if variant=='F0-PM-BROADCAST':
        rank=select_rank()['rank'] if rank is None else rank
        return FlatMatched(RepairConfig(variant=variant,arm='F0',rank=rank))
    return BroadcastCore(RepairConfig(variant=variant,arm='C1' if variant=='C1-BROADCAST' else 'C0'))


def architecture(model):
    c=model.config;n=c.cells*c.dim
    allocated=sum(p.numel() for p in model.parameters())
    if isinstance(model,FlatMatched):recurrent=4*n*c.rank
    else:recurrent=22*c.cells*c.dim*c.dim
    return {'allocated_parameters':allocated,'active_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'state_dimensions':n,'state_bytes':n*4,'event_coordinate_buffer_bytes':sum(v.numel()*v.element_size() for v in model.buffers()),'weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'rank':c.rank or 'NOT_APPLICABLE','estimated_forward_linear_flops':c.steps*recurrent+2*n*c.dim+2*c.dim*c.vocab,
            'flops_scope':'linear multiply/add estimate; excludes normalization, activation and memory; not compute matched',
            'event_rule':'e[(j-8*i) mod 128], equivalent to roll(e,8*i)',
            'communication':c.variant=='C1-BROADCAST','flat_storage_has_cell_semantics':False if isinstance(model,FlatMatched) else True}
