"""Functional event transitions; training and inference use exactly the same core."""
from dataclasses import dataclass, fields
import math
import torch
from torch import nn
from native_clm_runtime import RMSNorm, SwiGLU

@dataclass(frozen=True)
class Config:
    vocab: int = 2048
    cells: int = 16
    dim: int = 128
    steps: int = 4
    alpha: float = 0.1
    arm: str = 'C1'
    window: int = 4
    max_state_abs: float = 10000.

def config_from_protocol(protocol, arm='C1'):
    architecture = dict(protocol['architecture'])
    if architecture.pop('topology') != 'ring-radius1':
        raise ValueError('unsupported protocol topology')
    architecture['steps'] = architecture.pop('K')
    expected = {field.name for field in fields(Config)} - {'arm'}
    if set(architecture) != expected:
        raise ValueError('protocol architecture fields differ: ' + str(sorted(set(architecture) ^ expected)))
    return Config(arm=arm, **architecture)


def check_numerics(valid, message):
    # CUDA assertions stay on the device; CPU retains an immediate reference error.
    if valid.is_cuda:
        torch._assert_async(valid, message)
    elif not valid:
        raise FloatingPointError(message)

class Core(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = c = config
        if c.arm not in ('C0', 'C1', 'F0', 'L0') or min(c.cells,c.dim,c.steps,c.window) < 1 or not 0 < c.alpha <= 1 or not math.isfinite(c.max_state_abs) or c.max_state_abs <= 0:
            raise ValueError('invalid core configuration')
        self.embedding = nn.Embedding(c.vocab,c.dim)
        self.norm = RMSNorm(c.dim)
        self.left = nn.Linear(c.dim,c.dim,bias=False)
        self.center = nn.Linear(c.dim,c.dim,bias=False)
        self.right = nn.Linear(c.dim,c.dim,bias=False)
        self.update = nn.Sequential(nn.Linear(2*c.dim,c.dim,bias=False),SwiGLU(c.dim,2*c.dim))
        self.pool = nn.Linear(c.cells*c.dim,c.dim,bias=False)
        self.head = nn.Linear(c.dim,c.vocab,bias=False)
        if c.arm == 'F0':
            # Full-capacity flat state, low-rank dense recurrent mixing; report its actual cost.
            self.flat = nn.Sequential(RMSNorm(c.cells*c.dim),nn.Linear(c.cells*c.dim,c.dim,bias=False),
                                      nn.SiLU(),nn.Linear(c.dim,c.cells*c.dim,bias=False))

    def init_state(self,batch_size=1):
        return self.embedding.weight.new_zeros(batch_size,self.config.cells,self.config.dim)

    def reset(self,state,mask=None):
        if mask is None:
            return torch.zeros_like(state)
        return torch.where(mask[:,None,None],torch.zeros_like(state),state)

    def inject(self,state,token_id):
        event = self.embedding(torch.as_tensor(token_id,device=state.device).reshape(-1))
        if event.shape[0] != state.shape[0]:
            raise ValueError('one event per batch lane is required')
        if self.config.arm == 'L0':
            # FIFO contains at most window token embeddings, never a recurrent history.
            width = min(self.config.window,self.config.cells)
            return torch.cat((event[:,None],state[:,:width-1],torch.zeros_like(state[:,width:])),1)
        return state + torch.cat((event[:,None],torch.zeros_like(state[:,1:])),1)

    def evolve(self,state,steps=None,communication=True,trace=None):
        c = self.config
        steps = c.steps if steps is None else steps
        if steps < 0:
            raise ValueError('negative evolution count')
        for _ in range(steps):
            if c.arm == 'F0':
                delta = self.flat(state.flatten(1)).reshape_as(state)
                message = torch.zeros_like(state)
            else:
                normalized = self.norm(state)
                # C0 retains identical projection parameters, applied only to its own cell.
                neighbors = c.arm == 'C1' and communication
                message = self.center(normalized) + self.left(normalized.roll(1,1) if neighbors else normalized) + self.right(normalized.roll(-1,1) if neighbors else normalized)
                delta = self.update(torch.cat((normalized,message),-1))
            updated = state + c.alpha*delta
            check_numerics(updated.detach().abs().amax() <= c.max_state_abs,
                           'cellular state outside registered numerical range')
            if trace is not None:
                trace.append({'state_norm':updated.detach().norm(dim=-1).cpu().tolist(),
                              'delta_norm':delta.detach().norm(dim=-1).cpu().tolist(),
                              'message_norm':message.detach().norm(dim=-1).cpu().tolist()})
            state = updated
        return state

    def read(self,state):
        return self.head(self.norm(self.pool(state.flatten(1))))

    def step(self,state,token_id,evolve_steps=None,communication=True):
        state = self.inject(state,token_id)
        # L0 readout computes on a temporary window, preserving only its FIFO.
        computed = self.evolve(state,evolve_steps,communication)
        return (state if self.config.arm == 'L0' else computed),self.read(computed)
