"""Inference snapshots are distinct from optimizer checkpoints and contain no tokens."""
import hashlib
import io
import json
from dataclasses import asdict
import torch

def weight_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def checksum(meta,state):
    return hashlib.sha256(json.dumps(meta,sort_keys=True).encode()+state.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

class Runtime:
    def __init__(self,core):
        self.core=core
        self.events=0
        self.internal_steps=0
        self.session='default'

    @torch.no_grad()
    def step(self,state,token,evolve_steps=None):
        steps=self.core.config.steps if evolve_steps is None else evolve_steps
        result=self.core.step(state,token,steps)
        self.events+=1; self.internal_steps+=steps
        return result

    @torch.no_grad()
    def evolve(self,state,steps):
        result=self.core.evolve(state,steps)
        self.internal_steps+=steps
        return result

    def reset(self,state,session="default"):
        self.events=self.internal_steps=0; self.session=session
        return self.core.reset(state)

    def snapshot(self,state):
        tensor=state.detach().cpu().clone()
        meta={'format':1,'model':asdict(self.core.config),'weights':weight_hash(self.core),
              'shape':list(tensor.shape),'dtype':str(tensor.dtype),'events':self.events,
              'internal_steps':self.internal_steps,'session':self.session,'schedule_rng':None}
        return {'metadata':meta,'state':tensor,'sha256':checksum(meta,tensor)}

    def restore(self,snapshot):
        meta,state=snapshot['metadata'],snapshot['state']
        if meta['format']!=1 or meta['model']!=asdict(self.core.config) or meta['weights']!=weight_hash(self.core) or snapshot['sha256']!=checksum(meta,state):
            raise ValueError('snapshot identity or checksum mismatch')
        if list(state.shape)!=meta['shape'] or str(state.dtype)!=meta['dtype'] or tuple(state.shape[1:])!=(self.core.config.cells,self.core.config.dim) or not torch.isfinite(state).all():
            raise ValueError('invalid snapshot tensor')
        self.events,self.internal_steps,self.session=meta['events'],meta['internal_steps'],meta['session']
        return state.to(self.core.embedding.weight).clone()

    @torch.no_grad()
    def generate(self,tokens,count=32):
        state=self.core.init_state(1)
        for token in tokens:
            state,logits=self.step(state,[token])
        generated=[]
        for _ in range(count):
            token=int(logits.argmax(-1)); generated.append(token)
            state,logits=self.step(state,[token])
        return generated
