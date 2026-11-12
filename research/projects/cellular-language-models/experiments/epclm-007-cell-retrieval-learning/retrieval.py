"""Candidate retrieval with explicit split and acquisition boundaries."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent/'epclm-006-cross-session-cell-portability'))
from portable_model import PortableModel, NeuralMemory, state_hash
from cell_artifact import compatibility, export_cell, import_cell
from data import digest

OFFSETS = {'train': 0, 'validation': 1, 'test': 2}
SYMBOLS = 24
CANDIDATES = 8


def episode(split, seed, index):
    if split == 'test':
        raise ValueError('Test is sealed')
    identity = digest(['epclm007-v1', split, seed, index])
    rng = torch.Generator().manual_seed(int(identity[:16],16))
    keys = torch.randperm(SYMBOLS,generator=rng)[:CANDIDATES]
    mapping = torch.empty(SYMBOLS,dtype=torch.long)
    for residue in range(3):
        pool = torch.arange((residue+OFFSETS[split])%3,SYMBOLS,3)
        mapping[residue::3] = pool[torch.randperm(len(pool),generator=rng)]
    values = mapping[keys]
    order = torch.randperm(CANDIDATES,generator=rng)
    candidate_order = torch.randperm(CANDIDATES,generator=rng)
    candidates = values[candidate_order]
    targets = (values[order,None]==candidates).long().argmax(-1)
    return dict(id=identity,keys=keys,values=values,queries=keys[order],
                candidates=candidates,targets=targets)


def rank(readout,candidates):
    return 10*(F.normalize(readout,dim=-1) @ F.normalize(candidates,dim=-1).T)


class Retrieval(PortableModel):
    def __init__(self):
        super().__init__()
        self.backbone.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        self.memory_out.requires_grad_(False)
        self.gate.requires_grad_(False)

    def encode(self,ids,offset):
        with torch.no_grad():
            return self.encode_cell_segment((ids+offset).reshape(-1,1))

    def represent(self,row):
        k = F.normalize(F.silu(self.memory.key(self.encode(row['keys'],8))),dim=-1)
        v = F.silu(self.memory.value(self.encode(row['values'],264)))
        q = F.normalize(F.silu(self.memory.query(self.encode(row['queries'],8))),dim=-1)
        candidates = F.silu(self.memory.value(self.encode(row['candidates'],264)))
        return k,v,q,candidates

    def acquire_vectors(self,k,v,steps,mode):
        cell = self.memory.create(1,mode)
        initial = cell.detached()
        trace = []
        for _ in range(steps):
            cell,diag = self.memory.learn(k[None],v[None],cell,mode)
            trace.append({name:float(t.mean()) for name,t in diag.items()})
        return cell,dict(updates=trace,
                         fast_delta=sum(float((v.detach()-initial.fast[k]).square().sum()) for k,v in cell.fast.items())**.5,
                         momentum_norm=sum(float(v.detach().square().sum()) for v in cell.momentum.values())**.5)

    def forward(self,row):
        k,v,q,candidates = self.represent(row)
        cell,trace = self.acquire_vectors(k,v,8,'meta_train')
        return rank(self.memory.recall(q[None],cell,'meta_train')[0],candidates),trace
