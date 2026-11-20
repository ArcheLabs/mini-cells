"""Shared upstream Titans module; each Cell owns its complete recurrent state."""
import hashlib
import json
from pathlib import Path
import sys
import torch

HAM = Path(__file__).resolve().parents[1] / 'ham-001'
sys.path.insert(0,str(HAM))
from titans_adapter import Titans, load_upstream, REVISION

SEED=42


def pack(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x,tuple):
        return tuple(map(pack,x))
    if x is None or isinstance(x,(int,bool)):
        return x
    raise TypeError(type(x))


def fingerprint(x):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(repr((str(v.dtype),tuple(v.shape))).encode())
            h.update(v.contiguous().numpy().tobytes())
        elif isinstance(v,tuple):
            h.update(f'tuple:{len(v)};'.encode())
            for z in v:
                visit(z)
        else:
            h.update(repr(v).encode()+b';')
    visit(x)
    return h.hexdigest()


class Runtime:
    def __init__(self, upstream):
        from types import SimpleNamespace
        self.config=dict(heads=1,dimension=16)
        self.backend=Titans(SimpleNamespace(**self.config),load_upstream(upstream),SEED)
        self.cells={}
        self.identity=dict(revision=REVISION,config=self.config,seed=SEED,
            state_fields=list(self.backend.upstream.NeuralMemState._fields),
            weights=fingerprint(tuple(p.detach() for p in self.backend.memory.state_dict().values())))

    def create(self, cell):
        if cell in self.cells:
            raise ValueError(cell)
        self.cells[cell]=(0,None)

    def hashes(self):
        return {c:fingerprint((v,pack(s))) for c,(v,s) in self.cells.items()}

    def write_many(self, inputs, fail=None):
        staged={}
        for cell,(key,value) in inputs.items():
            version,state=self.cells[cell]
            self.backend.import_state(state)
            try:
                if cell==fail:
                    raise RuntimeError('injected write failure')
                self.backend.write(key,value)
                staged[cell]=(version+1,self.backend.export_state())
            finally:
                self.backend.import_state(None)
        self.cells.update(staged)

    def read_many(self, query, ids):
        results=[]
        for cell in ids:
            version,state=self.cells[cell]
            self.backend.import_state(state)
            try:
                value=self.backend.read(query)
                results.append(dict(cell_id=cell,state_version=version,read_result=value.tolist()))
            finally:
                self.backend.import_state(None)
        return results

    def save(self, directory):
        directory.mkdir()
        f=directory/'cells.pt'
        torch.save({c:(v,pack(s)) for c,(v,s) in self.cells.items()},f)
        (directory/'manifest.json').write_text(json.dumps(dict(identity=self.identity,
            sha256=hashlib.sha256(f.read_bytes()).hexdigest(),state_hashes=self.hashes()),indent=2))

    def restore(self,directory):
        receipt=json.loads((directory/'manifest.json').read_text())
        f=directory/'cells.pt'
        if receipt['identity']!=self.identity or receipt['sha256']!=hashlib.sha256(f.read_bytes()).hexdigest():
            raise ValueError('Invalid checkpoint identity or hash')
        restored={}
        for cell,(version,state) in torch.load(f,weights_only=True,map_location='cpu').items():
            state=self.backend.upstream.NeuralMemState(*state) if state is not None else None
            restored[cell]=(version,state)
        if {c:fingerprint((v,pack(s))) for c,(v,s) in restored.items()}!=receipt['state_hashes']:
            raise ValueError('Restored state fingerprints differ')
        self.cells=restored
