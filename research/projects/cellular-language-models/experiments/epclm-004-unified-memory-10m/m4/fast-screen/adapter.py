"""Fixed branch masks and portable Session state for frozen M4 E5."""
from dataclasses import replace
from pathlib import Path
import sys

import torch

M4=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(M4.parent))
sys.path.insert(0,str(M4))
from fusion import Runtime, ThreeWayFusion
from e0 import load_checkpoint, save_checkpoint
from data import HOLE, digest, file_hash

MASKS={'F00':('memory','ssm'),'F10':('memory',),'F01':('ssm',),'F11':()}


class Factorial(ThreeWayFusion):
    def __init__(self,arm,model,fusion):
        super().__init__(model,**fusion)
        self.arm=arm
        self.disabled=MASKS[arm]
        self.frozen_names=tuple(name for name,_ in self.named_parameters()
                                if any(name.startswith(prefix+'.') for prefix in
                                       (('ssm',) if 'ssm' in self.disabled else ())+
                                       (('memory','memory_output','memory_norm') if 'memory' in self.disabled else ())))
        self.apply_mask()

    def apply_mask(self):
        for name,p in self.named_parameters():
            if name in self.frozen_names:
                p.requires_grad_(False)

    def forward(self,ids):
        mode=self.memory_mode
        if not self.training and mode=='meta_train':
            self.memory_mode='online_adapt'
        try:
            return self.forward_state(ids,disabled=self.disabled)[0]
        finally:
            self.memory_mode=mode


def tensor_digest(tensors):
    import hashlib
    return digest({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                   for k,v in tensors.items()})


def frozen_hash(model):
    params=dict(model.named_parameters())
    return tensor_digest({name:params[name] for name in model.frozen_names})


def sessions(rows,chunk,device):
    prefixes,suffixes,positions,targets=[],[],[],[]
    for row in rows:
        q=row['query_positions']
        start=q[0]-2
        ids=row['input_ids']
        from data import QUERY, ANSWER
        assert all(ids[i-2]==QUERY and ids[i]==ANSWER and 8<=ids[i-1]<264 for i in q)
        assert [i for i,y in enumerate(row['labels']) if y!=-100]==q
        assert all(ids[j]==HOLE for j in range(start,len(ids)) if j not in
                   {i-d for i in q for d in range(3)})
        a,b=ids[:start],ids[start:]
        prefixes.append(a+[HOLE]*(-len(a)%chunk))
        suffixes.append(b+[HOLE]*(-len(b)%chunk))
        positions.append([i-start for i in q])
        targets.append([row['labels'][i] for i in q])
    tensors=[torch.tensor(x,device=device) for x in (prefixes,suffixes,positions,targets)]
    return (*tensors,dict(prefix_padding=len(prefixes[0])-start,
                          query_padding=len(suffixes[0])-(len(ids)-start),
                          prefix_tokens=start,query_tokens=len(ids)-start))


def wrapper(model,batch,ssm=None):
    if ssm is not None:
        ssm=replace(ssm,owner=id(model.ssm),weight_version=model.ssm._version(),kv_cache={})
    return Runtime(id(model),tuple(p._version for p in model.parameters()),model.session,
                   tuple(sorted(model.disabled)),batch,None,ssm)


def rebuild_ssm(model,payload,device):
    from baselines import Runtime as SSMRuntime
    states={int(k):tuple(v.to(device) for v in pair) for k,pair in payload['mamba'].items()}
    return SSMRuntime(id(model.ssm),payload['position'],states,{},model.ssm._version())


def save_session(model,folder,cell,ssm,identity,expected):
    folder.mkdir(parents=True,exist_ok=True)
    if cell is not None:
        model.memory.checkpoint(cell,folder/'cell',identity)
    payload=None if ssm is None else dict(position=ssm.position,
                                         mamba={k:tuple(v.cpu() for v in pair) for k,pair in ssm.mamba_state.items()})
    save_checkpoint(dict(identity=identity,ssm=payload,cell=cell is not None,
                         expected=expected.detach().cpu()),folder/'runtime.pt')
    return {'runtime_sha256':file_hash(folder/'runtime.pt'),
            'cell_manifest_sha256':file_hash(folder/'cell/manifest.json') if cell is not None else None}


def load_session(model,folder,identity,hashes):
    path=folder/'runtime.pt'
    if file_hash(path)!=hashes['runtime_sha256']:
        raise ValueError('Runtime checksum mismatch')
    data=load_checkpoint(path)
    if data['identity']!=identity:
        raise ValueError('Runtime identity mismatch')
    if data['cell'] and file_hash(folder/'cell/manifest.json')!=hashes['cell_manifest_sha256']:
        raise ValueError('Cell manifest checksum mismatch')
    cell=model.memory.restore(folder/'cell',identity) if data['cell'] else None
    ssm=None if data['ssm'] is None else rebuild_ssm(model,data['ssm'],next(model.parameters()).device)
    return cell,ssm,data['expected']
