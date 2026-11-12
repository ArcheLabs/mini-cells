"""Validated non-executable tensor artifacts and atomic attachment changes."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import torch

from portable_model import EP4,P,state_hash
from memory import Cell
from adapter import tensor_digest
from data import digest,file_hash
from e0 import write

FORMAT='minicells.portable-cell'
FILES={'manifest.json','state.pt','integrity.json'}


def compatibility(model,checkpoint_sha):
    return dict(backbone_checkpoint_id=checkpoint_sha,
                backbone_config_hash=digest(model.model_config),
                tokenizer_id=file_hash(EP4/'assets/tokenizer.json'),
                cell_interface_version=P['interface'],
                projection_version=tensor_digest({k:v for k,v in model.memory.state_dict().items()
                                                  if k.startswith(('key.','query.','value.'))}),
                memory_architecture='mag_lite_single_memory',
                memory_implementation_version=file_hash(EP4/'m3/memory.py'),
                update_rule_version='frozen-m3-transition-v1',dtype='float32')


def export_cell(cell,path,identity,source_ref):
    path=Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    uuid.UUID(source_ref)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.cell-',dir=path.parent))
    try:
        state={name:{k:v.detach().cpu().contiguous() for k,v in getattr(cell,name).items()}
               for name in ('fast','momentum')}
        shapes={k:{n:list(v.shape) for n,v in d.items()} for k,d in state.items()}
        torch.save(state,tmp/'state.pt')
        manifest=dict(format=FORMAT,schema_version=1,compatibility=identity,
                      cell_id=cell.metadata['cell_id'],revision=cell.metadata['revision'],
                      parent_cell_id=(cell.metadata.get('parent') or {}).get('cell_id'),
                      update_step=cell.update_step,tensor_shapes_hash=digest(shapes),
                      state_hash=state_hash(cell),created_from_session=source_ref,
                      created_at=datetime.now(timezone.utc).isoformat())
        write(tmp/'manifest.json',manifest)
        hashes={f:file_hash(tmp/f) for f in ('manifest.json','state.pt')}
        write(tmp/'integrity.json',dict(files=hashes,artifact_sha256=digest(hashes)))
        read_manifest(tmp)
        tmp.rename(path)
        return digest(hashes)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def read_manifest(path):
    if path.is_symlink() or {p.name for p in path.iterdir()}!=FILES:
        raise ValueError('Unexpected Cell artifact contents')
    if any((path/f).is_symlink() or not (path/f).is_file() for f in FILES):
        raise ValueError('Cell artifact must contain regular files')
    integrity=json.loads((path/'integrity.json').read_text())
    hashes={f:file_hash(path/f) for f in ('manifest.json','state.pt')}
    if integrity!=dict(files=hashes,artifact_sha256=digest(hashes)):
        raise ValueError('Cell integrity mismatch')
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('format')!=FORMAT or manifest.get('schema_version')!=1:
        raise ValueError('Unsupported Cell format')
    return manifest


def import_cell(path,model,identity):
    path=Path(path)
    m=read_manifest(path)
    if m['compatibility']!=identity:
        raise ValueError('Cell compatibility mismatch')
    uuid.UUID(m['cell_id'])
    uuid.UUID(m['created_from_session'])
    if m['parent_cell_id'] is not None:
        uuid.UUID(m['parent_cell_id'])
    if type(m['update_step']) is not int or m['update_step']<0 or type(m['revision']) is not int or m['revision']<0:
        raise ValueError('Invalid Cell counters')
    state=torch.load(path/'state.pt',weights_only=True,map_location='cpu')
    if set(state)!= {'fast','momentum'}:
        raise ValueError('Unexpected Cell tensor groups')
    p=next(model.parameters())
    for group in state.values():
        if set(group)!=set(model.memory.initial):
            raise ValueError('Unexpected Cell tensor names')
        for name,v in group.items():
            if not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or p.dtype!=v.dtype or (
                    list(v.shape)!=[1,*model.memory.initial[name].shape]) or not v.isfinite().all():
                raise ValueError('Invalid Cell tensor shape, dtype or values')
    shapes={k:{n:list(v.shape) for n,v in d.items()} for k,d in state.items()}
    if digest(shapes)!=m['tensor_shapes_hash']:
        raise ValueError('Cell tensor manifest mismatch')
    metadata=dict(cell_id=m['cell_id'],revision=m['revision'],
                  parent={'cell_id':m['parent_cell_id']} if m['parent_cell_id'] else None,
                  architecture_id='mag_lite_single_memory',interface_version=1,
                  memory_dim=model.memory.dim,hidden=model.memory.hidden,provenance=[])
    cell=Cell(*({k:v.to(p.device).clone() for k,v in state[group].items()}
                for group in ('fast','momentum')),m['update_step'],metadata)
    if state_hash(cell)!=m['state_hash']:
        raise ValueError('Cell state hash mismatch')
    return cell


def fork_cell(model,cell,new_id=None):
    result=model.memory.fork(cell)
    if new_id is None:
        return result
    uuid.UUID(new_id)
    if new_id==cell.metadata['cell_id']:
        raise ValueError('Fork identity must differ')
    return replace(result,metadata={**result.metadata,'cell_id':new_id})


class CellSlot:
    """An attached read-only Cell reference, independent of Session state."""
    def __init__(self,model,identity):
        self.model=model
        self.identity=identity
        self.cell=None

    def replace(self,path):
        candidate=import_cell(path,self.model,self.identity)
        self.cell=candidate
        return candidate

    def unload(self):
        self.cell=None

    @torch.no_grad()
    def read(self,hidden,packets):
        before=state_hash(self.cell) if self.cell is not None else None
        result=self.model.query(hidden,packets,self.cell)
        after=state_hash(self.cell) if self.cell is not None else None
        if before!=after:
            raise RuntimeError('Read-only Cell mutated')
        return result


def delete_cell_artifact(path):
    path=Path(path)
    read_manifest(path)
    shutil.rmtree(path)
