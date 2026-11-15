"""Cell-only persistence. Shared parameters and Session tensors live elsewhere."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

import torch

from memory import Shard


@dataclass(frozen=True)
class Cell:
    id: str
    revision: int
    fingerprints: dict
    shards: dict
    parent: tuple | None = None

    @classmethod
    def create(cls, memories, fingerprints):
        required = {'backbone', 'tokenizer', 'architecture', 'meta', 'layer_map'}
        if fingerprints.keys() != required or any(not value for value in fingerprints.values()):
            raise ValueError('Incomplete compatibility fingerprints')
        return cls(str(uuid.uuid4()), 0, dict(fingerprints),
                   {key: memory.create() for key, memory in memories.items()})

    def fork(self):
        shards = {key: Shard(tuple(w.detach().clone() for w in s.fast),
                             tuple(m.detach().clone() for m in s.momentum),
                             s.address.detach().clone(), s.step)
                  for key, s in self.shards.items()}
        return Cell(str(uuid.uuid4()), 0, dict(self.fingerprints), shards,
                    (self.id, self.revision))


def snapshot(cell, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp) / 'cell'
        stage.mkdir()
        payload = stage / 'tensors.pt'
        torch.save({k: dict(fast=tuple(w.detach().cpu() for w in s.fast),
                            momentum=tuple(m.detach().cpu() for m in s.momentum),
                            address=s.address.detach().cpu(), step=s.step)
                    for k, s in cell.shards.items()}, payload)
        manifest = dict(schema='persistent-cognitive-v1', id=cell.id, revision=cell.revision,
                        fingerprints=cell.fingerprints, parent=cell.parent,
                        sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
        (stage / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
        stage.rename(path)


def restore(path, memories, fingerprints):
    path = Path(path)
    if {item.name for item in path.iterdir()} != {'manifest.json', 'tensors.pt'}:
        raise ValueError('Unexpected Cell files')
    manifest = json.loads((path / 'manifest.json').read_text())
    payload = path / 'tensors.pt'
    if (set(manifest) != {'schema', 'id', 'revision', 'fingerprints', 'parent', 'sha256'}
            or manifest['schema'] != 'persistent-cognitive-v1'
            or manifest['fingerprints'] != fingerprints
            or type(manifest['revision']) is not int or manifest['revision'] < 0
            or hashlib.sha256(payload.read_bytes()).hexdigest() != manifest['sha256']):
        raise ValueError('Incompatible or damaged Cell')
    uuid.UUID(manifest['id'])
    records = torch.load(payload, weights_only=True, map_location='cpu')
    if records.keys() != memories.keys():
        raise ValueError('Cell shard mapping differs')
    shards = {}
    for key, record in records.items():
        reference = memories[key].create()
        if (record.keys() != {'fast', 'momentum', 'address', 'step'}
                or type(record['step']) is not int or record['step'] < 0):
            raise ValueError('Unexpected shard schema')
        def check(actual, expected):
            if (actual.shape != expected.shape or actual.dtype != expected.dtype
                    or not actual.isfinite().all()):
                raise ValueError('Invalid Cell tensor')
            return actual.to(expected.device)
        fast = tuple(check(a, b) for a, b in zip(record['fast'], reference.fast, strict=True))
        momentum = tuple(check(a, b) for a, b in zip(record['momentum'], reference.momentum, strict=True))
        shards[key] = Shard(fast, momentum, check(record['address'], reference.address), record['step'])
    return Cell(manifest['id'], manifest['revision'], dict(fingerprints), shards,
                tuple(manifest['parent']) if manifest['parent'] else None)
