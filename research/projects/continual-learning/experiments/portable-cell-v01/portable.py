"""Portable state boundary for upstream Titan-Llama memory."""
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'persistent-cognitive'))
from artifact import checked, read, write

SCHEMA = 'titan-llama-portable-cell-v1'


def flatten(value, tensors):
    if isinstance(value, torch.Tensor):
        key = str(len(tensors))
        tensors[key] = value
        return ['tensor', key]
    if value is None:
        return ['none']
    if type(value) is int:
        return ['int', value]
    if isinstance(value, dict) or hasattr(value, 'items'):
        return ['mapping', [[key, flatten(child, tensors)] for key, child in sorted(value.items())]]
    if isinstance(value, (list, tuple)):
        return ['tuple', [flatten(child, tensors) for child in value]]
    raise TypeError(f'Unsupported memory state field: {type(value).__name__}')


@dataclass(frozen=True)
class Cell:
    id: str
    revision: int
    position: int
    fingerprint: str
    memories: dict
    parent: tuple | None = None

    @classmethod
    def create(cls, fingerprint):
        if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('Expected immutable model bundle SHA256')
        return cls(str(uuid4()), 0, 0, fingerprint, {})

    def staged(self, memories, *, tokens, chunk_size):
        if tokens <= 0 or chunk_size <= 0 or (self.position + tokens) % chunk_size:
            raise ValueError('Commit requires a complete chunk boundary')
        if not memories or (self.memories and memories.keys() != self.memories.keys()):
            raise ValueError('Memory layer mapping differs')
        for state in memories.values():
            if any(x is not None and x.numel() for x in
                   (state.cache_store_segment, state.cache_store_mask)):
                raise ValueError('Pending writes cannot be committed')
            if int(state.seq_index) != self.position + tokens:
                raise ValueError('Memory sequence index differs from committed position')
        result = replace(self, memories=dict(memories), position=self.position + tokens,
                         revision=self.revision + 1)
        tree(result)
        return result


def tree(cell):
    tensors = {}
    schema = {}
    for layer, state in sorted(cell.memories.items()):
        if any(x is not None and x.numel() for x in
               (state.cache_store_segment, state.cache_store_mask)):
            raise ValueError('Pending writes cannot enter a Cell')
        if int(state.seq_index) != cell.position:
            raise ValueError('Invalid committed position')
        schema[str(layer)] = flatten((state.weights, state.states, state.updates), tensors)
    if not all(t.isfinite().all() for t in tensors.values()):
        raise ValueError('Nonfinite memory state')
    return schema, tensors


def export(cell, destination):
    layout, tensors = tree(cell)
    write(destination, dict(schema=SCHEMA, id=cell.id, revision=cell.revision,
                            position=cell.position, fingerprint=cell.fingerprint,
                            parent=cell.parent, layout=layout), tensors)


def rebuild(template, tensors, cursor):
    if isinstance(template, torch.Tensor):
        key = str(cursor[0])
        cursor[0] += 1
        return tensors[key].clone()
    if template is None or type(template) is int:
        return template
    if isinstance(template, dict) or hasattr(template, 'items'):
        return type(template)({k: rebuild(v, tensors, cursor) for k, v in sorted(template.items())})
    values = [rebuild(v, tensors, cursor) for v in template]
    return type(template)(*values) if hasattr(template, '_fields') else type(template)(values)


def _restore_memories(reference, tensors, position):
    cursor = [0]
    return {layer: state._replace(seq_index=position, cache_store_segment=None, cache_store_mask=None,
                                  weights=rebuild(state.weights, tensors, cursor),
                                  states=rebuild(state.states, tensors, cursor),
                                  updates=rebuild(state.updates, tensors, cursor))
            for layer, state in sorted(reference.memories.items())}


def restore(path, reference):
    metadata, tensors = read(path, SCHEMA)
    layout, expected = tree(reference)
    if (metadata.keys() != {'id', 'position', 'revision', 'fingerprint', 'parent', 'layout'}
            or metadata['fingerprint'] != reference.fingerprint or metadata['layout'] != layout
            or any(type(metadata[k]) is not int or metadata[k] < 0 for k in ('revision', 'position'))):
        raise ValueError('Incompatible portable Cell')
    UUID(metadata['id'])
    parent = metadata['parent']
    if parent is not None:
        if not isinstance(parent, list) or len(parent) != 2 or type(parent[1]) is not int or parent[1] < 0:
            raise ValueError('Invalid Cell lineage')
        UUID(parent[0])
    states = _restore_memories(reference, checked(tensors, expected), metadata['position'])
    return Cell(metadata['id'], metadata['revision'], metadata['position'],
                reference.fingerprint, states, tuple(parent) if parent else None)


def fork(cell):
    _, tensors = tree(cell)
    return replace(cell, id=str(uuid4()), revision=0, parent=(cell.id, cell.revision),
                   memories=_restore_memories(cell, tensors, cell.position))


class Slot:
    def __init__(self, cell):
        self.cell = cell
        self.lock = Lock()

    def commit(self, staged, expected_revision):
        with self.lock:
            old = self.cell
            if (old.revision != expected_revision or staged.revision != expected_revision + 1
                    or (old.id, old.fingerprint, old.parent) != (staged.id, staged.fingerprint, staged.parent)
                    or staged.position <= old.position):
                raise ValueError('Revision conflict or invalid commit')
            tree(staged)
            self.cell = staged


@dataclass(frozen=True)
class Session:
    cell_id: str
    revision: int
    position: int = 0
    attention_kv: tuple = ()

    @classmethod
    def attach(cls, cell):
        return cls(cell.id, cell.revision)

    def offsets(self, cell):
        if (self.cell_id, self.revision) != (cell.id, cell.revision):
            raise ValueError('Session attached to a different Cell revision')
        return self.position, cell.position
