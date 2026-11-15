"""Cell-only persistence. Shared parameters and Session tensors live elsewhere."""
from dataclasses import dataclass, replace
from threading import Lock
import uuid

from safetensors.torch import save

from artifact import canonical, checked, digest, read, write

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


def tensors(cell):
    return {f'{key}/{kind}/{i}': tensor
            for key, shard in cell.shards.items()
            for kind, values in [('fast', shard.fast), ('momentum', shard.momentum),
                                 ('address', (shard.address,))]
            for i, tensor in enumerate(values)}


def state_hash(cell):
    payload = save({k: v.detach().cpu().contiguous().clone() for k, v in tensors(cell).items()})
    return digest(canonical([digest(payload), {k: s.step for k, s in cell.shards.items()}]))


def snapshot(cell, path):
    write(path, dict(schema='persistent-cognitive-v2', id=cell.id, revision=cell.revision,
                     fingerprints=cell.fingerprints, parent=cell.parent,
                     steps={k: s.step for k, s in cell.shards.items()},
                     state_hash=state_hash(cell)), tensors(cell))


def restore(path, memories, fingerprints):
    meta, records = read(path, 'persistent-cognitive-v2')
    if (meta.keys() != {'id', 'revision', 'fingerprints', 'parent', 'steps', 'state_hash'}
            or meta['fingerprints'] != fingerprints
            or type(meta['revision']) is not int or meta['revision'] < 0
            or meta['steps'].keys() != memories.keys()
            or any(type(step) is not int or step < 0 for step in meta['steps'].values())):
        raise ValueError('Incompatible Cell metadata')
    uuid.UUID(meta['id'])
    parent = meta['parent']
    if parent is not None:
        if not isinstance(parent, list) or len(parent) != 2 or type(parent[1]) is not int or parent[1] < 0:
            raise ValueError('Invalid Cell parent')
        uuid.UUID(parent[0])
    fresh = Cell.create(memories, fingerprints)
    state = checked(records, tensors(fresh))
    shards = {key: Shard(tuple(state[f'{key}/fast/{i}'] for i in range(len(s.fast))),
                         tuple(state[f'{key}/momentum/{i}'] for i in range(len(s.momentum))),
                         state[f'{key}/address/0'], meta['steps'][key])
              for key, s in fresh.shards.items()}
    cell = Cell(meta['id'], meta['revision'], dict(fingerprints), shards,
                tuple(parent) if parent else None)
    if state_hash(cell) != meta['state_hash']:
        raise ValueError('Cell state hash differs')
    return cell


class CellSlot:
    """Serialize publication of competing staged revisions."""

    def __init__(self, cell):
        self._cell = cell
        self._lock = Lock()

    @property
    def current(self):
        with self._lock:
            return self._cell

    def commit(self, staged, expected_revision):
        with self._lock:
            current = self._cell
            if current.revision != expected_revision:
                raise ValueError('Cell revision conflict')
            if (staged.id != current.id or staged.fingerprints != current.fingerprints
                    or staged.parent != current.parent or staged.revision != expected_revision + 1
                    or staged.shards.keys() != current.shards.keys()
                    or any(staged.shards[k].step <= s.step for k, s in current.shards.items())):
                raise ValueError('Invalid staged Cell')
            checked(tensors(staged), tensors(current))
            self._cell = staged
            return staged
