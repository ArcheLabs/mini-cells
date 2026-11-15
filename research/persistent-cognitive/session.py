"""Session snapshots are independent of persistent Cell tensor artifacts."""
from dataclasses import dataclass
import uuid

from artifact import checked, read, write

CACHE_FIELDS = ('key_cache', 'value_cache', 'conv_states', 'recurrent_states')


@dataclass(frozen=True)
class Session:
    id: str
    cell_id: str
    cell_revision: int
    model_fingerprint: str
    position: int
    runtime_cache: object

    @classmethod
    def create(cls, cell, cache):
        return cls(str(uuid.uuid4()), cell.id, cell.revision,
                   cell.fingerprints['backbone'], 0, cache)

    def validate_cell(self, cell):
        if (self.cell_id, self.cell_revision, self.model_fingerprint) != (
                cell.id, cell.revision, cell.fingerprints['backbone']):
            raise ValueError('Session Cell identity or revision differs')


def _tensors(cache):
    return {f'{field}/{i}': tensor for field in CACHE_FIELDS
            for i, tensor in enumerate(getattr(cache, field)) if tensor is not None}


def snapshot_session(session, path):
    write(path, dict(schema='pca-session-v1', id=session.id, cell_id=session.cell_id,
                     cell_revision=session.cell_revision, model_fingerprint=session.model_fingerprint,
                     position=session.position), _tensors(session.runtime_cache))


def restore_session(path, cell, cache, reference):
    """Restore into a fresh cache using a trusted tensor layout from the adapter."""
    meta, tensors = read(path, 'pca-session-v1')
    if (meta.keys() != {'id', 'cell_id', 'cell_revision', 'model_fingerprint', 'position'}
            or type(meta['position']) is not int or meta['position'] < 0
            or type(meta['cell_revision']) is not int or meta['cell_revision'] < 0
            or _tensors(cache)):
        raise ValueError('Invalid Session metadata or destination cache')
    uuid.UUID(meta['id'])
    result = Session(runtime_cache=cache, **meta)
    result.validate_cell(cell)
    restored = checked(tensors, reference)
    updates = []
    for key, tensor in restored.items():
        field, index = key.split('/')
        if field not in CACHE_FIELDS or not index.isdecimal() or int(index) >= len(getattr(cache, field)):
            raise ValueError('Unexpected Session tensor mapping')
        updates.append((field, int(index), tensor.clone()))
    for field, index, tensor in updates:
        getattr(cache, field)[index] = tensor
    return result


def advance(session, slot, model, retrofit, ids, fingerprints, *, memory_enabled=True):
    """Advance a private working cache, then publish against the observed revision."""
    import copy
    from dataclasses import replace

    import torch

    current = slot.current
    session.validate_cell(current)
    if current.fingerprints != fingerprints:
        raise ValueError('Runtime compatibility fingerprints differ')
    cache = copy.copy(session.runtime_cache)
    for field in CACHE_FIELDS:
        setattr(cache, field, [None if t is None else t.detach().clone()
                               for t in getattr(session.runtime_cache, field)])
    positions = torch.arange(session.position, session.position + ids.shape[1],
                             device=ids.device)[None]
    with torch.no_grad():
        outputs, staged = retrofit(model, current, input_ids=ids, use_cache=True,
                                   past_key_values=cache, position_ids=positions,
                                   memory_enabled=memory_enabled)
        if not outputs.logits.isfinite().all():
            raise FloatingPointError('Nonfinite Session output')
    if staged is not current:
        slot.commit(staged, session.cell_revision)
    return outputs, replace(session, cell_revision=staged.revision,
                            position=session.position + ids.shape[1],
                            runtime_cache=outputs.past_key_values)
