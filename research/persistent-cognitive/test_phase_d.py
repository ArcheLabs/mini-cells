from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cell import Cell, CellSlot, restore, snapshot, state_hash
from memory import NeuralMemory
from session import CACHE_FIELDS, Session, restore_session, snapshot_session


def fresh_cache():
    return SimpleNamespace(**{name: [None] for name in CACHE_FIELDS})


def test_revision_and_independent_snapshots(tmp_path):
    memory = NeuralMemory(4)
    memories = {'3:0': memory}
    identity = {key: key for key in ('backbone', 'tokenizer', 'architecture', 'meta', 'layer_map')}
    original = Cell.create(memories, identity)
    slot = CellSlot(original)
    key = torch.ones(4)
    shard = original.shards['3:0']
    next_shard = memory.transition(key, key, shard, shard.fast)
    staged = replace(original, revision=1, shards={'3:0': next_shard})
    assert slot.commit(staged, 0) is staged
    with pytest.raises(ValueError, match='revision conflict'):
        slot.commit(staged, 0)
    assert slot.current is staged and original.shards['3:0'].step == 0
    snapshot(staged, tmp_path/'cell')
    loaded = restore(tmp_path/'cell', memories, identity)
    assert state_hash(loaded) == state_hash(staged)

    cache = fresh_cache()
    cache.key_cache[0] = torch.ones(1, 1, 2, 4)
    session = replace(Session.create(loaded, cache), position=8)
    snapshot_session(session, tmp_path/'session')
    reference = {'key_cache/0': cache.key_cache[0]}
    recovered = restore_session(tmp_path/'session', loaded, fresh_cache(), reference)
    assert recovered.position == 8 and recovered.cell_revision == 1
    torch.testing.assert_close(recovered.runtime_cache.key_cache[0], cache.key_cache[0])
    recovered.runtime_cache.key_cache[0].zero_()
    assert cache.key_cache[0].sum() == 8
    with pytest.raises(ValueError, match='identity or revision'):
        restore_session(tmp_path/'session', original, fresh_cache(), reference)
    with pytest.raises(ValueError, match='schema'):
        restore(tmp_path/'session', memories, identity)
    artifact = tmp_path/'cell'/'state.safetensors'
    artifact.write_bytes(artifact.read_bytes() + b'bad')
    with pytest.raises(ValueError, match='integrity'):
        restore(tmp_path/'cell', memories, identity)
