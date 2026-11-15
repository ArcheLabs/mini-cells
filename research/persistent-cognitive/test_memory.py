import pytest
import torch

from cell import Cell, restore, snapshot
from memory import NeuralMemory


def test_token_causality_and_outer_gradient():
    torch.manual_seed(3)
    memory = NeuralMemory(4)
    k, v = torch.randn(2, 4), torch.randn(2, 4)
    q = torch.randn(2, 4, 4)
    initial = memory.create()
    reads, state = memory.scan(k, v, q, initial, chunk=2, meta=True)
    first, _ = memory.scan(k[:1], v[:1], q[:1], initial, meta=True)
    torch.testing.assert_close(reads[:1], first)
    assert state.step == 2 and initial.step == 0
    reads.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in memory.parameters())


def test_cell_only_restore_and_fork(tmp_path):
    memories = {'3:0': NeuralMemory(4)}
    fingerprint = {k: k for k in ('backbone', 'tokenizer', 'architecture', 'meta', 'layer_map')}
    cell = Cell.create(memories, fingerprint)
    path = tmp_path / 'cell'
    snapshot(cell, path)
    loaded = restore(path, memories, fingerprint)
    assert loaded.id == cell.id
    fork = loaded.fork()
    assert fork.parent == (loaded.id, loaded.revision)
    assert fork.shards['3:0'].fast[0].data_ptr() != loaded.shards['3:0'].fast[0].data_ptr()
    with pytest.raises(ValueError):
        restore(path, memories, {**fingerprint, 'architecture': 'stage6c'})
    (path / 'session.pt').touch()
    with pytest.raises(ValueError):
        restore(path, memories, fingerprint)
