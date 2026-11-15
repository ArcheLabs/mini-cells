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


def test_phase_c_reference_and_long_run():
    torch.manual_seed(7)
    memory = NeuralMemory(4)
    x = torch.eye(4)
    expected = torch.tensor(memory.update.controls)
    torch.testing.assert_close(memory.controls(x).sigmoid(), expected.expand(4, -1))
    base = memory.create()
    single, _ = memory.scan(x, x, x[:, None], base)
    manual = []
    state = base
    with torch.no_grad():
        for key in x:
            state = memory.transition(key, key, state, state.fast)
            manual.append(memory.read(key[None], state))
        torch.testing.assert_close(single, torch.stack(manual))
        chunked, _ = memory.scan(x, x, x[:, None], base, chunk=4)
        changed = x.clone()
        changed[2:] *= -1
        prefix, _ = memory.scan(changed, changed, x[:, None], base, chunk=4)
        torch.testing.assert_close(chunked[:2], prefix[:2])
        state = memory.transition(x[0], x[0], base, base.fast)
        for _ in range(512):
            state = memory.transition(x[1], x[1], state, state.fast)
        projected = memory.project(x, state.address)
    assert state.step == 513
    assert 0 < projected[0, 0] < projected[1, 1] <= 1
    assert projected[2:].count_nonzero() == 0
    assert all(t.isfinite().all() and t.norm() < 100 for t in (*state.fast, *state.momentum))
