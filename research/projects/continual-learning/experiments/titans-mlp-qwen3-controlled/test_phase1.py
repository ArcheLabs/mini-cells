import pytest
import torch
from torch.nn.functional import normalize

from cell import export, fork, restore, tensors
from memory import Memory


def test_cell_roundtrip_isolation_and_read_before_write(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = Memory()
    empty, independent = model.create(), model.create()
    for key, value in tensors(empty).items():
        assert value.data_ptr() != tensors(independent)[key].data_ptr()
    key, value = normalize(torch.randn(2, 512), dim=-1)
    expected = model.read(empty, key)
    _, padded = model.event(empty, key, value, key, valid=False)
    assert padded is empty
    readout, learned = model.event(empty, key, value, key)
    again, repeated = model.event(empty, key, value, key)
    assert torch.equal(readout, expected) and torch.equal(again, readout)
    assert not torch.equal(model.read(learned, key), expected)
    for name, tensor in tensors(learned).items():
        assert torch.equal(tensor, tensors(repeated)[name])
        assert tensor.dtype == torch.float32
    before = {k: v.clone() for k, v in tensors(learned).items()}
    model.read(learned, key)
    assert all(torch.equal(before[k], v) for k, v in tensors(learned).items())
    export(tmp_path / 'cell', learned, 'fingerprint')
    with pytest.raises(ValueError, match='metadata'):
        restore(tmp_path / 'cell', empty, 'wrong')
    restored = restore(tmp_path / 'cell', empty, 'fingerprint')
    assert restored.updates == 1
    assert all(torch.equal(before[k], v) for k, v in tensors(restored).items())
    child = fork(restored)
    for name, tensor in tensors(child).items():
        assert tensor.data_ptr() != tensors(restored)[name].data_ptr()
    _, child = model.event(child, key, -value, key)
    assert child.updates == 2 and restored.updates == 1
    assert all(torch.equal(tensors(restored)[k], v) for k, v in before.items())
    assert any(not torch.equal(tensors(child)[k], v) for k, v in before.items())


def test_recurrence_momentum_and_forgetting():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    model = Memory()
    key, value = normalize(torch.randn(2, 512), dim=-1)
    initial = model.create()
    _, state = model.event(initial, key, value, key)
    _, only_momentum = model.event(state, key, value, key, eta=.5, theta=0., alpha=0.)
    _, only_forgetting = model.event(state, key, value, key, eta=0., theta=0., alpha=.1)
    assert any(torch.count_nonzero(v) for v in state.momentum.values())
    for name, weight in state.weights.items():
        torch.testing.assert_close(only_momentum.momentum[name], .5 * state.momentum[name])
        torch.testing.assert_close(only_momentum.weights[name], weight + only_momentum.momentum[name])
        torch.testing.assert_close(only_forgetting.weights[name], .9 * weight)
        assert not torch.count_nonzero(only_forgetting.momentum[name])
