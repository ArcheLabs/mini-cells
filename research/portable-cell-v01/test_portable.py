from collections import namedtuple

import pytest
import torch

from portable import Cell, Session, Slot, export, fork, restore

State = namedtuple('NeuralMemState', 'seq_index weights cache_store_segment states updates cache_store_mask')


def test_cell_roundtrip_and_revision(tmp_path):
    key = torch.randn(2, 3)
    memory = State(4, {'w': key}, None, (key + 1, key + 2), {'w': key + 3}, None)
    initial = Cell.create('a'*64)
    with pytest.raises(ValueError, match='Pending'):
        initial.staged({0: memory._replace(cache_store_segment=key)}, tokens=4, chunk_size=4)
    staged = initial.staged({0: memory}, tokens=4, chunk_size=4)
    slot = Slot(initial)
    slot.commit(staged, 0)
    with pytest.raises(ValueError, match='Revision'):
        slot.commit(staged, 0)
    export(staged, tmp_path/'cell')
    loaded = restore(tmp_path/'cell', staged)
    torch.testing.assert_close(loaded.memories[0].updates['w'], memory.updates['w'])
    assert Session.attach(loaded).offsets(loaded) == (0, 4)
    child = fork(loaded)
    child.memories[0].weights['w'].zero_()
    torch.testing.assert_close(loaded.memories[0].weights['w'], key)
    assert child.parent == (loaded.id, 1)
    (tmp_path/'cell'/'state.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError, match='integrity'):
        restore(tmp_path/'cell', staged)


def test_restore_in_fresh_process(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    cell = Cell.create('b' * 64)
    weights = torch.tensor([1., 2.])
    cell = cell.staged({0: State(4, {'w': weights}, None, None, None, None)},
                       tokens=4, chunk_size=4)
    export(cell, tmp_path / 'original')
    subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
import torch
from test_portable import State
from portable import Cell, Session, export, restore

root = Path(sys.argv[1])
reference = Cell.create('b' * 64).staged(
    {0: State(4, {'w': torch.zeros(2)}, None, None, None, None)},
    tokens=4, chunk_size=4)
cell = restore(root / 'original', reference)
assert Session.attach(cell).offsets(cell) == (0, 4)
torch.testing.assert_close(cell.memories[0].weights['w'], torch.tensor([1., 2.]))
state = cell.memories[0]
export(cell.staged({0: state._replace(seq_index=8)}, tokens=4, chunk_size=4),
       root / 'next')
''', str(tmp_path)], cwd=Path(__file__).parent, check=True)
    next_cell = restore(tmp_path / 'next', cell)
    assert (next_cell.id, next_cell.revision, next_cell.position) == (cell.id, 2, 8)
