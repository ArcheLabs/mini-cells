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


def test_runtime_comparison_detects_state_and_retrieval_errors(tmp_path):
    from safetensors.torch import save_file

    from run_runtime import CHUNK, compare

    value = torch.ones(2)
    cell = Cell.create('c' * 64)
    for position in (CHUNK, 2 * CHUNK):
        state = State(position, {'w': value}, None, (value, value), (value,), None)
        cell = cell.staged({8: state}, tokens=CHUNK, chunk_size=CHUNK)
    for arm in ('continuous', 'restore'):
        export(cell, tmp_path / arm)
        save_file({'retrieval': value}, str(tmp_path / f'{arm}-read.safetensors'))
    assert compare(tmp_path)['seq_index'] == 2 * CHUNK
    save_file({'retrieval': value + 0.1}, str(tmp_path / 'restore-read.safetensors'))
    with pytest.raises(AssertionError):
        compare(tmp_path)


def test_bootstrap_batches_round_trip_in_upstream_order(tmp_path):
    from types import SimpleNamespace

    from bootstrap_data import LocalBootstrap, prepare

    config = SimpleNamespace(dataset_name='longalign_fineweb_pretrain',
                             longalign_shuffle_seed=314159, total_steps=2, sequence_length=4)
    rows = [{name: torch.full((4,), index, dtype=torch.int64)
             for name in ('input_ids', 'attention_mask', 'labels')} for index in (1, 2)]
    prepare(tmp_path, lambda _: iter(rows), config)
    loader = torch.utils.data.DataLoader(LocalBootstrap(tmp_path, config), batch_size=1)
    for expected, actual in zip(rows, loader, strict=True):
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name].unsqueeze(0))
    (tmp_path / 'bootstrap-data.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='identity mismatch'):
        LocalBootstrap(tmp_path, config)
