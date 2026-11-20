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


def test_bootstrap_resume_reuses_both_sources(tmp_path):
    from bootstrap_data import SampleCache, collect, stats_for_source

    identity = {'revision': 'pinned'}
    cache = SampleCache(tmp_path, identity)

    def interrupted():
        yield torch.tensor([1, 2, 3, 4])
        raise OSError('interrupted')

    for source in ('FineWeb-Edu', 'LongAlign'):
        with pytest.raises(OSError, match='interrupted'):
            collect(source, 2, cache, interrupted, lambda item: item, 0, 4, stats_for_source())

    resumed = SampleCache(tmp_path, identity)
    for source in ('FineWeb-Edu', 'LongAlign'):
        encoded = []

        def encode(item):
            encoded.append(item.tolist())
            return item

        stats = stats_for_source()
        rows = collect(source, 2, resumed,
                       lambda: iter((torch.arange(1, 5), torch.arange(5, 9))),
                       encode, 0, 4, stats)
        assert [row.tolist() for row in rows] == [[1, 2, 3, 4], [0, 5, 6, 7]]
        assert encoded == [[5, 6, 7, 8]]
        assert stats['cache_hits'] == stats['raw_cache_hits'] == 1
        rows = collect(source, 2, resumed, lambda: pytest.fail('Unexpected network read'),
                       encode, 0, 4, stats_for_source())
        assert len(rows) == 2
    assert SampleCache(tmp_path, {'revision': 'other'}).read('LongAlign', 'block', 0) is None


def test_longalign_raw_length_and_response_truncation():
    from types import SimpleNamespace

    from bootstrap_data import MAX_RAW_BYTES, encode_longalign, stats_for_source

    def messages(pair):
        return [{'role': 'user', 'content': pair[0]}, {'role': 'assistant', 'content': pair[1]}]

    dataset = SimpleNamespace(max_length=4, _messages_from_item=messages,
                              _chat_token_ids=lambda _: ([1, 2], [1, 2, 3, 4, 5]))
    calls = []

    def tokenize(pair):
        calls.append(pair)
        dataset._chat_token_ids(messages(pair))
        return {'input_ids': torch.tensor([1, 2, 4, 5]), 'attention_mask': torch.ones(4)}

    dataset._tokenize_item = tokenize
    stats = stats_for_source()
    assert encode_longalign(dataset, ('a' * (MAX_RAW_BYTES + 1), 'b'), stats) is None
    assert encode_longalign(dataset, ('a', 'b' * (MAX_RAW_BYTES + 1)), stats) is None
    assert encode_longalign(dataset, ('a', 'b'), stats).tolist() == [1, 2, 4, 5]
    assert calls == [('a', 'b')]
    assert stats['skipped'] == 2 and stats['truncated'] == 1


def test_bootstrap_fallback_materializes_local_only_training(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import bootstrap_data as data

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha='pinned'))))

    def raw_stream(dataset_id, *_):
        if dataset_id == data.DATASETS['LongAlign']:
            raise data.PreparationLimit('LongAlign network read exceeded budget')
        yield {'text': 'example'}

    monkeypatch.setattr(data, 'raw_stream', raw_stream)

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, *_args, **_kwargs):
            return {'input_ids': list(range(1, 9))}

    config = SimpleNamespace(dataset_name='longalign_fineweb_pretrain', total_steps=2,
                             longalign_shuffle_seed=314159, sequence_length=4,
                             mixed_pretrain_fineweb_shuffle_buffer_size=1024)
    dataset = SimpleNamespace(fineweb=SimpleNamespace(tokenizer=Tokenizer()),
                              longalign=SimpleNamespace(shuffle_buffer_size=1024,
                                                        tokenizer=Tokenizer()))
    data.prepare(tmp_path, lambda _: dataset, config, tmp_path / 'cache', 'tokenizer', 'upstream')
    manifest = json.loads((tmp_path / 'bootstrap-data.json').read_text())
    assert manifest['status'] == 'FINEWEB_ONLY_FALLBACK'
    assert manifest['source_order'] == ['FineWeb-Edu'] * 2
    batches = list(torch.utils.data.DataLoader(data.LocalBootstrap(tmp_path, config), batch_size=1))
    assert [b['input_ids'].tolist() for b in batches] == [[[1, 2, 3, 4]], [[5, 6, 7, 8]]]
    (tmp_path / 'bootstrap-data.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='identity mismatch'):
        data.LocalBootstrap(tmp_path, config)
