from types import SimpleNamespace

import pytest

from segment6c import SegmentState


def test_segment_boundary_and_partial_flush():
    cache = SimpleNamespace(key_cache=[1, 2], value_cache=[3, 4], recurrent_states=[5, 6])
    commits = []
    state = SegmentState(cache, [1], lambda: commits.append(state.position), length=4)
    state.advance(2)
    assert cache.key_cache == [1, 2]
    with pytest.raises(ValueError, match='boundary'):
        state.advance(3)
    state.advance(2)
    assert cache.key_cache == [1, None]
    assert cache.value_cache == [3, None]
    assert cache.recurrent_states == [5, 6]
    assert commits == [4]
    state.advance(1)
    state.flush_segment()
    state.flush_segment()
    assert commits == [4, 5] and state.position == 5


def test_failed_commit_keeps_pending_segment():
    cache = SimpleNamespace(key_cache=[1], value_cache=[2])

    def fail():
        raise RuntimeError('write failed')

    state = SegmentState(cache, [0], fail, length=4)
    state.advance(2)
    with pytest.raises(RuntimeError, match='write failed'):
        state.flush_segment()
    assert state.pending_tokens == 2 and cache.key_cache == [1]
