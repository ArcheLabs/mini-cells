from __future__ import annotations

import torch

from minicells.continual_learning import (
    PARAMETER_COUNT,
    build_adaptation_pool,
    build_old_pool,
    delta_vector,
    load_q88_model,
    save_q88_model,
)
from minicells.vocab import CharVocab


def test_old_and_adaptation_domains_are_disjoint() -> None:
    vocab = CharVocab()
    old = build_old_pool(vocab, seed=1, examples=128)
    new = build_adaptation_pool(vocab, seed=2, examples=128)
    marker = torch.tensor(vocab.encode("??"), dtype=torch.long)

    assert not ((old.input_ids[:, :2] == marker).all(dim=1)).any()
    assert ((new.input_ids[:, :2] == marker).all(dim=1)).all()


def test_adaptation_changes_exactly_one_payload_token() -> None:
    vocab = CharVocab()
    batch = build_adaptation_pool(vocab, seed=3, examples=64)
    changed = (batch.input_ids != batch.target_ids) & batch.mask

    assert torch.equal(changed, batch.changed_mask)
    assert (changed.sum(dim=1) == 1).all()
    assert (changed[:, 2]).all()


def test_spsa_block_delta_is_sparse_and_global_is_dense() -> None:
    parent_hash = bytes(range(32))
    global_delta = delta_vector(parent_hash, generation=1, block_size=PARAMETER_COUNT)
    block_delta = delta_vector(parent_hash, generation=1, block_size=512)

    assert ((global_delta == 1) | (global_delta == -1)).all()
    assert int((block_delta != 0).sum()) <= 512
    assert int((block_delta != 0).sum()) > 0
    assert ((block_delta == 0) | (block_delta == 1) | (block_delta == -1)).all()


def test_q88_model_binary_round_trip(tmp_path) -> None:
    source = torch.arange(PARAMETER_COUNT, dtype=torch.int64) % 4097 - 2048
    path = tmp_path / "model.bin"
    save_q88_model(path, source)
    loaded = load_q88_model(path)

    assert torch.equal(source, loaded)
    assert path.stat().st_size == PARAMETER_COUNT * 2
