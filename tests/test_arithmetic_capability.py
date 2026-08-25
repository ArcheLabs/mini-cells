from __future__ import annotations

import torch

from minicells.arithmetic_optimizer import guarded_spsa_step
from minicells.arithmetic_tasks import (
    all_arithmetic_examples,
    arithmetic_batch,
    split_arithmetic_examples,
)
from minicells.continual_learning import PARAMETER_COUNT, TaskBatch
from minicells.vocab import CharVocab


def test_arithmetic_universe_and_split_are_deterministic() -> None:
    all_examples = all_arithmetic_examples()
    train_a, heldout_a = split_arithmetic_examples(seed=4004)
    train_b, heldout_b = split_arithmetic_examples(seed=4004)

    assert len(all_examples) == 110
    assert len([x for x in all_examples if x.operation == "add"]) == 55
    assert len([x for x in all_examples if x.operation == "sub"]) == 55
    assert train_a == train_b and heldout_a == heldout_b
    assert len(train_a) == 88 and len(heldout_a) == 22
    assert set(train_a).isdisjoint(heldout_a)


def test_arithmetic_supervises_only_answer_cell() -> None:
    vocab = CharVocab()
    examples = [
        next(x for x in all_arithmetic_examples() if x.expression == "3plus4?"),
        next(x for x in all_arithmetic_examples() if x.expression == "9minus3?"),
    ]
    batch = arithmetic_batch(vocab, examples)

    assert (batch.changed_mask.sum(dim=1) == 1).all()
    for row, item in enumerate(examples):
        answer_position = len(item.expression) - 1
        assert batch.input_ids[row, answer_position] == vocab.token_to_id["?"]
        assert batch.target_ids[row, answer_position] == vocab.token_to_id[str(item.answer)]
        assert batch.changed_mask[row, answer_position]


def test_guarded_spsa_acceptance_is_strictly_improving_and_retention_safe() -> None:
    vocab = CharVocab()
    model = torch.zeros(PARAMETER_COUNT, dtype=torch.int64)
    old_ids = torch.zeros((1, 64), dtype=torch.long)
    old_ids[0, 0] = vocab.token_to_id["a"]
    old_mask = torch.zeros((1, 64), dtype=torch.bool)
    old_mask[0, 0] = True
    old = TaskBatch(
        old_ids,
        old_ids.clone(),
        old_mask,
        torch.tensor([1]),
        torch.zeros_like(old_mask),
    )
    arithmetic = arithmetic_batch(
        vocab, [next(x for x in all_arithmetic_examples() if x.expression == "0plus0?")]
    )

    _, info = guarded_spsa_step(
        model,
        old,
        arithmetic,
        old,
        bytes(32),
        generation=1,
        block_size=128,
    )

    if info["accepted"]:
        assert info["proposal_total"] < info["base_total"]
        assert info["proposal_anchor"] <= info["base_anchor"]
