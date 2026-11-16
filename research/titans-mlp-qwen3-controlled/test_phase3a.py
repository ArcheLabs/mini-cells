import torch

from phase3a import facts, metrics


def test_disjoint_deterministic_facts():
    split = [facts(42, i, 32) for i in range(3)]
    assert split[0] == facts(42, 0, 32)
    for column in (0, 2):
        assert len({row[column] for group in split for row in group}) == 96
    assert len({row[1] for row in split[0]}) == 8


def test_scores_and_ties():
    perfect = metrics(torch.eye(32))
    assert perfect['recall1'] == perfect['recall5'] == perfect['mrr'] == 1
    assert perfect['ci95'][0] > 1 / 32
    ties = metrics(torch.zeros(32, 32))
    assert ties['recall1'] == ties['recall5'] == 0
    assert ties['mrr'] == 1 / 32
