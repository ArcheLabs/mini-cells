from collections import Counter

import pytest

from phase3a_rejection import split_validation, unknown_queries
from phase3a_v21_data import dataset


@pytest.mark.parametrize('entities', [8, 64])
def test_entity_isolation_balance_and_unique_negative_facts(entities):
    rows = dataset(42, (8, entities, 0))['validation']
    selection, calibration = split_validation(rows, 42)
    assert len(selection) == len(calibration) == len(rows) // 2
    assert {e for e, _, _ in selection}.isdisjoint(e for e, _, _ in calibration)
    for partition in (selection, calibration):
        assert set(Counter(r for _, r, _ in partition).values()) == {len(partition)//8}
    negatives = unknown_queries(calibration)
    assert set(negatives) == {'unknown_entity','unknown_relation'}
    for rows in negatives.values():
        assert len(set(rows)) == len(rows) == len(calibration)
        assert set(Counter(r for _, r, _ in rows).values()) == {len(rows)//8}
    assert {(e,r) for e,r,_ in negatives['unknown_relation']}.isdisjoint(
        (e,r) for e,r,_ in calibration)
