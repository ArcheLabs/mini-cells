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


def test_three_disjoint_partitions_and_audit_threshold_independence():
    import torch

    from phase3a_rejection import independent_partitions, rejection_audit, wilson
    from phase3a_v21_dry_run import calibrate

    rows = dataset(42,(8,64,0))['validation']
    subsets = independent_partitions(rows,42)
    assert subsets == independent_partitions(rows,42)
    entities = [{e for e,_,_ in group} for group in subsets]
    assert [len(group) for group in entities] == [32,16,16]
    assert len(set.union(*entities)) == sum(map(len,entities))
    for group in subsets:
        assert set(Counter(r for _,r,_ in group).values()) == {len(group)//8}

    def fixture(negative):
        return {
            'known': {'A1':torch.eye(4), 'B':torch.eye(4)},
            'unknown_entity': {f:torch.full((4,4),negative) for f in ('A1','B')},
            'unknown_relation': {f:torch.full((4,4),negative) for f in ('A1','B')},
        }
    low = rejection_audit(fixture(.2),fixture(.1),calibrate,.05)
    high = rejection_audit(fixture(.2),fixture(.9),calibrate,.05)
    assert low['global']['thresholds'] == high['global']['thresholds']
    assert low['global']['audit']['fpr']['rate'] == 0
    assert high['global']['audit']['fpr']['rate'] == 1
    assert high['global']['audit']['families']['B']['unknown']['unknown_entity']['count'] == 4
    assert 0 < wilson(0,16)['ci95'][1] < .25


def test_five_seed_aggregation_rejects_protocol_mismatch():
    from phase3a_five_seed import aggregate

    seeds = [42,43,44,45,46]
    audit = {key:{'rate':.25} for key in ('fpr','coverage','selective_recall1','selective_recall5')}
    reports = [dict(seed=s,test='SEALED',config_sha256='test',unavailable_candidate_sizes=[512],
                    calibration={'adapter':{'global':{'audit':audit}}}) for s in seeds]
    assert aggregate(reports,seeds)['metrics']['adapter/global/fpr']['ci95'] == [.25,.25]
    with pytest.raises(ValueError):
        aggregate(reports[:-1],seeds)
    reports[0]['test'] = 'OPEN'
    with pytest.raises(ValueError):
        aggregate(reports,seeds)
