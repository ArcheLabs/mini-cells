"""Focused checks for evaluation identity, paired interaction and L1 budget."""
import pytest

from screening import MASKS, P, datasets, compare, CONFIG, Sampler, digest


def test_l1_data_budget_and_pair_isolation():
    splits,evaluation,manifest=datasets()
    assert manifest['train_heldout_pair_overlap']==0
    assert manifest['minimum_fact_query_delay']>manifest['swa_receptive_field']
    assert [len(bank) for bank in evaluation.values()]==[64]*3
    assert len(splits['validation'])==32
    sampler=Sampler(splits['train'],CONFIG['seed'],CONFIG['batch'])
    ids=[]
    ledger={'committed_updates':0}
    for _ in range(CONFIG['updates']):
        rows=sampler.propose()
        ids.extend(r['id'] for r in rows)
        sampler.resolve(True,ledger,{})
    assert len(ids)==len(set(ids))==128
    assert sum(len(r['input_ids']) for r in splits['train'])==65536
    assert sum(sum(r['query_mask']) for r in splits['train'])==1024


def test_paired_interaction_and_misalignment():
    arms={arm:{'61001':[dict(id='episode',exact_match=score,nll=1-score)]}
          for arm,score in zip(MASKS,(0,.25,.5,1))}
    result=compare({'delayed':arms})['delayed']
    assert result['exact_match']['interaction']==dict(mean=.25,ci95=[.25,.25])
    assert result['nll']['interaction']['mean']==.25
    arms['F10']['61001'][0]['id']='other'
    with pytest.raises(AssertionError):
        compare({'delayed':arms})
