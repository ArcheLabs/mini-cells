"""Focused dataset and paired-metric checks."""
from l1c import MASKS, compare, expected_order
from l1c_data import bank


def test_fixed_dataset_and_budget():
    train,val,evaluation,data=bank()
    assert data['train_heldout_pair_overlap']==0
    assert data['fact_to_first_b_answer']>data['swa_receptive_field']
    assert len(val)==32 and all(len(rows)==64 for rows in evaluation.values())
    order=expected_order(train)
    assert len(order)==32 and len({i for step in order for i in step['ids']})==128
    assert sum(len(row['input_ids']) for row in train)==65536
    assert sum(sum(row['query_mask']) for row in train)==1024
    for row in train:
        assert row['query_sources'].count('A')==row['query_sources'].count('B')==4
        assert not any(row['query_mask'][:272])


def test_paired_source_contrasts():
    values=(0.,.25,.5,1.)
    arms={}
    for arm,value in zip(MASKS,values):
        row={k:value for k in ('source_a_em','source_b_em','exact_match','episode_em')}
        row.update(id='same',nll=1-value)
        arms[arm]={'61001':[row]}
    results=compare(arms)
    assert results['source_a_em']['contrasts']['interaction']==dict(mean=.25,ci95=[.25,.25])
    assert results['source_b_em']['contrasts']['over_strongest']['mean']==.5
    assert results['nll']['contrasts']['interaction']['mean']==.25
