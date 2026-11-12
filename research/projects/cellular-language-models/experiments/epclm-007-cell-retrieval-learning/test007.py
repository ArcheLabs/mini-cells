import pytest
import torch
from retrieval import episode,rank
from screen007 import analyze

def test_labels_and_no_test_access():
    row=episode('validation',73002,0)
    assert len(set(row['candidates'].tolist()))==8
    assert torch.equal(row['candidates'][row['targets']],
                       torch.tensor([dict(zip(row['keys'].tolist(),row['values'].tolist()))[k]
                                     for k in row['queries'].tolist()]))
    assert (row['values']!=row['values'].roll(1)).all()
    with pytest.raises(ValueError,match='sealed'):
        episode('test',0,0)

def test_ranking_and_paired_delta():
    values=torch.eye(8)
    assert rank(values[[4,2]],values).argmax(1).tolist()==[4,2]
    rows=[{'metrics':{'C0':{'accuracy':.25},'C1':{'accuracy':1.,'nll':0.1,'prediction':[4,2]},'C2':{'accuracy':0.}}}]
    assert analyze(rows)['C1-C2']['ci95']==[1.,1.]
