"""Small metadata checks only; no trainer/model execution."""
import csv
import json
import pytest
from batch_c_analysis import checked_pairs, decision_basis
from repair_diagnostics import blind_review
from frozen_model import BaselineIntegrityError


def test_pair_integrity_and_inconclusive_decision():
    rows=[{'story_id':'same','tokens':2,'nll_sum':4.}]
    entry={'nll':2.,'paired_documents':dict.fromkeys(('validation','development_holdout'),rows)}
    models={name:{**entry,'validation':{'nll':2.},'development_holdout':{'nll':2.}} for name in ('C1-BROADCAST','IC-ALL-SELF')}
    evidence=checked_pairs(models)
    report={'statistical_evidence':evidence,'human_evaluation':{'reviews_complete':False}}
    assert decision_basis(report)['questions']['broadcast_reduces_nll_vs_independent']['status']=='INCONCLUSIVE_WITHIN_THIS_RUN'
    models['IC-ALL-SELF']['paired_documents']={'validation':rows*2,'development_holdout':rows}
    with pytest.raises(BaselineIntegrityError,match='duplicate'):checked_pairs(models)


def test_blind_review_requires_full_real_coverage(tmp_path):
    dims=['grammaticality','local_coherence','semantic_consistency','prompt_relevance','story_progression','repetition_and_degeneration']
    plan={'human_review':{'dimensions':dims}}
    models={'T1':{'status':'COMPLETE','samples':[{'prompt_index':0,'mode':{'name':'greedy'},'text':'a story','prompt':'Once'}]}}
    assert blind_review(tmp_path,models,plan)['reviews_complete'] is False
    with (tmp_path/'blind-review.csv').open() as f:sample=next(csv.DictReader(f))
    def score(reviewers):
        with (tmp_path/'human-review-scored.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['reviewer_id','sample_id']+dims);writer.writeheader()
            for reviewer in reviewers:writer.writerow({'reviewer_id':reviewer,'sample_id':sample['sample_id'],**dict.fromkeys(dims,3)})
    score(['one']);assert blind_review(tmp_path,models,plan)['reviews_complete'] is False
    score(['one','two']);assert blind_review(tmp_path,models,plan)['reviews_complete'] is True
    (tmp_path/'private/human-key.json').unlink()
    with pytest.raises(BaselineIntegrityError,match='original private blind key'):blind_review(tmp_path,models,plan)
