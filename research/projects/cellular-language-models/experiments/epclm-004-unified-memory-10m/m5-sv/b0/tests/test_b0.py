"""Focused new B0 invariants, using small original model instances only."""
from pathlib import Path
import copy
import json
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from b0_models import construct,state_hash,parameter_report
from b0_data import rows,validate_row,prepare,seal_test
from b0_metrics import query_metrics,evaluate,binomial_tail
from b0_training import fit_stage
from b0_audit import chunk_audit
from data import digest,batch
from e0 import masked_loss,load_checkpoint


def protocol():
    p=json.loads((Path(__file__).resolve().parents[1]/'configs/protocol.json').read_text())
    for config in p['models'].values():config.update(vocab=536,dim=16,layers=4,heads=2,kv_heads=2,ffn=32)
    p['models']['E0']['window']=4
    p['models']['E6_E7']['window']=4
    p['memory'].update(state_size=4,summary_tokens=2)
    p['metrics']['bootstrap_draws']=50
    return p


def sample(p,index=0):return rows(p['data'],'train',p['seeds']['overfit'],index+1)[index]


def test_eight_query_data_and_mask_contract():
    p=protocol();bank=rows(p['data'],'train',p['seeds']['overfit'],16);val=rows(p['data'],'validation',p['seeds']['validation'],4)
    assert len({tuple(pair) for r in bank for pair in r['bindings']})==128
    assert not {tuple(pair) for r in bank for pair in r['bindings']} & {tuple(pair) for r in val for pair in r['bindings']}
    b=batch([bank[0]]);assert b.input_ids.shape==(1,512) and int(b.query_mask.sum())==8
    assert torch.equal(b.query_mask,b.labels!=-100)
    logits=torch.randn(1,512,536,requires_grad=True);actual=masked_loss(logits,b.labels)
    expected=torch.nn.functional.cross_entropy(logits[b.query_mask],b.labels[b.query_mask]);assert torch.equal(actual,expected)
    actual.backward();assert not logits.grad[~b.query_mask].any()
    bad=copy.deepcopy(bank[0]);bad['labels'][1]=264
    with pytest.raises(ValueError):validate_row(bad,p['data'])
    # The three modulo partitions are disjoint by construction, without creating Test episodes.
    partitions=[{(k,v) for k in range(16) for v in range(16) if (k+v)%3==i} for i in range(3)]
    assert not any(partitions[i]&partitions[j] for i in range(3) for j in range(i))


@pytest.mark.parametrize('arm',['E0','E6','E7-Innovation'])
def test_complete_forward_causality_episode_isolation_and_real_update(arm,tmp_path):
    p=protocol();model=construct(arm,p);row=sample(p);other=sample(p,1);b=batch([row]);ids=b.input_ids
    model.train();train_logits=model(ids);model.eval()
    with torch.no_grad():
        evaluation=model(ids);assert train_logits.shape==(1,512,536) and torch.allclose(train_logits,evaluation,atol=2e-6,rtol=2e-6)
        future=ids.clone();future[:,400:]=520
        assert torch.allclose(evaluation[:,:400],model(future)[:,:400],atol=2e-6,rtol=2e-6)
        model(batch([other]).input_ids);after=model(ids);assert torch.equal(after,evaluation)
        if arm=='E0':
            with pytest.raises(ValueError):model.cached(ids)
            history=ids.clone();history[:,0]=3
            assert not torch.equal(evaluation[:,-1],model(history)[:,-1]),'E0 must read beyond the configured window=4'
        else:
            assert model.last_runtime.position==512
            if arm=='E7-Innovation':assert model.last_runtime.stores[0].stats['seen']==512
    del train_logits
    # One synthetic CPU optimizer update, not an overfit or Learnability run.
    p['training'].update(updates=1);p['overfit'].update(evaluation_every=1,minimum_correct=1)
    result=fit_stage(model,[row],[row],p,tmp_path,{'arm':arm,'scope':'CPU tiny single-step'},'overfit')
    saved=load_checkpoint(tmp_path/'checkpoint.pt')
    assert saved['ledger']['committed_updates']==1 and saved['ledger']['supervised_target_tokens']==8
    assert saved['optimizer']['b0_parameter_changes'][1]>0
    assert {'rng','sampler','optimizer','model','ledger'}<=saved.keys()
    evaluation=evaluate(model,[row]);assert evaluation['query_targets']==8


def test_paired_initialization_and_exact_chunk_semantics():
    p=protocol();a=construct('E6',p);b=construct('E7-Innovation',p)
    assert state_hash(a)==state_hash(b)
    ids=batch([sample(p)]).input_ids
    for model in (a,b):
        report=chunk_audit(model,ids);assert report['aligned_status']=='PASS'
        assert report['nonaligned_used_for_learnability'] is False
    assert not parameter_report(b)['store_trainable_parameters']


def test_known_metrics_and_test_guard(tmp_path):
    logits=torch.zeros(2,536);labels=torch.tensor([264,519]);logits[0,264]=3;logits[1,10]=4
    r=query_metrics(logits,labels)
    assert r['correct']==1 and r['value_correct']==1 and r['invalid_output_count']==1
    assert r['gold_rank'][0]==1 and r['gold_logit_margin'][0]==3
    assert abs(binomial_tail(1,1)-1/256)<1e-12
    assert binomial_tail(0,512)==1
    with pytest.raises(PermissionError):seal_test(tmp_path,protocol())
    assert not list(tmp_path.iterdir())
    with pytest.raises(PermissionError):evaluate(construct('E0',protocol()),[{'split':'test'}])


def test_resume_and_identity_protection(tmp_path):
    p=protocol();p['training'].update(updates=4);p['overfit'].update(evaluation_every=2,minimum_correct=1)
    bank=[sample(p),sample(p,1)];identity={'fixture':'B0 independent resume'}
    full=construct('E7-Innovation',p);fit_stage(full,bank,bank,p,tmp_path/'full',identity,'overfit')
    partial=construct('E7-Innovation',p);fit_stage(partial,bank,bank,p,tmp_path/'resume',identity,'overfit',stop_after=2)
    resumed=construct('E7-Innovation',p);fit_stage(resumed,bank,bank,p,tmp_path/'resume',identity,'overfit')
    assert state_hash(full)==state_hash(resumed)
    saved=load_checkpoint(tmp_path/'resume/checkpoint.pt');assert saved['ledger']['committed_updates']==4
    assert set(saved['optimizer']['b0_parameter_changes'])=={1,2,3,4}
    before=state_hash(resumed);fit_stage(resumed,bank,bank,p,tmp_path/'resume',identity,'overfit');assert state_hash(resumed)==before
    with pytest.raises(ValueError):fit_stage(construct('E7-Innovation',p),bank,bank,p,tmp_path/'resume',{'fixture':'wrong-config'},'overfit')
    assert load_checkpoint(tmp_path/'resume/checkpoint.pt')['ledger']['committed_updates']==4
