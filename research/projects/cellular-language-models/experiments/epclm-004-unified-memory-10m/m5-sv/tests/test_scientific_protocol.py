"""Core regressions for evaluation semantics absent from the frozen M5 tests."""
from pathlib import Path
import sys
import json
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sv_data import make_episode, training_rows, audit_splits
from sv_model import ValidationModel, evaluate_episode, query, runtime_digest, observe
from sv_metrics import admission, retrieval, discrimination, summarize_queries, bootstrap_paired
from data import MQARConfig, episode, QUERY, ANSWER
from e0 import load_checkpoint
from training import fit


def tiny(policy='innovation'):
    return ValidationModel(dict(vocab=536,dim=16,layers=4,heads=2,kv_heads=2,ffn=32,window=4),
        policy=policy,capacity=4,read_top_k=2,chunk=4,state_size=4,summary_tokens=2,seed=51017,store_seed=72017)


def test_unseen_pairs_and_metadata_without_answer_inputs():
    condition=dict(name='tiny',length=64,bindings=4,queries=4)
    groups={s:[make_episode(condition,s,i,seed) for i in range(8)] for s,seed in zip(('train','validation','test'),(61,71,81))}
    assert audit_splits(groups)['status']=='PASS'
    for split,rows in groups.items():
        for row in rows:
            original=episode(MQARConfig(bindings=4,length=76,queries=4,seed={'train':61,'validation':71,'test':81}[split]),split,rows.index(row))
            assert row['history']==original['input_ids'][:original['query_positions'][0]-2]
            for q in row['queries']:
                assert q['input_ids']==[QUERY,q['key_id'],ANSWER]
                r=next(r for r in row['records'] if r['record_id'] in q['required_record_ids'])
                assert row['history'][r['source_position']]==q['expected_value_id']
    assert sum(len(r['input_ids']) for r in training_rows(61,1000000))==1000000


def test_query_snapshot_on_off_oracle_and_causal_isolation():
    row=make_episode(dict(name='tiny',length=64,bindings=4,queries=4),'validation',0,71)
    model=tiny().eval()
    with torch.no_grad():
        result=evaluate_episode(model,row,oracle=True,no_history=True)
        assert all(q['admission']['coverage']==1 for q in result['queries'])
        assert all(q['admission']['precision']==1 for q in result['queries'])
        assert len(result['no_history'])==4
        # Reverse Query order, preserving the same history and Oracle required set.
        other=evaluate_episode(model,{**row,'queries':list(reversed(row['queries']))},oracle=True)
        assert {q['query_id']:q['prediction_on'] for q in result['queries']}=={q['query_id']:q['prediction_on'] for q in other['queries']}
        _,state,_=model.forward_state(torch.tensor([row['history']]),trace=False)
        before=runtime_digest(state);q=row['queries'][0]
        enabled=query(model,state,q,True);disabled=query(model,state,q,False)
        assert runtime_digest(state)==before
        assert all(all(p<0 for p in r['selected_positions']) for r in disabled['trace'])
        assert len(enabled['trace'])==2
        with observe(model,read_enabled=True):pass
        assert runtime_digest(state)==before


def test_metrics_known_answers_ties_and_undefined_cases():
    assert admission(['a','x'],['a'],['a','b'])=={'coverage':1.,'precision':.5,'recall':.5}
    assert retrieval(['x','a'],['a','b'])=={'hit_at_1':0,'hit_at_k':1,'recall_at_k':.5,'rr':.5}
    assert discrimination([.9,.8,.2,.1],[1,1,0,0])['auroc']==1
    assert discrimination([1,1],[1,0])['auroc']==.5
    assert discrimination([1,1],[1,0])['auprc']==.5
    assert discrimination([1,2],[1,1])['auroc'] is None
    rows=[dict(query_id='a',correct_on=1,correct_off=0,retrieval=retrieval(['x'],['a'])),
          dict(query_id='b',correct_on=0,correct_off=1,retrieval=retrieval(['b'],['b']))]
    result=summarize_queries(rows,{'a':0,'b':1})
    assert result['conditional_em']==0 and result['rescue_rate']==1 and result['regression_rate']==1
    ci=bootstrap_paired({'episodeA':1,'episodeB':1},{'episodeA':0,'episodeB':0},draws=50)
    assert ci['ci95']==[1.,1.] and ci['unit']=='episode'


def test_scoped_observation_preserves_native_computation():
    model=tiny().eval();ids=torch.tensor([episode(MQARConfig(bindings=2,length=32),'validation',2)['input_ids']])
    with torch.no_grad():
        plain=model(ids)
        with observe(model,scores=True) as scores:instrumented=model(ids)
        assert torch.equal(plain,instrumented) and len(scores)==32
    import bmojo
    before=bmojo.attention_read
    with pytest.raises(RuntimeError):
        with observe(model,False):raise RuntimeError('forced observation failure')
    assert bmojo.attention_read is before


def test_existing_atomic_fit_resume_commits_same_updates(tmp_path):
    rows=[episode(MQARConfig(bindings=2,length=32,seed=61),'train',i) for i in range(4)]
    val=[episode(MQARConfig(bindings=2,length=32,seed=71),'validation',0)]
    cfg={'seed':61,'batch':1,'lr':.001,'updates':4,'precision':'fp32','optimizer':'AdamW',
         'tf32':False,'scheduler':'constant-engineering-smoke','checkpoint':{'every_updates':2,'every_seconds':120}}
    identity={'fixture':'M5-SV resume','data_seed':61}
    full=tiny();fit(full,rows,val,cfg,tmp_path/'full',identity)
    split=tiny();fit(split,rows,val,cfg,tmp_path/'split',identity,stop_after=2)
    resumed=tiny();record=fit(resumed,rows,val,cfg,tmp_path/'split',identity)
    assert record['ledger']['committed_updates']==4 and record['ledger']['input_tokens']==128
    assert all(torch.equal(a,b) for a,b in zip(full.state_dict().values(),resumed.state_dict().values()))
    saved=load_checkpoint(tmp_path/'split/checkpoint.pt');assert saved['sampler'] and saved['rng']


def entry_module():
    import importlib.util
    p=Path(__file__).resolve().parents[1]/'run.py'
    spec=importlib.util.spec_from_file_location('m5_sv_entry',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_failed_pilot_cannot_unseal_test_or_expand_budget(tmp_path,monkeypatch):
    entry=entry_module();calls=[]
    def trained(out,protocol,arm,seeds,budget,*args):
        calls.append((arm,budget))
        return {'initial_evaluation':{'nll':5.},'validation':{'nll':5.,'query_exact_match':0.,'query_targets':16},'ledger':{'input_tokens':budget}},[]
    monkeypatch.setattr(entry,'train_arm',trained)
    protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/protocol.json').read_text())
    with pytest.raises(RuntimeError,match='PILOT_INCONCLUSIVE'):entry.pilot_and_freeze(tmp_path,protocol,{}, {},float('inf'))
    assert len(calls)==4 and all(b==16384 for _,b in calls)
    assert not (tmp_path/'freeze.json').exists() and not (tmp_path/'test-access.json').exists()


def test_missing_output_remains_blocked_on_repeated_launch(tmp_path,monkeypatch):
    entry=entry_module();identity={'fixture':'missing-output','source_commit':'a'*40}
    from e0 import write
    write(tmp_path/'run-manifest.json',{'identity':identity,'requires_attached_output':True})
    monkeypatch.setattr(entry.subprocess,'check_output',lambda *a,**k:'a'*40+' refs/heads/kaggle/prior\n')
    with pytest.raises(RuntimeError,match='BLOCKED_MISSING_OUTPUT'):entry.resume_output(tmp_path,identity)


def test_scientific_verdict_never_supports_single_class_or_unstable_fit(tmp_path):
    from sv_analysis import scientific_decision
    from e0 import write
    protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/protocol.json').read_text())
    protocol['conditions']=[{'name':'fixture'}];protocol['decision']['primary_long_conditions']=['fixture'];protocol['decision']['bootstrap_draws']=20
    for seed in protocol['seeds']:
        for arm in (*protocol['arms'],'E7-Oracle'):
            for cap in ([0] if arm=='E6' else [64,128]):
                eps=[dict(episode_id='a',em=0.,em_off=0.,delta_em=0.,admission_coverage=0.,hit_at_k=0.,mrr=0.)]
                report={'episodes':eps,'recoverability':{'auroc':None},'resources':{'mean_query_latency_seconds':1.,'mean_dynamic_bytes':100}}
                write(tmp_path/str(seed['model_seed'])/arm/str(cap)/'fixture.json',report)
    decision=scientific_decision(tmp_path,protocol,{'fixture':{'stable':True}})
    assert decision['H5_STATUS']=='INCONCLUSIVE' and not decision['m6_authorized']
