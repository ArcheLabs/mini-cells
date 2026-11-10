"""Core P2 data, instrumentation, epoch/resume and budget regression tests."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from p2_dataset import build,digest,validate_row
from p2_metrics import probability_metrics,evaluate
import p2_training
from b0_models import construct,state_hash
from b0_data import rows
from e0 import Sampler,load_checkpoint
from p2_run import checked_data,plan_budget


def protocol():return json.loads((BASE/'configs/protocol.json').read_text())


def test_balanced_data_coverage_leakage_identity_and_approved_limits(tmp_path):
    p=protocol();banks,audit,manifest=checked_data(tmp_path/'first',p)
    again,second,identity=checked_data(tmp_path/'second',p)
    assert manifest==identity and audit==second and digest(banks)==digest(again)
    assert audit['acceptance_status']=='PASS' and audit['candidate_pairs']==4252
    assert audit['train_validation_pair_overlap']==0
    for name,count,unique in [('training',512,3746),('validation',64,506)]:
        bank=banks[name];r=audit['banks'][name]
        assert len(bank)==count and r['unique_pairs']==unique and r['unique_keys']==r['unique_values']==256
        assert r['episode_duplicate_fraction']==0
        for row in bank:validate_row(row,p['data'])
    assert set(audit['banks']['validation']['unique_key_degrees'])<={1,2}
    assert set(audit['banks']['validation']['unique_value_degrees'])<={1,2}
    assert not list(tmp_path.rglob('*test*.json'))
    bad=copy.deepcopy(p);bad['p2']['observed_marginal_jsd_bits_max']=0
    with pytest.raises(ValueError,match='marginal'):checked_data(tmp_path/'rejected',bad)


def test_stable_nll_decomposition_and_prediction_collapse():
    scores=torch.zeros(16,536);scores[:,1]=1000;targets=np.array([264]*16)
    r,logp=probability_metrics(scores,targets,['a']*8+['b']*8,draws=50)
    assert r['type_nll']>990 and abs(r['value_conditional_nll']-np.log(256))<1e-10
    assert abs(r['full_vocabulary_nll']-r['type_nll']-r['value_conditional_nll'])<1e-10
    assert np.isfinite(logp).all() and r['dominant_token_rate']==1 and r['unique_predicted_tokens']==1
    assert r['episode_bootstrap_95ci']['exact_match']==[0,0]


def test_common_sampler_two_epochs_and_restore():
    bank=[{'id':str(i)} for i in range(512)];sampler=Sampler(bank,76003,1);ledger={'committed_updates':0};order=[]
    for _ in range(1024):order.append(sampler.propose()[0]['id']);sampler.resolve(True,ledger,{})
    assert set(order[:512])==set(order[512:])=={str(i) for i in range(512)}
    assert order[:512]!=order[512:]
    other=Sampler(bank,76003,1);restored=[]
    for _ in range(640):restored.append(other.propose()[0]['id']);other.resolve(True,{'committed_updates':0},{})
    state=other.snapshot();resume=Sampler(bank,76003,1);resume.restore(state)
    for _ in range(384):restored.append(resume.propose()[0]['id']);resume.resolve(True,{'committed_updates':0},{})
    assert restored==order


@pytest.mark.parametrize('arm',['E0','E6','E7-Innovation'])
def test_readonly_instrumentation_and_real_p2_resume(arm,tmp_path,monkeypatch):
    p=protocol()
    for cfg in p['models'].values():cfg.update(vocab=536,dim=8,layers=2,heads=2,kv_heads=2,ffn=16)
    p['memory'].update(state_size=2,summary_tokens=2);p['training']['updates']=4;p['metrics']['bootstrap_draws']=20
    train=rows(p['data'],'train',75023,2);val=rows(p['data'],'validation',85009,1)
    model=construct(arm,p);before=state_hash(model)
    model.eval();ids=torch.tensor([train[0]['input_ids']])
    with torch.no_grad():expected=model(ids).clone()
    result=evaluate(model,val,tmp_path/'readonly',draws=20)
    with torch.no_grad():assert torch.equal(model(ids),expected)
    assert state_hash(model)==before and result['query_targets']==8
    monkeypatch.setattr(p2_training,'boundaries',lambda _: [2,4])
    identity={'phase':'b0-p2','fixture':arm}
    full=construct(arm,p);p2_training.fit(full,train,val,train,p,tmp_path/'full',identity)
    split=construct(arm,p);p2_training.fit(split,train,val,train,p,tmp_path/'split',identity,stop_after=2)
    resumed=construct(arm,p);p2_training.fit(resumed,train,val,train,p,tmp_path/'split',identity)
    assert state_hash(full)==state_hash(resumed)!=before
    saved=load_checkpoint(tmp_path/'split/checkpoint.pt')
    assert saved['ledger']['committed_updates']==4 and saved['ledger']['supervised_target_tokens']==32
    assert type(saved['model']) is dict and set(saved['optimizer']['p2_observations'])=={1,2,3,4}
    checkpoint=(tmp_path/'split/checkpoint.pt').read_bytes()
    p2_training.fit(resumed,train,val,train,p,tmp_path/'split',identity)
    assert (tmp_path/'split/checkpoint.pt').read_bytes()==checkpoint
    with pytest.raises(ValueError):p2_training.fit(resumed,train,val,train,p,tmp_path/'split',{'phase':'b0-p2','fixture':'wrong'},stop_after=4)
    if arm=='E0':
        import os,shutil
        from e0 import write
        from p2_integrity import verify_payload
        from worker import verify
        payload=tmp_path/'payload';shutil.copytree(tmp_path/'full',payload/'training')
        (payload/'checkpoints').mkdir();os.link(payload/'training/checkpoint.pt',payload/'checkpoints/final.pt')
        write(payload/'datasets/training.json',train);write(payload/'datasets/validation.json',val)
        assert verify_payload(payload,p,train,val,identity)['status']=='PASS'
        verify(payload,construct(arm,p),train,val,train,p,identity)
        assert json.loads((payload/'independent-restored/verification.json').read_text())['status']=='PASS'
        probability=payload/'training/evaluation/4/validation/probabilities.npz'
        with np.load(probability) as arrays:changed={k:arrays[k] for k in arrays.files}
        changed['targets']=changed['targets']+1;np.savez_compressed(probability,**changed)
        with pytest.raises(ValueError,match='label'):verify_payload(payload,p,train,val,identity)



@pytest.mark.parametrize('step,chosen',[(1,1024),(2,768),(2.4,640),(3,None)])
def test_one_budget_for_all_arms_or_stop(step,chosen):
    p=protocol();bench={a:{'steps':[{'total':step}],'validation_seconds':0,'diagnostic_seconds':0,'checkpoint_seconds':0} for a in p['arms']}
    previous={a:{'history':[{'compute_seconds':0}]} for a in p['arms']}
    result=plan_budget(bench,previous,0,p)
    assert result['updates_per_arm']==chosen
    assert result['status']==('PASS' if chosen else 'BLOCKED_RESOURCE_LIMIT')


def test_archive_hardlinks_hashes_and_mutation_guard(tmp_path):
    import os,tarfile
    from p2_integrity import freeze_bundle,verify_bundle
    f=tmp_path/'checkpoints';f.mkdir();(f/'final.pt').write_bytes(b'x'*(1024*1024+1));os.link(f/'final.pt',f/'latest.pt')
    package,files=freeze_bundle(tmp_path);verify_bundle(package,files)
    with tarfile.open(package) as tar:assert tar.getmember('checkpoints/latest.pt').islnk()
    before=package.read_bytes();freeze_bundle(tmp_path);assert package.read_bytes()==before
    (f/'latest.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Immutable'):freeze_bundle(tmp_path)


def test_hard_cumulative_deadline_kills_child_and_records_failure(tmp_path,monkeypatch):
    import subprocess
    import p2_run
    actual=subprocess.Popen
    def cpu_child(*args,**kwargs):return actual([sys.executable,'-c','import time;time.sleep(5)'],**kwargs)
    monkeypatch.setattr(p2_run.subprocess,'Popen',cpu_child)
    (tmp_path/'gpu-budget.json').write_text(json.dumps({'charged_seconds':7199.9,'phases':[]}))
    with pytest.raises(TimeoutError,match='TIME_BUDGET_EXCEEDED'):
        p2_run.charged_phase(tmp_path,'benchmark','E0',tmp_path/'unused.json',tmp_path,tmp_path/'unused')
    budget=json.loads((tmp_path/'gpu-budget.json').read_text())
    assert budget['status']=='TIME_BUDGET_EXCEEDED' and 'active' not in budget
    assert budget['phases'][-1]['status']=='STOPPED'


def test_independent_download_checks_every_byte_and_rejects_bad_hash(tmp_path,monkeypatch):
    import types
    import requests
    import p2_integrity
    source=tmp_path/'source';source.mkdir();(source/'metrics.json').write_text('{"example":1}')
    package,files=p2_integrity.freeze_bundle(source);payload=package.read_bytes()
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield payload
    class Session:
        def __init__(self):self.headers={}
        def get(self,*args,**kwargs):return Response()
    monkeypatch.setattr(requests,'Session',Session)
    monkeypatch.setattr(p2_integrity,'publisher',lambda:types.SimpleNamespace(load_github_token=lambda _: 'fixture'))
    artifact={'artifact_id':1,'bytes':len(payload),'sha256':p2_integrity.file_hash(package)}
    extracted,proof=p2_integrity.download(artifact,tmp_path/'download')
    assert proof['status']=='PASS' and proof['files_verified']==1 and (extracted/'metrics.json').read_text()=='{"example":1}'
    artifact['sha256']='0'*64
    with pytest.raises(ValueError,match='HASH_MISMATCH'):p2_integrity.download(artifact,tmp_path/'bad')


def test_old_experiment_output_is_rejected_without_rewriting(tmp_path):
    from types import SimpleNamespace
    from p2_run import launch
    prior=b'{"experiment":"original-B0","status":"COMPLETED"}'
    (tmp_path/'decision.json').write_bytes(prior)
    with pytest.raises(ValueError,match='unrecognized'):launch(SimpleNamespace(output=tmp_path,execute=False))
    assert (tmp_path/'decision.json').read_bytes()==prior and not (tmp_path/'data').exists()
