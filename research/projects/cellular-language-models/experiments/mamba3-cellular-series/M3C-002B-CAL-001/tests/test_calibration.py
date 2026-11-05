"""Only task leakage, real update/resume and current-version binding regressions."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch

BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE),str(BASE.parent/'M3C-002')]
from data import build,validate,episode
from identity import validate as validate_identity
from train import fit,file_sha


def test_task_oracle_no_feedback_and_split_isolation():
    a=build('train-dev',12,2710101);b=build('validation-dev',12,2710101)
    assert validate(a,b)['oracle_exact_match']==1
    with pytest.raises(ValueError):episode('formal-test',0,2710101)
    bad=(a[0].copy(),a[1],a[2]);q=a[2][3][0];bad[0][3,q['answer_position']+1]=q['target']
    with pytest.raises(AssertionError):validate(bad,b)


def test_two_real_updates_and_completed_checkpoint_resume(tmp_path):
    from mamba3 import Mamba3Config
    from model import Model
    sys.path.insert(0,str(BASE.parent.parent/'native-clm-v0/implementation'))
    from native_clm_runtime import _atomic_save
    p=json.loads((BASE/'protocol.json').read_text());p=copy.deepcopy(p)
    p['training'].update(updates_per_arm=2,episodes_per_arm=4,batch_episodes=2,targets_per_arm=32,evaluation_updates=[0,2],checkpoint_every_updates=1)
    cfg=Mamba3Config(d_model=16,n_layer=1,d_state=8,headdim=8,chunk_size=16,vocab_size=256)
    model=Model(cfg,window=16,seed=27101);before=model.base.backbone.embedding.weight.detach().clone()
    data=build('train-dev',4,2710101);dev=build('validation-dev',12,2710101)
    report=fit(model,data,dev,p,tmp_path,'fixture-source-protocol',float('inf'),_atomic_save)
    assert report['targets']==32 and report['updates']==2 and not torch.equal(before,model.base.backbone.embedding.weight)
    digest=file_sha(tmp_path/'checkpoint.pt')
    with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('completed model retrained')):
        resumed=fit(Model(cfg,window=16,seed=27101),data,dev,p,tmp_path,'fixture-source-protocol',float('inf'),_atomic_save)
    assert resumed==report and file_sha(tmp_path/'checkpoint.pt')==digest
    with pytest.raises(ValueError):fit(model,data,dev,p,tmp_path,'wrong-source',float('inf'),_atomic_save)


def test_script_version_must_be_current_and_source_bound():
    assert validate_identity({'123456'},123456,'TAG','TAG','TAG')=='123456'
    for ids,saved,runtime,source in [(set(),123456,'TAG','TAG'),({'123456'},999999,'TAG','TAG'),({'123456'},123456,'TAG','OLD')]:
        with pytest.raises(RuntimeError):validate_identity(ids,saved,runtime,source,'TAG')


def test_failed_identity_still_publishes_without_execution_object(tmp_path,monkeypatch):
    sys.path.insert(0,str(next(p for p in BASE.parents if (p/'scripts').exists())))
    import run as launcher
    from scripts.research import publish_experiment_results as publisher
    out=tmp_path/'output';out.mkdir();(out/'decision.json').write_text('{}')
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    with patch.object(publisher,'prepare_artifacts',return_value=tmp_path) as prepare, \
         patch.object(publisher,'push_results'), \
         patch.object(publisher,'remote_branch_sha',return_value='a'*40), \
         patch.object(launcher.subprocess,'check_output',return_value='a'*40):
        launcher.publish(out,{'run_id':'blocked-fixture','execution':None,'status':'BLOCKED_EXECUTION_IDENTITY'})
    assert prepare.call_args.args[3] is None
    assert json.loads((out/'publication-verification.json').read_text())['status']=='REMOTE_SHA_VERIFIED'


def test_development_missing_id_is_explicitly_pending_only_when_authorized(tmp_path,monkeypatch):
    import identity
    script=tmp_path/'script.py';script.write_text('TAG')
    monkeypatch.setattr(identity,'Path',lambda _:script)
    monkeypatch.setitem(sys.modules,'kaggle_session',None)
    with patch.dict(identity.os.environ,{},clear=True):
        record=identity.capture(tmp_path,'bingjiangli/minicells','TAG',allow_deferred=True)
        assert record['status']=='ID_PENDING' and record['script_version_id'] is None
        assert record['formal_training_allowed'] is False
        with pytest.raises(RuntimeError):identity.capture(tmp_path,'bingjiangli/minicells','TAG')
        with pytest.raises(RuntimeError):identity.capture(tmp_path,'bingjiangli/minicells','OTHER',allow_deferred=True)


def test_saved_version_supplement_never_claims_independent_verification():
    from identity import saved_version_record
    decision={'run_id':'fixture','source_commit':'abc','execution':{'kernel':'bingjiangli/minicells','status':'ID_PENDING'}}
    record=saved_version_record('https://www.kaggle.com/code/bingjiangli/minicells?scriptVersionId=123456',decision)
    assert record['script_version_id']=='123456' and record['run_id']=='fixture'
    assert record['independent_platform_verification']=='NOT_PERFORMED'
    conflicting={**decision,'execution':{'kernel':'bingjiangli/minicells','script_version_id':'999999'}}
    with pytest.raises(ValueError):saved_version_record('https://www.kaggle.com/code/bingjiangli/minicells?scriptVersionId=123456',conflicting)
    for url in ('https://www.kaggle.com/code/other/notebook?scriptVersionId=123456',
                'https://www.kaggle.com/code/bingjiangli/minicells?version=183'):
        with pytest.raises(ValueError):saved_version_record(url,decision)


def test_completed_pending_run_republishes_without_entering_training(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import run as launcher
    old={'status':'DEVELOPMENT_CALIBRATION_COMPLETE_ID_PENDING','source_commit':'a'*40,'execution':{'status':'ID_PENDING'},'run_id':'fixture'}
    path=tmp_path/'decision.json';path.write_text(json.dumps(old));before=path.read_bytes()
    args=SimpleNamespace(started=None,output=tmp_path,publish_only=False,saved_version_url=None)
    with patch.object(launcher.subprocess,'check_output',return_value='a'*40), \
         patch.object(launcher,'publish') as publish:
        launcher.launch(args)
    assert publish.call_args.args==(tmp_path,old) and path.read_bytes()==before
    with patch.object(launcher.subprocess,'check_output',return_value='b'*40), pytest.raises(ValueError):
        launcher.launch(args)
    assert path.read_bytes()==before
