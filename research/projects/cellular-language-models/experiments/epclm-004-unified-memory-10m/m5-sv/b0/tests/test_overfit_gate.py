"""Scientific pass never follows from publication success alone."""
from pathlib import Path
import importlib.util


def test_three_arms_and_verified_publication_are_required():
    path=Path(__file__).resolve().parents[1]/'overfit.py'
    spec=importlib.util.spec_from_file_location('b0_overfit_gate_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arms=['E0','E6','E7-Innovation']
    results={a:{'status':'OVERFIT_ESTABLISHED','publication_status':'VERIFIED'} for a in arms}
    assert module.overfit_gate(results,arms)
    results['E6']['status']='OVERFIT_NOT_ESTABLISHED';assert not module.overfit_gate(results,arms)
    results['E6']['status']='OVERFIT_ESTABLISHED';results['E0']['publication_status']='FAILED'
    assert not module.overfit_gate(results,arms)
    del results['E0'];assert not module.overfit_gate(results,arms)


import json
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('state',['fresh','trained','published'])
def test_launcher_routes_new_training_and_publication_retry(tmp_path,monkeypatch,state):
    """Control-flow fixture only: no GPU, remote request or fabricated recall."""
    path=Path(__file__).resolve().parents[1]/'overfit.py'
    spec=importlib.util.spec_from_file_location('b0_overfit_routes_'+state,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    protocol=json.loads((module.BASE/'configs/protocol.json').read_text());arms=protocol['arms'];events=[]
    ledger={'committed_updates':512,'input_tokens':262144,'supervised_target_tokens':4096}
    result={'status':'OVERFIT_ESTABLISHED','ledger':ledger}
    for arm in arms:
        if state!='fresh':
            folder=tmp_path/arm;folder.mkdir()
            (folder/'run-manifest.json').write_text(json.dumps({'arm':arm}))
            (folder/'result.json').write_text(json.dumps(result))
            if state=='published':(folder/'publication.json').write_text('{}')
    monkeypatch.setattr(module,'verify_dependencies',lambda:None)
    monkeypatch.setattr(module,'prepare',lambda *a:({'overfit':[{'fixture':True}]},{'fixture':'data'}))
    monkeypatch.setattr(module,'verify_gate_a',lambda *a:{arm:{'manifest':{'initialization_sha256':'fixture-init'}} for arm in arms})
    monkeypatch.setattr(module.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(module.torch.cuda,'get_device_name',lambda:'Tesla T4 fixture')
    monkeypatch.setattr(module.torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(module.torch,'use_deterministic_algorithms',lambda *a:None)
    monkeypatch.setattr(module,'environment',lambda:{k:'fixture' for k in ('python','torch','cuda','gpu','driver','precision','tf32','amp')})
    monkeypatch.setattr(module,'construct',lambda arm,p:SimpleNamespace(cuda=lambda:object()))
    monkeypatch.setattr(module,'state_hash',lambda model:'fixture-init')
    monkeypatch.setattr(module,'publisher',lambda:SimpleNamespace(load_github_token=lambda name:'fixture'))
    import requests
    monkeypatch.setattr(requests,'get',lambda *a,**k:SimpleNamespace(status_code=404))
    def fit(model,rows,validation,p,folder,identity,*args):
        assert (folder.parent/'run-manifest.json').exists()
        events.append(('train',identity['arm']));return dict(result)
    def publish(folder):
        assert (folder/'run-manifest.json').exists() and (folder/'result.json').exists()
        events.append(('publish',folder.name));return {'publication_status':'VERIFIED'}
    monkeypatch.setattr(module,'fit_stage',fit)
    monkeypatch.setattr(module,'verify_checkpoint',lambda folder:{'ledger':ledger})
    monkeypatch.setattr(module,'publish_arm',publish)
    module.launch(SimpleNamespace(output=tmp_path,publish_only=False))
    assert [e for e in events if e[0]=='train']==([('train',a) for a in arms] if state=='fresh' else [])
    assert [e for e in events if e[0]=='publish']==[('publish',a) for a in arms]
    decision=json.loads((tmp_path/'decision.json').read_text())
    assert decision['overfit_gate']=='PASS' and decision['pilot_started'] is False
