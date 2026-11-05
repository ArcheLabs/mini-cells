"""One tiny CPU contract check; no training, real assets, network, or benchmark."""
import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
import pytest
import torch
from model import Core, Config
from runtime import weight_hash
from trainer import evaluate, save_checkpoint
import frozen_model as frozen
import mechanism_diagnostic as diagnostic
from native_clm_runtime import sha256_file


def test_frozen_diagnostic_contract(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args,**kwargs): raise AssertionError('training must not be entered')
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    monkeypatch.setattr(torch.Tensor,'backward',forbidden)
    base=Path(__file__).resolve().parents[1]
    plan=json.loads((base/'diagnostic-protocol.json').read_text())
    protocol=json.loads((base/'protocol.json').read_text())
    protocol['architecture'].update(vocab=16,cells=4,dim=8)
    protocol_path=tmp_path/'protocol.json';protocol_path.write_text(json.dumps(protocol))
    config=Config(vocab=16,cells=4,dim=8)
    original=Core(config)
    assets=tmp_path/'assets';(assets/'assets').mkdir(parents=True)
    documents=[{'id':'a','tokens':[1,2,3,4,5]},{'id':'b','tokens':[2,4,6,8]}]
    (assets/'assets/development.json').write_text(json.dumps(documents))
    (assets/'assets/tokenizer.json').write_text('{}')
    metadata={'dataset_id':protocol['data']['id'],'dataset_revision':protocol['data']['revision'],
              'split':'core001-content-sha256-mod10-development','files':{n:sha256_file(assets/'assets'/n) for n in ('tokenizer.json','development.json')}}
    (assets/'assets/manifest.json').write_text(json.dumps(metadata))
    identity={**plan['training_identity'],'protocol':sha256_file(protocol_path),'data':sha256_file(assets/'assets/manifest.json')}
    save_checkpoint({'identity':{**identity,'model':asdict(config),'target_tokens':250000},
                     'model':original.state_dict(),'tokens':250000,'updates':977,'precision':'fp32'},assets/'C1.pt')
    checkpoint={**plan['checkpoint_identity'],'sha256':sha256_file(assets/'C1.pt'),'weights_sha256':weight_hash(original)}
    class Tokenizer:
        def get_vocab_size(self): return 16
    monkeypatch.setattr(frozen,'load_tokenizer',lambda path:Tokenizer())
    model,_,docs=frozen.load_frozen_model(identity,checkpoint,assets,protocol_path,'cpu')
    before=weight_hash(model)
    with torch.inference_mode():
        assert diagnostic.functional_preflight()=='PASS_FORWARD_ONLY_NO_TRAINING'
        probes=[]; normal=diagnostic.run_documents(model,docs,float('inf'),probes=probes)
        reference=evaluate(original,docs)
        assert normal['nll']==reference['nll']
        assert all(v=='PASS' for v in diagnostic.integrity_checks(model,docs[0],float('inf')).values())
        state=torch.randn_like(model.init_state())
        saved=state.clone(); branch=state.clone();branch[:,1]=0
        model.step(branch,[2]);assert torch.equal(state,saved)
        norm=model.norm(state)
        center=model.center(norm)
        expected=state+config.alpha*model.update(torch.cat((norm,center),-1))
        torch.testing.assert_close(model.evolve(state,1,'CROSS_CELL_REMOVED'),expected,rtol=0,atol=0)
        message=center+model.left(norm)+model.right(norm)
        expected=state+config.alpha*model.update(torch.cat((norm,message),-1))
        torch.testing.assert_close(model.evolve(state,1,False),expected,rtol=0,atol=0)
        torch.testing.assert_close(model.evolve(state),original.evolve(state),rtol=0,atol=0)
        calls=[];inject=model.inject
        def counted(*args): calls.append(1);return inject(*args)
        monkeypatch.setattr(model,'inject',counted)
        local_plan={**plan,'probe_cells':[0,1,2]}
        connections,evolution=diagnostic.event_probes(model,probes[:1],local_plan,float('inf'))
        # Normal + one removed-message branch + three cell branches. Extra evolve injects nothing.
        assert len(calls)==5
        assert [r['extra_steps'] for r in evolution['single_event_extra_evolution']]==[0,4,8]
        assert evolution['single_event_extra_evolution'][0]['gain']==0
        assert weight_hash(model)==before and not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert diagnostic.comparison(normal,normal,plan)['document_bootstrap_95']==[0,0]
    with pytest.raises(frozen.BaselineIntegrityError):
        frozen.load_frozen_model(identity,{**checkpoint,'sha256':'0'*64},assets,protocol_path,'cpu')
    # Reuse the real transport interface with a tiny private-output fixture. Stop at
    # the first exact match; no broad scan after finding the required trained model.
    root=next(p for p in base.parents if (p/'pyproject.toml').exists())
    monkeypatch.syspath_prepend(str(base.parent/'epclm-002-a0-persistent-state-recall/scripts'))
    import kaggle_output_recovery as backend
    namespace=checkpoint['namespace']+'/'
    names=['C1.pt','assets/manifest.json','assets/tokenizer.json','assets/development.json']
    manifest={'identity':identity,'files':{n:sha256_file(assets/n) for n in names}}
    payload={namespace+n:(assets/n).read_bytes() for n in names}
    payload[namespace+'recovery.json']=json.dumps(manifest).encode()
    visited=[]
    class API:
        def metadata(self): return {'version_number':2}
        def page(self,version,token):
            visited.append(version)
            assert version==2, 'must stop scanning once exact trained assets are found'
            return {'files':[{'file_name':n} for n in payload]}
    requested=[]
    def fetch(api,inventory,names,destination):
        for name in names:
            requested.append(name);target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload[name])
    monkeypatch.setattr(backend,'_load_backend',lambda root:(API(),fetch))
    recovered=tmp_path/'recovered'
    frozen.recover_frozen_assets(root,recovered,identity,checkpoint,float('inf'))
    assert visited==[2] and sha256_file(recovered/'C1.pt')==checkpoint['sha256']
    assert not (recovered/'assets/train.json').exists()
    frozen.load_frozen_model(identity,checkpoint,recovered,protocol_path,'cpu')
    # A completed saved diagnostic is recovered for publication without loading a model.
    progress=tmp_path/'diagnostic'; progress_identity={'source':'diagnostic-source','protocol':'plan','data':identity['data']}
    result=json.dumps({'status':'COMPLETE','identity':{'fixture':True}}).encode()
    import hashlib
    payload.clear();payload.update({progress.name+'/decision.json':result,
        progress.name+'/recovery.json':json.dumps({'identity':progress_identity,
        'files':{'decision.json':hashlib.sha256(result).hexdigest()}}).encode()})
    # A model-cache manifest must not take priority over a completed diagnostic.
    payload['another-cache/training-recovery.json']=b'not read: progress has priority'
    terminal=frozen.recover_frozen_assets(root,tmp_path/'another-cache',identity,checkpoint,float('inf'),
                                         progress_output=progress,progress_identity=progress_identity)
    assert terminal=='DIAGNOSTIC_TERMINAL' and json.loads((progress/'decision.json').read_text())['status']=='COMPLETE'
    # Exercise the actual CLI branch: it exits before launch()/CPU training preflight.
    called=[]
    monkeypatch.setattr(diagnostic,'launch_diagnostic',lambda *args:called.append(args))
    monkeypatch.setattr(sys,'argv',['run.py','--diagnose-c1','--device','cpu','--output',str(tmp_path/'output')])
    with pytest.raises(SystemExit) as exit_status: runpy.run_path(str(base/'run.py'),run_name='__main__')
    assert exit_status.value.code==0 and len(called)==1
