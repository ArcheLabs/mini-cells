"""One mocked transport check; no model execution, downloads or training."""
import hashlib
import json
from pathlib import Path
import repair_io as recovery


def test_fast_inputs_and_selective_resume(tmp_path,monkeypatch):
    base=Path(__file__).resolve().parents[1];root=next(p for p in base.parents if (p/'pyproject.toml').exists())
    monkeypatch.syspath_prepend(str(base.parent.parent/'epclm-002-a0-persistent-state-recall/scripts'))
    import kaggle_output_recovery as transport
    sha=lambda data:hashlib.sha256(data).hexdigest()
    files={'assets/tokenizer.json':b'{}','assets/train.json':b'[]','assets/development.json':b'[]','C1.pt':b'historical-c1','F0.pt':b'historical-flat'}
    metadata={'files':{name.split('/')[-1]:sha(files[name]) for name in files if name.startswith('assets/')}}
    files['assets/manifest.json']=json.dumps(metadata).encode()
    identity={'source':'original','protocol':'original','data':sha(files['assets/manifest.json'])}
    manifest={'identity':identity,'files':{name:sha(data) for name,data in files.items()}}
    payload={'training-pair/'+name:data for name,data in files.items()};payload['training-pair/recovery.json']=json.dumps(manifest).encode()
    initialized=[];requested=[];versions=[]
    class API:
        def metadata(self):return {'version_number':99}
        def page(self,version,token):
            versions.append(version);return {'files':[{'file_name':name} for name in payload]}
    def fetch(api,inventory,names,destination):
        for name in names:
            requested.append(name);target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload[name])
    def backend(root):initialized.append(1);return API(),fetch
    monkeypatch.setattr(transport,'_load_backend',backend);monkeypatch.setattr(recovery,'_BACKENDS',{})
    gc={'training_identity':identity,'C1_checkpoint':{'namespace':'training-pair','sha256':sha(files['C1.pt'])},'F0_checkpoint':{'sha256':sha(files['F0.pt'])}}
    out=tmp_path/'assets';trace={}
    proof=recovery.recover_inputs(root,out,gc,[142],float('inf'),trace=trace)
    assert versions==[142] and len(initialized)==1 and not any(n.endswith('.pt') for n in requested)
    assert len(proof['files'])==4 and proof['historical_weights_requested'] is False
    before=len(requested);recovery.recover_inputs(root,out,gc,[142],float('inf'))
    assert len(requested)==before
    (out/'assets/train.json').write_bytes(b'corrupt')
    recovery.recover_inputs(root,out,gc,[142],float('inf'))
    assert (out/'assets/train.json').read_bytes()==b'[]' and len(initialized)==1
    # A matching progress bundle only transfers the unfinished model; completed
    # checkpoints/milestones stay at their recorded private version locator.
    current={'source':'repair','protocol':'repair','data':identity['data'],'batch':'a'}
    report={'identity':current,'status':'INCOMPLETE','models':{
        'C1-BROADCAST':{'status':'COMPLETE','origin':'NEW_REPAIR_CONTROL'},
        'IC-ALL-SELF':{'status':'INCOMPLETE','origin':'NEW_REPAIR_CONTROL'}}}
    progress={'decision.json':json.dumps(report).encode(),'C1-BROADCAST.pt':b'complete-model',
              'IC-ALL-SELF.pt':b'unfinished-model','IC-ALL-SELF-milestone-250.pt':b'old-milestone'}
    payload.clear();payload.update({'progress/'+n:data for n,data in progress.items()})
    payload['progress/recovery.json']=json.dumps({'identity':current,'files':{n:sha(data) for n,data in progress.items()}}).encode()
    requested.clear();versions.clear();trace={}
    destination=tmp_path/'progress'
    recovery.recover_namespace(root,destination,'progress',current,float('inf'),max_versions=2,trace=trace,resume_only=True,cache=tmp_path/'downloads')
    assert versions==[99] and len(initialized)==1
    assert (destination/'IC-ALL-SELF.pt').read_bytes()==b'unfinished-model'
    assert not (destination/'C1-BROADCAST.pt').exists() and not (destination/'IC-ALL-SELF-milestone-250.pt').exists()
    assert trace['deferred_private_assets']['version']==99
