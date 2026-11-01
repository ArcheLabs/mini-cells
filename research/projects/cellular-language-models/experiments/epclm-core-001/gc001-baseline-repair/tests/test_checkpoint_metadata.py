"""Exercise completed-checkpoint bookkeeping without any training."""
from pathlib import Path
from unittest.mock import patch
import torch
import repair_pipeline as pipeline


def test_completed_checkpoint_metadata(tmp_path):
    model=torch.nn.Linear(1,1)
    path=tmp_path/'C1-BROADCAST.pt';path.write_bytes(b'verified-checkpoint')
    current={'tokens':250000,'updates':977,'curve':[]}
    entry={'gradient_activity':dict.fromkeys(dict(model.named_parameters()),True)}
    with patch.object(pipeline,'load_checkpoint',return_value=current),patch.object(pipeline,'checkpoint_model',return_value=current),patch.object(pipeline,'training_diagnostics',return_value={}),patch.object(pipeline,'train',side_effect=AssertionError('completed checkpoint must not train')):
        result=pipeline.learning(model,'C1-BROADCAST',[],{},entry,path,{}, {'training':{'milestone_updates':[]}},float('inf'),lambda:None)
    assert result is current
    assert entry['checkpoint_bytes']==path.stat().st_size
    assert entry['checkpoint_sha256']==pipeline.sha256_file(path)


def test_audited_source_resume(tmp_path):
    import json
    import repair_io
    old={'source':pipeline.METADATA_FIX_SOURCE,'protocol':'same','data':'same','batch':'a'}
    new={**old,'source':'metadata-hotfix'}
    report={'identity':old,'status':'INCOMPLETE','models':{'C1-BROADCAST':{'status':'INCOMPLETE','origin':'NEW_REPAIR_CONTROL'}}}
    files={'decision.json':json.dumps(report).encode(),'C1-BROADCAST.pt':b'saved-optimizer'}
    import hashlib
    manifest={'identity':old,'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
    files['recovery.json']=json.dumps(manifest).encode()
    class API:
        def metadata(self):return {'version_number':99}
        def page(self,version,token):return {'files':[{'file_name':'progress/'+n} for n in files]}
    def fetch(api,inventory,names,destination):
        for name in names:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(files[name.split('/')[-1]])
    # Existing adapter interprets metadata; isolate it to avoid network/import setup.
    import sys,types
    adapter=types.ModuleType('kaggle_output_recovery');adapter._version_number=lambda value:value['version_number']
    with patch.object(repair_io,'backend',return_value=(API(),fetch)),patch.dict(sys.modules,{'kaggle_output_recovery':adapter}):
        repair_io.recover_namespace(tmp_path,tmp_path/'out','progress',new,float('inf'),max_versions=2,resume_only=True,cache=tmp_path/'cache',compatible_identity=old)
    assert (tmp_path/'out/C1-BROADCAST.pt').read_bytes()==b'saved-optimizer'
    assert json.loads((tmp_path/'out/decision.json').read_text())['identity']==old
