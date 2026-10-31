"""Old-source data may be reused; old-source model state must not be downloaded."""
import hashlib
import json
from pathlib import Path
import pytest
from storage import recover


def test_assets_only_recovery(tmp_path,monkeypatch):
    base=Path(__file__).resolve().parents[1]
    root=next(p for p in base.parents if (p/'pyproject.toml').exists())
    monkeypatch.syspath_prepend(str(base.parent/'epclm-002-a0-persistent-state-recall/scripts'))
    import kaggle_output_recovery as backend
    digest=lambda content:hashlib.sha256(content).hexdigest()
    files={'assets/tokenizer.json':b'{"vocab":"fixture"}',
           'assets/train.json':b'[{"tokens":[1,2]}]', 'assets/development.json':b'[{"tokens":[3,4]}]'}
    metadata={'dataset_id':'fixture','dataset_revision':'pinned','requested_stories':2,
              'split':'core001-content-sha256-mod10-development',
              'files':{name.split('/')[-1]:digest(value) for name,value in files.items()}}
    files['assets/manifest.json']=json.dumps(metadata).encode()
    files['C1-pilot.pt']=b'old model/optimizer checkpoint must remain excluded'
    prefix='epclm-core-001-pilot/'
    locations={name:'recovery-objects/'+digest(content) for name,content in files.items()}
    manifest={'identity':{'source':'old','protocol':'old','data':digest(files['assets/manifest.json'])},
              'files':{name:digest(content) for name,content in files.items()},'objects':locations}
    payload={prefix+locations[name]:content for name,content in files.items()}
    payload[prefix+'recovery.json']=json.dumps(manifest).encode()
    class API:
        def metadata(self): return {'version_number':1}
        def page(self,*args): return {'files':[{'file_name':n} for n in payload],'next_page_token':None}
    requested=[]
    def fetch(api,inventory,names,destination):
        for name in names:
            requested.append(name)
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload[name])
    monkeypatch.setattr(backend,'_load_backend',lambda root:(API(),fetch))
    spec={'id':'fixture','revision':'pinned','stories':2}
    output=tmp_path/'one/epclm-core-001-development-pair'
    recover(root,output,'old','old',asset_spec=spec,cache=tmp_path/'cache',asset_namespace='epclm-core-001-pilot')
    assert not (output/'C1-pilot.pt').exists()
    assert prefix+locations['C1-pilot.pt'] not in requested
    assert (output/'assets/tokenizer.json').read_bytes()==files['assets/tokenizer.json']
    proof=json.loads((output/'asset-reuse.json').read_text())
    assert proof['status']=='VERIFIED_ASSETS_ONLY' and proof['model_optimizer_checkpoint_reused'] is False
    payload[prefix+locations['assets/tokenizer.json']]=b'corrupt'
    with pytest.raises(ValueError,match='checksum'):
        recover(root,tmp_path/'two/epclm-core-001-pilot','new','new',asset_spec=spec,cache=tmp_path/'cache2')
    # A notebook rerun can also find an old source in its current working folder.
    # Preserve that entire run before starting a new source with data assets only.
    local=tmp_path/'local/epclm-core-001-pilot'; local.mkdir(parents=True)
    for name,content in files.items():
        path=local/locations[name];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    (local/'recovery.json').write_text(json.dumps(manifest))
    (local/'pilot-report.json').write_text('{"status":"INCOMPLETE"}')
    (local/'C1-pilot.pt').write_bytes(files['C1-pilot.pt'])
    count=len(requested)
    recover(root,local,'new','new',asset_spec=spec,cache=tmp_path/'unused')
    assert len(requested)==count and not (local/'C1-pilot.pt').exists()
    archive=next(local.parent.glob('epclm-core-001-pilot-previous-*'))
    assert json.loads((archive/'pilot-report.json').read_text())['status']=='INCOMPLETE'
    assert (archive/'C1-pilot.pt').exists()
