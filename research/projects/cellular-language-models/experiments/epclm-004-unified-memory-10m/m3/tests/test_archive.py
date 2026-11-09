"""Archive integrity/remote read-back and the frozen training boundary only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import pytest

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m3_archive',BASE/'archive.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(ROOT))
from scripts.research import publish_experiment_results as publisher


def test_bundle_roundtrip_and_reject_corrupted_cell(tmp_path,monkeypatch):
    base=tmp_path/'m3';(base/'freeze').mkdir(parents=True)
    (base.parent/'assets').mkdir();(base.parent/'assets/tokenizer.json').write_bytes(b'tokenizer-fixture')
    (base/'freeze/BASELINE.json').write_text('{"fixture_only":true}')
    source=tmp_path/'output';expected={}
    for name in ('training/checkpoint.pt','recall-cell/generation/fast_weights.pt','recall-cell/generation/momentum.pt'):
        path=source/'gpu'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('fixture-'+name).encode())
        expected[name]=archive.sha256(path)
    frozen={'run_id':'fixture-not-real-weights','execution':{'script_version_id':367010882},
            'training_source_commit':'0'*40,'expected_gpu_files':expected,
            'tokenizer_sha256':archive.sha256(base.parent/'assets/tokenizer.json'),
            'archive':{'asset':'fixture.tar.gz'}}
    monkeypatch.setattr(archive,'BASE',base)
    path,receipt=archive.pack(source,tmp_path/'archive',frozen)
    assert receipt['status']=='LOCAL_ARCHIVE_READBACK_VERIFIED' and not receipt['remote_archive_verified']
    with tarfile.open(path) as tar:
        m=json.load(tar.extractfile('archive-manifest.json'));assert m['new_training_updates']==0
        assert 'gpu/recall-cell/generation/momentum.pt' in m['files']
    original=path.read_bytes();archive.pack(source,path.parent,frozen);assert path.read_bytes()==original
    (source/'gpu/recall-cell/generation/momentum.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='mismatched'):archive.pack(source,tmp_path/'rejected',frozen)
    assert not (tmp_path/'rejected').exists()


def test_existing_remote_asset_is_read_back_and_corruption_rejected(tmp_path,monkeypatch):
    (tmp_path/'pyproject.toml').touch();base=tmp_path/'m3';base.mkdir();monkeypatch.setattr(archive,'BASE',base)
    monkeypatch.setattr(publisher,'load_github_token',lambda name:'fixture-only-token')
    path=tmp_path/'fixture.tar.gz';path.write_bytes(b'actual-fixture-archive')
    receipt={'sha256':archive.sha256(path),'bytes':path.stat().st_size}
    frozen={'training_source_commit':'a'*40,'archive':{'repository':'ArcheLabs/mini-cells','tag':'fixture'}}
    class Response:
        status_code=200
        def __init__(self,value=None,body=None):self.value=value;self.body=body
        def raise_for_status(self):pass
        def json(self):return self.value
        def iter_content(self,size):yield self.body
    class Session:
        headers={}
        corrupt=False
        proof=None
        def get(self,url,**kwargs):
            if '/releases/tags/' in url:return Response({'assets':[{'name':path.name,'id':1,'browser_download_url':'https://example.invalid/fixture'}],'html_url':'https://example.invalid/release','upload_url':'https://example.invalid/upload{?name}'})
            if '/git/ref/tags/' in url:return Response({'object':{'type':'commit','sha':'a'*40}})
            if url.endswith('/2'):return Response(body=self.proof)
            return Response(body=b'corrupted' if self.corrupt else path.read_bytes())
        def post(self,*args,**kwargs):
            assert kwargs['params']['name']=='archive-receipt.json'  # Existing binary is never replaced.
            self.proof=kwargs['data'];return Response({'id':2})
    import requests
    session=Session();monkeypatch.setattr(requests,'Session',lambda:session)
    assert archive.upload(path,receipt,frozen)['remote_archive_verified']
    session.corrupt=True
    with pytest.raises(ValueError,match='read-back'):archive.upload(path,receipt,frozen)


def test_frozen_entry_cannot_start_training(tmp_path):
    result=subprocess.run([sys.executable,str(BASE/'run.py'),'--output',str(tmp_path/'blocked')],capture_output=True,text=True)
    assert result.returncode!=0 and 'M3_FROZEN_NO_FURTHER_TRAINING' in result.stderr
    assert not list(tmp_path.rglob('checkpoint.pt'))
