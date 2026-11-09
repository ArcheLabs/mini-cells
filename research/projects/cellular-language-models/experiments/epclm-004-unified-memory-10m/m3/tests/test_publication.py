"""Real git regression: publish twice without altering the experiment checkout."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('m3_publication_entry',BASE/'run.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
sys.path.insert(0,str(entry.ROOT))
from scripts.research import publish_experiment_results as publisher


def test_two_publications_preserve_source_and_completed_evidence(tmp_path,monkeypatch):
    def git(root,*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.PIPE).strip()
    root=tmp_path/'source';root.mkdir();remote=tmp_path/'remote.git'
    git(tmp_path,'init','--bare',str(remote));git(root,'init');git(root,'symbolic-ref','HEAD','refs/heads/main')
    git(root,'config','user.name','Test');git(root,'config','user.email','test@example.invalid')
    (root/'README.md').write_text('baseline\n');git(root,'add','.');git(root,'commit','-m','baseline')
    git(root,'remote','add','origin',str(remote));git(root,'push','origin','main')
    git(root,'switch','-c','research/m3');(root/'implementation.txt').write_text('source\n')
    git(root,'add','.');git(root,'commit','-m','source');source=git(root,'rev-parse','HEAD')
    monkeypatch.setattr(entry,'ROOT',root)
    calls=[]
    def local_push(isolated,destination,experiment,branch,secret,base):
        assert isolated!=root and base=='origin/main'
        git(isolated,'fetch','origin','main:refs/remotes/origin/main')
        git(isolated,'switch','-C',branch,'origin/main')
        git(isolated,'config','user.name','Test');git(isolated,'config','user.email','test@example.invalid')
        git(isolated,'add',str(destination.relative_to(isolated)));git(isolated,'commit','-m','results')
        git(isolated,'push','--force','origin',f'HEAD:refs/heads/{branch}')
        assert git(isolated,'ls-remote','origin',f'refs/heads/{branch}').split()[0]==git(isolated,'rev-parse','HEAD')
        calls.append(git(isolated,'rev-parse','HEAD'))
    monkeypatch.setattr(publisher,'push_results',local_push)
    out=tmp_path/'output';out.mkdir()
    decision={'format':'minicells.epclm004.m3.v1','identity':{'source':source},
              'run_id':'test-completed','branch':'kaggle/test-m3','status':'GPU_SMOKE_PASS_PUBLICATION_PENDING'}
    (out/'evidence.json').write_text('{"completed_updates":2}')
    entry.write(out/'decision.json',decision)
    entry.finish_publication(out,decision)
    assert json.loads((out/'publication.json').read_text())['status']=='REMOTE_SHA_VERIFIED'
    assert git(root,'rev-parse','HEAD')==source and git(root,'status','--porcelain')==''
    assert len(calls)==2 and calls[0]!=calls[1]
    path='artifacts/experiments/epclm004-m3/test-completed/'
    assert json.loads(git(remote,'show',f'kaggle/test-m3:{path}decision.json'))['status']=='M3_ENGINEERING_COMPLETE'
    assert git(remote,'show',f'kaggle/test-m3:{path}evidence.json')=='{"completed_updates":2}'
