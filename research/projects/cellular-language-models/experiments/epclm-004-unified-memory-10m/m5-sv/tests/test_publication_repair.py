"""Regressions for direct-script imports and experiment-free publication retries."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

BASE=Path(__file__).resolve().parents[1]


def test_publisher_loads_without_repo_cwd_despite_scripts_collision(tmp_path):
    helper=BASE/'src/sv_publication.py'
    code="""import runpy,sys,types
sys.modules['scripts']=types.ModuleType('scripts')
runpy.run_path(sys.argv[1],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-I','-c',code,str(helper)],cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'PUBLICATION_IMPORT_PREFLIGHT' in result.stdout and 'publish_experiment_results.py' in result.stdout


def test_publication_only_preserves_original_state_and_never_copies_weights(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('sv_repair_fixture',BASE/'repair_publication.py')
    repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
    source=tmp_path/'source';source.mkdir()
    decision={'run_id':'fixture','status':'BLOCKED','error':'original experiment failure',
              'identity':{'source_commit':'a'*40},'branch':'kaggle/fixture'}
    (source/'decision.json').write_text(json.dumps(decision));(source/'raw.log').write_text('original log\n')
    (source/'checkpoint.pt').write_bytes(b'not a real tensor; must never be loaded')
    original=(source/'decision.json').read_bytes();calls=[]
    def publish(out, received):
        calls.append(received)
        assert received==decision and (out/'decision.json').read_bytes()==original
        assert not (out/'checkpoint.pt').exists()
        proof={'status':'REMOTE_SHA_VERIFIED','commit':'b'*40,'branch':'kaggle/fixture'}
        (out/'publication.json').write_text(json.dumps(proof))
        return proof
    monkeypatch.setattr(repair,'verify_original',lambda path:decision)
    monkeypatch.setattr(repair,'entry_module',lambda:SimpleNamespace(publish=publish))
    receipt=repair.republish(source,tmp_path/'repair',{'version_number':190})
    assert len(calls)==1 and receipt['new_optimizer_updates']==0 and not receipt['experiment_reexecuted']
    assert (source/'decision.json').read_bytes()==original and receipt['original_error']==decision['error']
    retry=repair.republish(source,tmp_path/'repair',{'version_number':190})
    assert len(calls)==2 and retry['new_optimizer_updates']==0
    assert 'publication-repair.json' not in retry['original_metadata_sha256']
    assert 'publication.json' not in retry['original_metadata_sha256']


def test_publication_failure_does_not_hide_original_experiment_error(tmp_path,monkeypatch,capsys):
    import pytest
    sys.path.insert(0,str(BASE/'src'))
    spec=importlib.util.spec_from_file_location('sv_visibility_fixture',BASE/'run.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    monkeypatch.setattr(entry,'source_identity',lambda protocol:{'source_commit':'a'*40})
    monkeypatch.setattr(entry,'verify_baseline',lambda:None)
    monkeypatch.setattr(entry,'resume_output',lambda *a:None)
    def experiment_failure(*args):raise ValueError('original SV0 failure')
    def publication_failure(*args):raise RuntimeError('publisher failed')
    monkeypatch.setattr(entry,'functional_gate',experiment_failure)
    monkeypatch.setattr(entry,'publish',publication_failure)
    monkeypatch.setattr(entry,'report',lambda out,decision:entry.write(out/'decision.json',decision))
    with pytest.raises(RuntimeError,match='publisher failed'):
        entry.launch(SimpleNamespace(output=tmp_path,started=None,publish=True))
    printed=capsys.readouterr().out
    assert 'original SV0 failure' in printed and 'RESULT_PUBLICATION_FAILED' in printed
    decision=json.loads((tmp_path/'decision.json').read_text())
    failure=json.loads((tmp_path/'publication-error.json').read_text())
    assert decision['error']=='ValueError: original SV0 failure' and failure['experiment_error']==decision['error']


def test_real_publication_from_shallow_clone_with_fetch_head_only_source(tmp_path,monkeypatch):
    """Use real Git and artifact preparation; only the final GitHub push is replaced."""
    def git(root,*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
    seed=tmp_path/'seed';seed.mkdir();git(seed,'init');git(seed,'symbolic-ref','HEAD','refs/heads/main')
    git(seed,'config','user.name','Publication fixture');git(seed,'config','user.email','fixture@example.invalid')
    (seed/'pyproject.toml').write_text('[project]\nname="publication-fixture"\n')
    git(seed,'add','.');git(seed,'commit','-m','Original experiment source');original=git(seed,'rev-parse','HEAD')
    (seed/'later.txt').write_text('later source\n');git(seed,'add','.');git(seed,'commit','-m','Later repair source')
    remote=tmp_path/'remote.git';subprocess.run(['git','clone','--quiet','--bare',str(seed),str(remote)],check=True)
    git(remote,'config','uploadpack.allowReachableSHA1InWant','true')
    git(remote,'config','uploadpack.allowAnySHA1InWant','true')
    parent=tmp_path/'shallow';subprocess.run(['git','clone','--quiet','--depth=1',remote.as_uri(),str(parent)],check=True)
    assert git(parent,'rev-parse','--is-shallow-repository')=='true'
    git(parent,'fetch','--depth=1','origin',original)
    assert git(parent,'cat-file','-t',original)=='commit'
    sys.path.insert(0,str(BASE/'src'))
    spec=importlib.util.spec_from_file_location('sv_shallow_publication_fixture',BASE/'run.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    existing=entry.publisher_module();calls=[]
    out=tmp_path/'output';out.mkdir()
    decision={'format':'minicells.m5-sv.results.v1','identity':{'source_commit':original},
              'run_id':'shallow-fixture','branch':'kaggle/shallow-fixture','status':'BLOCKED'}
    original_bytes=json.dumps(decision).encode();(out/'decision.json').write_bytes(original_bytes)
    def local_push(root,folder,*args):
        assert git(root,'rev-parse','HEAD')==original
        assert json.loads((folder/'metadata.json').read_text())['source_commit']==original
        assert (folder/'decision.json').read_bytes()==original_bytes
        git(root,'add',str(folder.relative_to(root)))
        git(root,'-c','user.name=Publication fixture','-c','user.email=fixture@example.invalid','commit','-m','Metadata publication fixture')
        calls.append(original)
    monkeypatch.setattr(entry,'ROOT',parent)
    monkeypatch.setattr(entry,'publisher_module',lambda:SimpleNamespace(ExperimentSpec=existing.ExperimentSpec,
        prepare_artifacts=existing.prepare_artifacts,push_results=local_push))
    proof=entry.publish(out,decision)
    assert len(calls)==1 and len(proof['commit'])==40
    assert (out/'decision.json').read_bytes()==original_bytes
