"""Launcher regression with explicit fixture assets, never claimed as WikiText evidence."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run as entry


def test_entry_manifest_completed_retry_and_sealed_test(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'prepare_wikitext',lambda _: {
        'fixture_only':True,'files':{'tokenizer.json':'explicit-test-fixture'},'test_status':'SEALED_NOT_ACCESSED'})
    args=SimpleNamespace(output=tmp_path/'out',assets=tmp_path/'assets',profile='cpu-tiny',task='mqar',
                         device='cpu',publish=False,publish_only=False)
    entry.launch(args)
    manifest=json.loads((args.output/'manifest.json').read_text())
    metrics=json.loads((args.output/'metrics.json').read_text())
    assert manifest['ledger']['committed_updates']==12
    assert manifest['ledger']['input_tokens']==1536
    assert manifest['ledger']['query_target_tokens']==48
    assert metrics['checkpoint']['restored_evaluation']=='PASS'
    assert manifest['test_status']=='SEALED_NOT_ACCESSED'
    before=(args.output/'checkpoint.pt').read_bytes()
    monkeypatch.setattr(entry,'fit',lambda *a,**k: (_ for _ in ()).throw(AssertionError('completed training repeated')))
    entry.launch(args)
    assert (args.output/'checkpoint.pt').read_bytes()==before
