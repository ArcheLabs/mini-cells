"""Development component checks only; live gate chain still requires archive Gate 0."""
from pathlib import Path
import sys
import pytest
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'src'))
import m5_checks


@pytest.mark.parametrize('number',[1,2,3,4,5])
def test_new_component_oracles(number,tmp_path):
    path=tmp_path/str(number);path.mkdir()
    result=getattr(m5_checks,f'gate{number}')(path)
    assert result['status']=='PASS'


def test_gate0_failure_stops_all_later_stages(tmp_path,monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('m5_entry_test',BASE/'run.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    monkeypatch.setattr(entry,'verify_frozen',lambda:None)
    monkeypatch.setattr(entry.gate0,'run',lambda out:{'status':'BLOCKED'})
    def forbidden(*args,**kwargs):raise AssertionError('later stage executed after archive failure')
    monkeypatch.setattr(entry,'ordered_checks',forbidden);monkeypatch.setattr(entry,'gpu',forbidden)
    with pytest.raises(SystemExit):entry.launch(SimpleNamespace(output=tmp_path,publish_only=False,publish=False,started=None))
    import json
    decision=json.loads((tmp_path/'decision.json').read_text())
    assert decision['status']=='BLOCKED_GATE0' and list(decision['gates'])==['0']


def test_shared_data_transforms_preserve_labels_and_pair_isolation():
    import torch
    from tasks import variant
    from data import MQARConfig,FACT
    torch.manual_seed(51007);reference=torch.randn(536,16)
    values={}
    for mode in ('low','random','high'):
        row=variant(MQARConfig(length=256),'validation',4,reference,mode,True)
        bindings=dict(row['bindings'])
        assert all((k+v)%3==1 for k,v in row['bindings'])
        for p in row['query_positions']:assert row['labels'][p]==264+bindings[row['input_ids'][p-1]-8]
        for p,t in enumerate(row['input_ids'][:-2]):
            if t==FACT:assert row['input_ids'][p+2]==264+bindings[row['input_ids'][p+1]-8]
        key=row['input_ids'][-2]-8;matrix=torch.nn.functional.normalize(reference[8:264],dim=-1)
        values[mode]=float(torch.stack([matrix[key]@matrix[k] for k,v in row['bindings'] if k!=key]).mean())
    assert values['high']>values['random']>values['low']
    induction=variant(MQARConfig(length=512),'validation',5,task='induction')
    assert len(induction['input_ids'])==512 and induction['labels'][-1]==264+dict(induction['bindings'])[induction['input_ids'][-1]-8]
