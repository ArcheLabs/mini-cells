"""Pilot-only regression checks; no canonical training or Test generation."""
import copy
import json
import sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from preflight import BASE,audit,run
from b0_data import rows
from b0_models import construct,state_hash
from b0_training import fit_stage


def test_frozen_data_audit_and_stop_on_changed_budget_or_binding(tmp_path):
    p=json.loads((BASE/'configs/protocol.json').read_text());banks,manifest,r=run(tmp_path,p)
    assert r['pilot_train']['unique_pairs']==3746
    assert r['pilot_train']['maximum_pair_multiplicity']==3
    assert r['validation']['unique_pairs']==506 and r['train_validation_pair_overlap']==0
    assert not list(tmp_path.rglob('*test*'))
    bad=copy.deepcopy(p);bad['training']['updates']=511
    with pytest.raises(ValueError,match='budget'):audit(bad,banks,manifest)
    badbanks=copy.deepcopy(banks);badbanks['validation'][0]['bindings'][0]=banks['pilot_train'][0]['bindings'][0]
    with pytest.raises((ValueError,KeyError)):audit(p,badbanks,manifest)


@pytest.mark.parametrize('arm',['E0','E6','E7-Innovation'])
def test_pilot_boundary_resume_and_completed_idempotency(arm,tmp_path):
    p=json.loads((BASE/'configs/protocol.json').read_text())
    for config in p['models'].values():config.update(vocab=536,dim=8,layers=2,heads=2,kv_heads=2,ffn=16)
    p['memory'].update(state_size=2,summary_tokens=2)
    p['training']['updates']=2;p['pilot']['validation_updates']=[1,2];p['metrics']['bootstrap_draws']=10
    train=rows(p['data'],'train',p['seeds']['pilot_train'],2)
    validation=rows(p['data'],'validation',p['seeds']['validation'],1)
    fresh=construct(arm,p);initial=state_hash(fresh)
    full=fit_stage(fresh,train,validation,p,tmp_path/'full',{'scope':'CPU Pilot resume fixture'},'pilot')
    partial=construct(arm,p);assert state_hash(partial)==initial
    fit_stage(partial,train,validation,p,tmp_path/'resume',{'scope':'CPU Pilot resume fixture'},'pilot',stop_after=1)
    restored=construct(arm,p)
    resumed=fit_stage(restored,train,validation,p,tmp_path/'resume',{'scope':'CPU Pilot resume fixture'},'pilot')
    assert state_hash(fresh)==state_hash(restored)!=initial
    assert resumed['ledger']==full['ledger'] and resumed['ledger']['supervised_target_tokens']==16
    assert [r['update'] for r in resumed['curve']]==[1,2]
    before=(tmp_path/'resume/checkpoint.pt').read_bytes()
    fit_stage(restored,train,validation,p,tmp_path/'resume',{'scope':'CPU Pilot resume fixture'},'pilot')
    assert (tmp_path/'resume/checkpoint.pt').read_bytes()==before
