"""New P3 semantics only; CPU fixtures never count as canonical GPU learning."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from p3_data import episode,prepare,render,validate,seed_for,forward,forward_batch,StageSampler,digest
from p3_metrics import candidate_loss,statistics,aggregate,stage1_gate,evaluate
from p3_training import train_stage
from b0_models import construct,state_hash
from e0 import load_checkpoint
from p2_training import ObservedAdamW
from p3_run import plan


def protocol():return json.loads((BASE/'configs/protocol.json').read_text())


def tiny():
    p=protocol()
    for model in p['models'].values():model.update(vocab=536,dim=8,layers=2,heads=2,kv_heads=2,ffn=16)
    p['memory'].update(state_size=2,summary_tokens=2);p['gate']['bootstrap_draws']=30
    p['stages']['1'].update(updates=2,validation_updates=[0,1,2]);p['stages']['2'].update(updates=2,epochs=1,validation_updates=[0,1,2])
    return p


@pytest.fixture(scope='module')
def banks(tmp_path_factory):return prepare(tmp_path_factory.mktemp('p3-data'),protocol())[0]


def test_episode_layout_randomness_shift_namespaces_and_real_oracle(banks):
    for level in ('8','16'):
        train=banks[level]['train'];val=banks[level]['validation'];n=int(level)
        assert len(train)==(1024 if n==8 else 3072) and len(val)==256
        assert not {r['mapping_hash'] for r in train}&{r['mapping_hash'] for r in val}
        assert not {r['episode_hash'] for r in train}&{r['episode_hash'] for r in val}
        assert any(len(r['candidates'])<n for r in train),'Value sampling must allow replacement'
        seen={}
        for row in train[:128]:
            validate(row)
            for k,v in row['bindings']:seen.setdefault(k,set()).add(v)
            assert row['valid_length']==(59 if n==8 else 115)
            for pos,key in zip(row['query_positions'],row['query_keys']):assert row['input_ids'][pos+1]==dict(row['bindings'])[key]==row['labels'][pos]
        assert any(len(v)>1 for v in seen.values())
        assert any(row['query_keys']!=[k for k,v in row['bindings']] for row in train[:128])
        assert train[0]['mapping_hash']!=train[512]['mapping_hash']
        assert digest(train[0])==digest(episode(protocol()['master_seed'],'train',n,0,0))
    with pytest.raises(PermissionError):episode(1,'test',8,0,0)
    assert seed_for(1,'train',8,0,0)!=seed_for(1,'validation',8,0,0)


def test_candidate_only_gradient_and_three_part_decomposition():
    row=render([(8,264),(9,265)],[8],32,{'split':'train'})
    z=torch.randn(32,536,requires_grad=True);loss=candidate_loss(z,row)
    expected=torch.nn.functional.cross_entropy(z[row['query_positions']][:,[264,265]],torch.tensor([0]),reduction='sum')
    assert torch.equal(loss,expected);loss.backward()
    mask=torch.zeros_like(z,dtype=torch.bool);mask[row['query_positions'],264:266]=True
    assert not z.grad[~mask].any()
    r,_=statistics(z[row['query_positions']],row)
    assert abs(r['full_nll']-r['type_nll']-r['candidate_nll']-r['active_nll'])<1e-10
    singleton=render([(8,264),(9,264)],[9],32,{'split':'train'})
    assert candidate_loss(z,singleton)==0
    r,_=statistics(z[singleton['query_positions']],singleton)
    assert r['random_active_nll']==0 and r['random_active_accuracy']==1
    bad=copy.deepcopy(row);bad['labels'][row['query_positions'][0]]=300
    with pytest.raises(ValueError):candidate_loss(z,bad)


@pytest.mark.parametrize('arm',['E0','E6','E7-Innovation'])
def test_native_batch_isolation_causality_padding_and_accumulation(arm,banks,tmp_path):
    p=tiny();rows=banks['8']['train'][:8];model=construct(arm,p);model.eval()
    with torch.no_grad():
        baseline=forward_batch(model,rows).clone();changed=copy.deepcopy(rows)
        answer=rows[0]['query_positions'][0];changed[0]['input_ids'][answer+1:rows[0]['valid_length']]=[264]*(rows[0]['valid_length']-answer-1)
        later=forward_batch(model,changed)
        assert torch.allclose(baseline[0,:answer+1],later[0,:answer+1],atol=2e-6,rtol=2e-6)
        assert torch.equal(baseline[1:],later[1:]),'Other streams must be independent'
        if hasattr(model,'last_runtime'):
            assert model.last_runtime.position==59
            if arm=='E7-Innovation':
                assert len({id(s) for s in model.last_runtime.stores})==8
                assert all(s.stats['seen']==59 for s in model.last_runtime.stores)
    native=construct(arm,p);output=forward_batch(native,rows)
    sum(candidate_loss(output[i],row) for i,row in enumerate(rows)).div(64).backward()
    accumulated=construct(arm,p)
    for row in rows:(candidate_loss(forward(accumulated,row),row)/64).backward()
    maximum=0.
    for a,b in zip(native.parameters(),accumulated.parameters()):
        assert a.grad is not None and b.grad is not None
        maximum=max(maximum,float((a.grad-b.grad).abs().max()))
        assert torch.allclose(a.grad,b.grad,atol=3e-6,rtol=3e-5)
    (tmp_path/('batch-gradient-'+arm+'.json')).write_text(json.dumps({'scope':'tiny CPU fixture','arm':arm,'max_abs_error':maximum,'atol':3e-6,'rtol':3e-5}))


@pytest.mark.parametrize('arm',['E0','E6','E7-Innovation'])
def test_controlled_real_gradient_improves_one_query(arm,tmp_path):
    p=tiny();model=construct(arm,p);row=render([(8,264),(9,265),(10,266)],[9],32,{'split':'train'})
    before=state_hash(model);output=forward(model,row);loss=candidate_loss(output,row)
    prior=statistics(output[row['query_positions']],row)[0]['active_gold_probability']
    optimizer=ObservedAdamW(model.parameters(),lr=p['training']['lr'],betas=tuple(p['training']['betas']),weight_decay=p['training']['weight_decay'])
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
    after=forward(model,row);later=candidate_loss(after,row);assert later<loss.detach()
    assert statistics(after[row['query_positions']],row)[0]['active_gold_probability']>prior
    assert state_hash(model)!=before
    (tmp_path/('gradient-direction-'+arm+'.json')).write_text(json.dumps({'scope':'tiny CPU fixture','arm':arm,'loss_before':float(loss.detach()),'loss_after':float(later.detach()),'gold_probability_before':prior,'gold_probability_after':statistics(after[row['query_positions']],row)[0]['active_gold_probability'],'parameter_change_norm':optimizer.b0_mutations[1]}))


def test_curriculum_optimizer_global_step_and_exact_resume(banks,tmp_path):
    p=tiny();identity={'phase':'p3','fixture':'CPU curriculum','dataset_sha256':'fixture-data'};train=banks['8']['train'];val=banks['8']['validation'][:8]
    full=construct('E7-Innovation',p);train_stage(full,train,val,p,tmp_path/'full',identity,1)
    partial=construct('E7-Innovation',p);train_stage(partial,train,val,p,tmp_path/'resume',identity,1,stop_after=1)
    resumed=construct('E7-Innovation',p);train_stage(resumed,train,val,p,tmp_path/'resume',identity,1)
    assert state_hash(full)==state_hash(resumed)
    before=(tmp_path/'resume/checkpoint.pt').read_bytes();train_stage(resumed,train,val,p,tmp_path/'resume',identity,1)
    assert (tmp_path/'resume/checkpoint.pt').read_bytes()==before
    result=train_stage(resumed,banks['16']['train'],banks['16']['validation'][:8],p,tmp_path/'stage2',identity,2,continuation=tmp_path/'resume/checkpoint.pt')
    saved=load_checkpoint(tmp_path/'stage2/checkpoint.pt')
    assert saved['global_step']==4 and saved['stage_step']==2 and result['stage_ledger']['supervised_target_tokens']==256
    assert set(saved['optimizer']['b0_parameter_changes'])=={1,2,3,4}
    assert saved['ledger']['supervised_target_tokens']==384
    with pytest.raises(ValueError):train_stage(construct('E7-Innovation',p),train,val,p,tmp_path/'resume',{'phase':'wrong'},1)
    import importlib.util,os,shutil
    from p3_data import write
    from p3_run import verify_payload
    spec=importlib.util.spec_from_file_location('_p3_cpu_restore_worker',BASE/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    bundle=tmp_path/'bundle';(bundle/'checkpoints').mkdir(parents=True)
    for stage,source in [(1,tmp_path/'resume'),(2,tmp_path/'stage2')]:
        shutil.copytree(source,bundle/'training'/('stage'+str(stage)))
        os.link(source/'checkpoint.pt',bundle/('checkpoints/stage'+str(stage)+'-final.pt'))
        write(bundle/('stage'+str(stage)+'-result.json'),json.loads((source/'result.json').read_text()))
    fixture_banks={'8':{'train':train,'validation':val},'16':{'train':banks['16']['train'],'validation':banks['16']['validation'][:8]}}
    for level,data in fixture_banks.items():
        for name,rows in data.items():write(bundle/'datasets'/(level+'-'+name+'.json'),rows)
    write(bundle/'dataset-manifest.json',{'dataset_sha256':'fixture-data'})
    assert verify_payload(bundle,fixture_banks,p,identity,[1,2])['status']=='PASS'
    worker.independent_verify(construct('E7-Innovation',p),bundle/'datasets',p,bundle,identity)
    assert json.loads((bundle/'independent-restore/verification.json').read_text())['status']=='PASS'
    probability=bundle/'training/stage2/evaluation/2/probabilities.npz'
    with np.load(probability) as a:changed={k:a[k] for k in a.files}
    changed['targets']=changed['targets']+1;np.savez_compressed(probability,**changed)
    with pytest.raises(ValueError,match='labels'):verify_payload(bundle,fixture_banks,p,identity,[1,2])



def test_stage1_gate_all_five_criteria_and_candidate_collapse():
    p=protocol();v={'active_exact_match':.8,'active_bootstrap_95ci':[.7,.9],'random_active_accuracy':.13,
        'random_active_nll':2.,'active_nll':1.,'collapse_episode_fraction':0.}
    online={'correct':6144,'queries':8192,'numerically_finite':True}
    assert stage1_gate({**v,'full_exact_match':0.,'full_dominant_rate':1.},online,p)['status']=='PASS'
    for key,value in [('active_exact_match',.59),('active_bootstrap_95ci',[.12,.8]),('active_nll',1.6),('collapse_episode_fraction',.5)]:
        bad={**v,key:value};assert stage1_gate(bad,online,p)['status']=='BLOCKED'
    assert stage1_gate(v,{**online,'correct':6143},p)['status']=='BLOCKED'


def test_query_isolated_mapping_clusters_and_no_new_test_access(banks):
    rows=banks['8']['query_isolated'];assert len(rows)==256 and len({r['mapping_hash'] for r in rows})==64
    for row in rows:validate(row,all_queries=False);assert len(row['query_positions'])==1 and row['valid_length']==31


def test_shared_budget_downgrade_or_stop():
    p=protocol()
    for step,expected in [(.1,384),(3.3,320),(3.8,256),(4.5,192),(5.3,None)]:
        bench={a:{str(s):{'steps':[{'total':step}],'validation_seconds':0,'checkpoint_seconds':0} for s in (1,2)} for a in p['models_order']}
        result=plan(bench,0,p);assert result['stage2_updates']==expected
        if expected is not None:assert result['forecasts_seconds'][expected]<=6480
        else:assert result['status']=='BLOCKED_RESOURCE_LIMIT'


def test_epoch_counter_sampler_uses_new_mappings_and_restores_boundary(banks):
    rows=banks['8']['train'];p=protocol();sampler=StageSampler(rows,p['master_seed'],8);groups=[];snapshot=None
    for i in range(128):
        groups.append([r['id'] for r in sampler.propose()]);sampler.resolve(True,{'committed_updates':0},{})
        if i==63:snapshot=sampler.snapshot()
    assert len(set(v for group in groups for v in group))==1024
    restored=StageSampler(rows,p['master_seed'],8);restored.restore(snapshot);again=[]
    for _ in range(64):again.append([r['id'] for r in restored.propose()]);restored.resolve(True,{'committed_updates':0},{})
    assert again==groups[64:]


def test_variable_candidate_baselines_are_not_ln8():
    rows=[render([(8,264),(9,264)],[8],32,{'split':'validation'}),render([(8,264),(9,265)],[9],32,{'split':'validation'})]
    records=[statistics(torch.zeros(1,536),row)[0] for row in rows];r=aggregate(records,draws=30)
    assert r['random_active_accuracy']==.75 and abs(r['random_active_nll']-.5*np.log(2))<1e-12


def test_summary_identity_reconstruction():
    from recover_summary import common_identity, BRANCHES
    fields=('experiment','source_commit','template_sha256','dataset_sha256','environment_sha256','protocol_sha256','config_sha256')
    manifests=[{**dict.fromkeys(fields,'same'),'arm':arm} for arm in BRANCHES]
    assert common_identity(manifests)==dict.fromkeys(fields,'same')
    with pytest.raises(TypeError):common_identity(['overwritten-string']*3)
    manifests[1]['dataset_sha256']='different'
    with pytest.raises(ValueError):common_identity(manifests)
