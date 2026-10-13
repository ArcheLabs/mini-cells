import copy,json
import numpy as np
import pytest
from exp006 import assets,config,publish,training
from exp006.checkpoint import identity
from exp006.reuse import canonical_sha256
from exp006.report import analyze,matrix

def bundle(tmp_path,c):
    root=tmp_path/'assets';root.mkdir()
    for role in ('train','development'):np.save(root/(role+'.npy'),np.arange(32,dtype=np.int32))
    (root/'tokenizer.json').write_text('{}')
    windows={'train':{'starts':[0,8],'groups':[0,0],'story_hashes':['train']},
             'development':{'starts':[0,8],'groups':[1,2],'story_hashes':['dev1','dev2']}}
    (root/'windows.json').write_text(json.dumps(windows))
    c['data']['source_sha256']='a'*64
    manifest={'experiment_id':'006-path-first-clm','source_parquet_sha256':'a'*64,
              'data_definition':{k:v for k,v in c['data'].items() if k not in ('manifest_sha256','source_sha256','confirmation_manifest_sha256')},
              'sequence_length':8,'vocab_size':32,'files':{n:config.digest(root/n) for n in ('train.npy','development.npy','tokenizer.json','windows.json')},'confirmation_materialized':False}
    (root/'manifest.json').write_text(json.dumps(manifest));c['data']['manifest_sha256']=config.digest(root/'manifest.json')
    return root

def test_asset_integrity_after_authorized_fixture(c,tmp_path,monkeypatch):
    # This fixture supplies no real data or permission to run an external job.
    monkeypatch.setattr(assets,'authorize',lambda *a,**kw:None)
    root=bundle(tmp_path,c);loaded=assets.load_assets(root,c,{},606,'development')
    assert len(loaded['eval_starts'])==2
    (root/'development.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt asset'):assets.load_assets(root,c,{},606,'development')

def test_registered_confirmation_still_sealed(c,monkeypatch):
    monkeypatch.setattr(config,'blockers',lambda *a:[]);c['seeds']['confirmation']=[1,2,3]
    with pytest.raises(RuntimeError,match='sealed'):config.authorize(c,{'confirmation_review_approved':False},1,'confirmation')

def test_confirmation_preparation_gate_before_source_read(c,tmp_path):
    _,p,_=config.load()
    with pytest.raises(RuntimeError,match='NOT REGISTERED'):
        assets.prepare_confirmation(c,p,1,tmp_path/'output','missing-assets','missing-source')
    assert not (tmp_path/'output').exists()

def test_registration_evidence_cannot_be_reused_for_changed_config(c,tmp_path,monkeypatch):
    _,p,_=config.load();monkeypatch.setattr(config,'BASE',tmp_path)
    e=tmp_path/'registration/evidence.json';e.parent.mkdir()
    e.write_text(json.dumps({'experiment_id':'006-path-first-clm','config_sha256':'0'*64,'thresholds':p['thresholds'],
        'user_approved':True,'seed_collision_audit_passed':True,'approval_record_sha256':'a'*64}))
    p['registration_evidence_sha256']=config.digest(e)
    assert any('does not bind' in v for v in config.blockers(c,p))

def test_training_fault_is_recorded_with_checkpoint(c,tmp_path,monkeypatch):
    original=training.make_model
    def faulty(*a,**kw):
        model=original(*a,**kw);forward=model.forward;calls=[0]
        def fail(*args,**kwargs):
            calls[0]+=1
            if model.training and calls[0]>1:raise RuntimeError('injected training failure')
            return forward(*args,**kwargs)
        model.forward=fail;return model
    monkeypatch.setattr(training,'make_model',faulty)
    run={'experiment_id':'006-path-first-clm','arm':'R2','seed':606,'role':'development'}
    tokens=np.arange(32,dtype=np.int32);a={'train':tokens,'train_starts':[0,8],'development':tokens,'eval_starts':[0,8],'eval_groups':[0,1]}
    with pytest.raises(RuntimeError,match='injected'):training._train(c,run,a,tmp_path,'cpu')
    failure=json.loads((tmp_path/'failure.json').read_text())
    assert failure['status']=='INCOMPLETE' and failure['last_checkpoint_step']==1
    assert not (tmp_path/'metrics.json').exists()

def test_failure_and_completed_publication_are_distinct_immutable_records(c,tmp_path,monkeypatch):
    monkeypatch.setattr(publish,'authorize',lambda *a:None);monkeypatch.setattr(publish,'BASE',tmp_path/'unit')
    out=tmp_path/'run';out.mkdir();c['data']['manifest_sha256']='a'*64;p={}
    i=identity('b'*40,c,p,'a'*64,'R2',606,'development')
    (out/'failure.json').write_text(json.dumps({'identity':i,'status':'INCOMPLETE','phase':'training','reason':'fixture'}))
    failed=publish.archive(out,c,p,606,'development',expected_source='b'*40)
    assert 'R2-incomplete' in str(failed) and publish.archive(out,c,p,606,'development')==failed
    (out/'metrics.json').write_text(json.dumps({'identity':i,'status':'completed','checkpoint_sha':'c'*64}))
    (out/'decode-profile.json').write_text(json.dumps({'identity':i,'checkpoint_sha':'c'*64,'parity':{'passed':True}}))
    (out/'latest.pt').write_bytes(b'not git evidence')
    completed=publish.archive(out,c,p,606,'development')
    assert completed!=failed and not (completed/'latest.pt').exists()
    with pytest.raises(ValueError,match='source'):publish.archive(out,c,p,606,'development',expected_source='d'*40)
    (out/'metrics.json').write_text((out/'metrics.json').read_text()+' ')
    with pytest.raises(FileExistsError,match='immutable'):publish.archive(out,c,p,606,'development')

def rows(c,p):
    c['data']['manifest_sha256']='a'*64;result={}
    for arm,loss in [('R0',2.),('R1',1.9),('R2',1.8),('M0',1.7)]:
        i=identity('b'*40,c,p,'a'*64,arm,606,'development')
        result[arm]={'identity':i,'status':'completed','decode_profile':{'warm_p50_ms':1.},
            'development_eval':{'sample_nll':[loss,loss],'nll':loss,'perplexity':float(np.exp(loss)),
                'window_starts':[0,8],'story_groups':[0,1],'compute':{'observed_gemm_flops_per_token':10.}}}
    return result

def test_analysis_rejects_identity_and_window_mismatch(c):
    p={};r=rows(c,p);r['R2']['identity']['source_commit']='d'*40
    with pytest.raises(ValueError,match='identities'):analyze(r,c,p)
    r=rows(c,p);r['R2']['development_eval']['story_groups']=[5,6]
    with pytest.raises(ValueError,match='unpaired'):analyze(r,c,p)

def test_complete_development_does_not_establish_path_value(c):
    _,p,_=config.load();r=rows(c,p);result=analyze(r,c,p)
    assert result['status']=='DEVELOPMENT_SIGNAL' and result['scientific_decision'] is False
    r['R2']['development_eval']['compute']['observed_gemm_flops_per_token']=11
    assert analyze(r,c,p)['status']=='DEVELOPMENT_NO_PATH_SIGNAL'
    assert matrix([{'seed':1,'status':'DEVELOPMENT_SIGNAL'}],c,p)['status']=='INCOMPLETE'


def test_direct_public_train_api_retains_registration_gate(c,tmp_path):
    with pytest.raises(RuntimeError,match='NOT REGISTERED'):
        training.train(c,{'seed':1,'role':'development'},{},tmp_path,'cuda:999')
    assert not list(tmp_path.iterdir())

def test_confirmation_matrix_distinguishes_mechanism_from_system(c):
    _,p,_=config.load();p['status']='REGISTERED'
    p['thresholds'].update(connection_ppl_ratio_ci_upper=1.,connection_cost_ratio_max=1.,state_ppl_ratio_ci_upper=1.,
        state_cost_ratio_max=1.,moe_quality_ratio_max=1.,moe_decode_p50_ratio_max=1.,intervention_nll_delta_ci_lower=0.,
        collapse_max_fraction=.8,max_unused_cells_fraction=.5,max_sampled_greedy_nll_gap=1.,min_context_read_fraction=.01)
    c['seeds']['confirmation']=[701,702,703];c['data']['confirmation_manifest_sha256']='a'*64
    reports=[]
    for seed in c['seeds']['confirmation']:
        r=rows(c,p)
        for arm,row in r.items():
            row['identity'].update(role='confirmation',seed=seed)
            row['confirmation_holdout']=row.pop('development_eval')
        r['R2']['routing_diagnostics']={'modal_path_fraction':.2,'unused_cell_fraction':.1,'context_read_fraction':.2}
        r['R2']['sampled_greedy_nll_gap']=0.
        r['R2']['interventions']={'interventions':{label:{'window_starts':[0,8],'story_groups':[0,1],
            'paired_effect':{'paired_bootstrap_95pct_ci':[.01,.02]},'order_changed_fraction':.8} for label in
            ('connection_removed','connection_shuffled','same_multiset_order_permuted')}}
        reports.append({'seed':seed,**analyze(r,c,p)})
    result=matrix(reports,c,p)
    assert result['evidence_verdict']=='PATH_MECHANISM_SIGNAL_ONLY'
    assert result['scientific_decision'] is False and result['system_superiority_supported'] is False
    reports[1]['frozen_criteria']['connection_value']=False
    assert matrix(reports,c,p)['evidence_verdict']=='NOT_SUPPORTED'
