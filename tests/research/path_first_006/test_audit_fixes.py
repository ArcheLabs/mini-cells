import copy,itertools,json
import numpy as np
import pytest,torch
from torch.nn import functional as F
from exp006.baseline import make_model
from exp006.training import objective,_train
from exp006.accounting import ComputeCounter,summarize,module_size
from exp006.evaluation import diagnostics,intervention_suite
from exp006.assets import load_assets
from exp006 import assets as asset_module
from exp006.config import digest


def test_joint_cell_router_gradient_matches_exact_expectation(c):
    cfg={**c['model'],'cell_count':1};model=make_model(cfg,'R2',42)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.ndim>=2 and 'critic' not in name:p.normal_(std=.25)
    schedule={**c['training'],'critic_coefficient':0.,'entropy_coefficient':0.}
    x=torch.tensor([[3]]);y=torch.tensor([[4]])
    params=[p for name,p in model.named_parameters() if 'critic' not in name]
    exact=0.;estimate=0.;pathwise=0.
    for path in itertools.product(range(2),repeat=2):
        out=model(x,replay=torch.tensor(path).reshape(1,1,2));prob=out.log_prob.sum().exp()
        ce=F.cross_entropy(out.logits.float().reshape(1,-1),y.flatten())
        loss,_=objective(out,y,schedule)
        exact=exact+prob*ce;estimate=estimate+prob.detach()*loss;pathwise=pathwise+prob.detach()*ce
    expected=torch.autograd.grad(exact,params,retain_graph=True)
    actual=torch.autograd.grad(estimate,params,retain_graph=True)
    for a,b in zip(expected,actual):assert torch.allclose(a,b,atol=2e-6,rtol=3e-5)
    cell=model.cells[0].up.weight
    complete=torch.autograd.grad(exact,cell,retain_graph=True)[0]
    local=torch.autograd.grad(pathwise,cell)[0]
    assert float((complete-local).abs().max())>1e-6  # real policy dependence on Cell-computed state
    index=int((complete-local).abs().argmax());original=float(cell.flatten()[index]);eps=.002
    def expected_value():
        total=0.
        for path in itertools.product(range(2),repeat=2):
            out=model(x,replay=torch.tensor(path).reshape(1,1,2))
            total+=float(out.log_prob.sum().exp()*F.cross_entropy(out.logits.float().reshape(1,-1),y.flatten()))
        return total
    with torch.no_grad():
        cell.flatten()[index]=original+eps;plus=expected_value()
        cell.flatten()[index]=original-eps;minus=expected_value();cell.flatten()[index]=original
    assert abs((plus-minus)/(2*eps)-float(complete.flatten()[index]))<2e-4


def test_confirmation_has_no_intermediate_evaluation_and_loads_only_after_checkpoint(c,tmp_path):
    tokens=np.arange(32,dtype=np.int32);run={'experiment_id':'006-path-first-clm','arm':'R1','seed':606,'role':'confirmation','data_sha256':'a'*64}
    initial={'train':tokens,'train_starts':[0,8]};calls=[]
    def sealed_loader():
        payload=torch.load(tmp_path/'latest.pt',map_location='cpu',weights_only=False)
        assert payload['step']==c['training']['steps']
        history=json.loads((tmp_path/'learning_curve.json').read_text())
        assert all('greedy_eval' not in row and 'sampled_eval' not in row for row in history)
        calls.append(payload['step'])
        return {**initial,'confirmation':tokens,'eval_starts':[0,8],'eval_groups':[0,1],'manifest_sha256':'a'*64}
    result=_train(c,run,initial,tmp_path,'cpu',final_assets_loader=sealed_loader)
    assert calls==[2] and result['holdout_read_phase']=='after_final_checkpoint'
    assert result['confirmation_holdout'] is not None


def test_confirmation_asset_loader_does_not_open_or_hash_token_file_by_default(c,tmp_path,monkeypatch):
    from test_contract_and_failures import bundle
    root=bundle(tmp_path,c);m=json.loads((root/'manifest.json').read_text());m['confirmation_materialized']=True
    windows=json.loads((root/'windows.json').read_text());windows['confirmation']={'story_hashes':['sealed'],'starts':[0,8],'groups':[3,4]}
    (root/'windows.json').write_text(json.dumps(windows));m['files']['windows.json']=digest(root/'windows.json')
    m['files']['confirmation.npy']='0'*64  # nonexistent file must remain unread during training
    (root/'manifest.json').write_text(json.dumps(m));c['data']['confirmation_manifest_sha256']=digest(root/'manifest.json')
    monkeypatch.setattr(asset_module,'authorize',lambda *a,**kw:None)
    loaded=load_assets(root,c,{},606,'confirmation')
    assert 'confirmation' not in loaded and loaded['evaluation_loaded'] is False
    with pytest.raises(FileNotFoundError):load_assets(root,c,{},606,'confirmation',include_evaluation=True)


def test_context_first_read_and_never_read_diagnostics():
    actions=torch.tensor([[[0,1,0],[2,0,2],[0,2,1]]])
    result=diagnostics(actions,2)
    assert result['first_context_step_histogram']==[1,1,1,0]
    assert result['no_context_fraction']==pytest.approx(1/3)


@pytest.mark.parametrize('arm',['R1','R2'])
def test_symmetric_order_and_context_interventions(arm,c):
    model=make_model(c['model'],arm).eval()
    suite=intervention_suite(model,np.arange(32,dtype=np.int32),[0,8],[0,1],c)
    assert {'same_multiset_order_permuted','context_noop_same_path'}<=set(suite['interventions'])
    if arm=='R2':
        repeats=[r for label,r in suite['interventions'].items() if label.startswith('connection_prior_row_permuted_seed_')]
        assert len(repeats)==3
        for row in repeats:assert sorted(row['predecessor_row_permutation'])==list(range(5))
    assert suite['natural']['context_read_nll_probe']['calls']>=0


def test_exact_inventory_and_repeat_execution_work(c):
    model=make_model(c['model'],'R1').eval();x=torch.arange(8)[None]
    replay=torch.zeros(1,8,2,dtype=torch.long)
    with ComputeCounter(model) as count:out=model(x,replay=replay)
    report=summarize(model,out,count)
    assert sum(p['numel'] for p in report['parameter_catalog'].values())==sum(p.numel() for p in model.parameters())
    assert report['execution_trace']['verified']
    assert count.parameter_work['cells.0.up']==16*module_size(model.cells[0].up)
    assert count.parameter_work['cells.0.down']==16*module_size(model.cells[0].down)
    assert report['executed_parameter_work_per_token']>report['unique_touched_weights']['mean']


def test_failed_final_confirmation_evaluation_resumes_without_training(c,tmp_path):
    tokens=np.arange(32,dtype=np.int32);initial={'train':tokens,'train_starts':[0,8]}
    run={'experiment_id':'006-path-first-clm','arm':'R1','seed':606,'role':'confirmation','data_sha256':'a'*64}
    def fail_after_freeze():raise RuntimeError('injected final loader failure')
    with pytest.raises(RuntimeError,match='injected'):_train(c,run,initial,tmp_path,'cpu',final_assets_loader=fail_after_freeze)
    sha=digest(tmp_path/'latest.pt')
    def retry():return {**initial,'confirmation':tokens,'eval_starts':[0,8],'eval_groups':[0,1],'manifest_sha256':'a'*64}
    result=_train(c,run,initial,tmp_path,'cpu',final_assets_loader=retry)
    assert result['steps']==2 and digest(tmp_path/'latest.pt')==sha


def test_gpu_micro_does_not_accept_corpus_or_confirmation():
    import subprocess,sys
    result=subprocess.run([sys.executable,'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/run_experiment_006.py','--mode','gpu-micro-preflight',
        '--assets','not-a-bundle','--role','confirmation'],text=True,capture_output=True)
    assert result.returncode and 'accepts no corpus' in result.stderr


def test_context_noop_is_not_charged_as_an_executed_read(c):
    model=make_model(c['model'],'R1').eval();x=torch.arange(8)[None]
    replay=torch.full((1,8,2),3,dtype=torch.long)
    with ComputeCounter(model) as count:out=model(x,replay=replay,context_mode='zero_replay')
    row=summarize(model,out,count)
    assert row['selected_context_actions']==16 and row['context_read_calls']==0
    assert row['context_attention_gemm_flops']==0


def test_micro_evidence_creates_missing_parent_and_preserves_existing(tmp_path):
    from exp006.gpu_micro import save_evidence
    import json
    import pytest
    output=tmp_path/'fresh'/'nested'/'micro.json'
    save_evidence(str(output),{'status':'fixture'})
    assert json.loads(output.read_text())=={'status':'fixture'}
    assert not output.with_suffix('.tmp').exists()
    with pytest.raises(FileExistsError):
        save_evidence(output,{'status':'overwrite'})
    assert json.loads(output.read_text())=={'status':'fixture'}
