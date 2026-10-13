import copy,json,random
import numpy as np,pytest,torch
from exp006.baseline import make_model
from exp006.training import objective,_train as train
from exp006.checkpoint import save,restore
from exp006.config import load,authorize
from exp006.assets import load_assets
from exp006.data import encode_stories,assert_disjoint

def test_unregistered_and_confirmation_gate_before_io(tmp_path,c):
    original,p,_=load()
    with pytest.raises(RuntimeError,match='NOT REGISTERED'):load_assets('/does-not-exist',original,p,1,'confirmation')
    with pytest.raises(RuntimeError,match='NOT REGISTERED'):authorize(original,p,1)

def test_exact_interrupted_trajectory_optimizer_rng(c,tmp_path):
    torch.manual_seed(23);random.seed(23);model=make_model(c['model'],'R2',4);opt=torch.optim.AdamW(model.parameters(),lr=.001);data=np.random.default_rng(42);paths=torch.Generator().manual_seed(51);x=torch.arange(16).reshape(2,8)
    def step():
        starts=data.integers(0,100,size=3);out=model(x,policy='sampled',generator=paths);opt.zero_grad();loss,_=objective(out,(x+1)%32,c['training']);loss.backward();opt.step();return starts,out.actions.clone()
    step();run_id={'data':'fixed','source':'fixed'};save(tmp_path/'c.pt',model,opt,1,run_id,data,paths)
    starts,trace=step();expected={n:p.clone() for n,p in model.state_dict().items()};random_value=random.random();torch_value=torch.rand(3);numpy_value=np.random.random(3)
    stepno,_=restore(tmp_path/'c.pt',model,opt,run_id,data,paths);actual_starts,actual_trace=step()
    assert stepno==1 and np.array_equal(starts,actual_starts) and torch.equal(trace,actual_trace)
    assert all(torch.equal(p,expected[n]) for n,p in model.state_dict().items());assert random.random()==random_value and torch.equal(torch.rand(3),torch_value) and np.array_equal(np.random.random(3),numpy_value)
    with pytest.raises(ValueError):restore(tmp_path/'c.pt',model,opt,{'data':'other'},data,paths)

def test_cpu_toy_end_to_end_schema(c,tmp_path):
    # Explicit engineering fixture only, no real corpus or confirmation.
    run={'experiment_id':'006-path-first-clm','arm':'R2','seed':606,'role':'development'}
    tokens=np.arange(32,dtype=np.int32);assets={'train':tokens,'train_starts':[0,8,16],'development':tokens,'eval_starts':[0,8],'eval_groups':[0,1]}
    result=train(c,run,assets,tmp_path,device='cpu')
    assert result['status']=='completed' and result['scientific_decision'] is False
    assert result['confirmation_holdout'] is None and result['decode_profile'] is None
    assert set(result['interventions']['interventions'])=={'connection_removed','connection_shuffled','same_multiset_order_permuted'}
    with pytest.raises(FileExistsError):train(c,run,assets,tmp_path,device='cpu')

def test_duplicate_story_detector():
    with pytest.raises(ValueError):assert_disjoint(['a','b'],['b','c'])
