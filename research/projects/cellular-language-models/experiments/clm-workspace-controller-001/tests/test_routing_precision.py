"""Two small regressions for the observed AMP routing failure; no GPU claims."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import gpu_gate
from models import Model

torch.set_num_threads(1)
C={'dim':8,'heads':2,'ffn':12,'encoder_ffn':12,'layers':3,'vocab':16,'cells':8,'state_dim':6,'window':256,'chunk':4,'steps':2,'top_k':2}


def test_fp32_routing_resolves_half_rounded_boundary_without_disabling_expert_amp():
    model=Model('C2',C);model.trace_tensors=True;model.tensor_trace={'steps':[]}
    with torch.no_grad():
        model.router.weight.zero_();model.state_router.weight.zero_()
        model.router.bias.copy_(torch.tensor([1.,.5,.5001,0.,0.,0.,0.,0.]))
    v=torch.randn(1,2,8);z=torch.randn(1,2,6);expert_dtypes=[]
    hooks=[cell.register_forward_hook(lambda m,a,o:expert_dtypes.append(o.dtype)) for cell in model.cells]
    with torch.autocast('cpu',dtype=torch.float16):
        index,weights,sparse,balance=model.route(v,z,0)
        delta=model.dispatch(v,index,weights)
    for hook in hooks:hook.remove()
    assert model.router.bias.half()[1]==model.router.bias.half()[2]
    assert index.tolist()==[[[0,2],[0,2]]]
    assert model.tensor_trace['steps'][0]['logits'].dtype==weights.dtype==sparse.dtype==torch.float32
    assert expert_dtypes and set(expert_dtypes)=={torch.float16}
    delta.square().sum().backward()
    assert torch.isfinite(model.router.bias.grad).all() and model.router.bias.grad.abs().sum()>0
    model=Model('C2',C)
    with torch.autocast('cpu',dtype=torch.float16):loss=model(torch.tensor([[2,4,5,3]]))[0].float().square().mean()
    loss.backward()
    assert torch.isfinite(model.controller.weight_ih.grad).all() and model.controller.weight_ih.grad.abs().sum()>0


def test_pretraining_fp32_fallback_preserves_failed_evidence_and_requires_new_gates(monkeypatch,tmp_path):
    calls=[];fail_fp32=False
    def select(requested,device):return {'selected_precision':'fp16' if requested=='auto' else requested}
    def check(kind,p):
        selected=select(p['precision']['requested'],'cuda')['selected_precision'];calls.append((kind,selected))
        return {'status':'FAIL' if kind=='correctness' and (selected=='fp16' or fail_fp32) else 'PASS'}
    measured={a:{s:{'status':'PASS','effective_targets_per_second':100.} for s in ('fp16','fp32')} for a in ('C2','T0-F')}
    monkeypatch.setattr(gpu_gate,'select',select)
    monkeypatch.setattr(gpu_gate,'benchmark',lambda c,g,p,o,d:(calls.append(('benchmark','both')) or {'status':'PASS','models':measured}))
    monkeypatch.setattr(gpu_gate,'controller_equivalence',lambda c,p,d:check('controller',p))
    monkeypatch.setattr(gpu_gate,'correctness',lambda c,p,d:check('correctness',p))
    p={'model':C,'precision':{'requested':'auto','consistency_fallback':'fp32'}}
    report=gpu_gate.pilot_engineering_gates({},[],copy.deepcopy(p),tmp_path,0)
    assert report['status']=='PASS' and report['selected_precision']=='fp32'
    assert report['attempts']['fp16']['gpu_correctness']['status']=='FAIL'
    assert report['attempts']['fp32']['status']=='PASS'
    assert calls==[('benchmark','both'),('controller','fp16'),('correctness','fp16'),('controller','fp32'),('correctness','fp32')]
    assert json.loads((tmp_path/'engineering-gates.json').read_text())==report
    calls.clear();fail_fp32=True
    report=gpu_gate.pilot_engineering_gates({},[],copy.deepcopy(p),tmp_path,0)
    assert report['status']=='FAIL' and len(calls)==5
