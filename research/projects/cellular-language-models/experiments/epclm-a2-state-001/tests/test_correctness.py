"""Sixteen required checks, tiny CPU fixtures, no independent GPU smoke or benchmarks."""
import copy
import torch
from torch.nn import functional as F
from bootstrap import shared,old_architecture,old_runtime
from a2state.architecture import InteractionModel,initial_backbone,ARMS
from a2state.runtime import Session,empty,parallel_forward,stages
from a2state.flops import report
from a2state.state_read import attention

torch.set_num_threads(1)
TINY={'dim':8,'heads':2,'ffn':16,'vocab':32}


def build(arm='B1'):return InteractionModel(arm,**TINY)
def ids(n=70,b=1):return torch.randint(0,32,(b,n))


def test_a2_reference_equivalence():
    base=initial_backbone();model=InteractionModel('B1');model.read_enabled=False
    assert model.base_initial_sha256==shared.model_hash(base)
    sample=torch.randint(0,2048,(1,125))
    with torch.no_grad():actual,cache=model.forward_extended(sample,return_session=True);expected=base(sample)
    assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-5)
    assert cache['layers']['stage:2/iteration:0']['k'].shape[2]==125
    assert cache['state_revision']==4 and cache['closed']
    before=torch.get_rng_state();other=InteractionModel('B2');assert torch.equal(before,torch.get_rng_state())
    assert other.base_initial_sha256==model.base_initial_sha256 and other.state_initial_sha256==model.state_initial_sha256


def test_attention_state_memory():
    model=build().eval();model.record_diagnostics=True;sample=ids(32)
    with torch.no_grad():a=model(sample);model.read_enabled=False;b=model(sample)
    assert not torch.equal(a,b)
    model.read_enabled=True;model(sample)
    assert {r['path'] for r in model.observations}=={'attention'}
    assert all(float(r['memory_mass'])>0 and r['per_cell_mass'].numel()==16 for r in model.observations)


def test_ffn_state_conditioning():
    model=build('B2').eval();model.record_diagnostics=True;x=torch.randn(1,5,8)
    first=empty(model,1);second=empty(model,1);second['persistent']+=1
    with torch.no_grad():
        a=attention(model,model.stages[0],x,first,0,0);b=attention(model,model.stages[0],x,second,0,0)
        assert torch.equal(a,b)
        model(ids())
    assert {r['path'] for r in model.observations}=={'ffn'}
    assert all(not str(k).startswith('state:') for k in first['layers'])


def test_combined_state_interaction():
    model=build('B3').eval();model.record_diagnostics=True
    with torch.no_grad():model(ids())
    assert {r['path'] for r in model.observations}=={'attention','ffn'}
    for revision in {r['state_revision'] for r in model.observations}:
        records=[r for r in model.observations if r['state_revision']==revision]
        assert len({r['state_pointer'] for r in records})==1
    cache=empty(model,1);outs=[];ff=[]
    hooks=[s.attn.out.register_forward_hook(lambda m,a,o:outs.append(o)) for s in model.stages]
    hooks += [s.ff.register_forward_hook(lambda m,a,o:ff.append(o)) for s in model.stages]
    with torch.no_grad():_,features=stages(model,model.token(ids(4)),cache,0)
    for hook in hooks:hook.remove()
    assert torch.allclose(features,torch.stack([(outs[2*i]+ff[i])/2 for i in range(3)]).mean(0))


def test_state_reset():
    model=build().eval();session=Session(model);initial=model.persistent.init(1)
    session.prefill(ids(40));session.reset();assert torch.equal(session.cache['persistent'],initial)
    assert session.cache['position']==0 and not session.cache['pending']


def test_state_persistence():
    model=build().eval();session=Session(model);session.prefill(ids(32));state=session.cache['persistent'].clone()
    assert not torch.equal(state,model.persistent.init(1))
    session.prefill(ids(8));assert torch.equal(state,session.cache['persistent'])
    sizes=[];original=model.persistent.write
    def capture(state,h,mask=None):sizes.append(h.shape[1]);return original(state,h,mask)
    model.persistent.write=capture;session.reset();session.prefill(ids(125));session.finish()
    assert sizes==[32,32,32,29]


def test_batch_isolation():
    model=build('B3').eval();sample=ids(70,2);other=sample.clone();other[1]=(other[1]+1)%32
    with torch.no_grad():a=model(sample);b=model(other)
    assert torch.equal(a[0],b[0])


def test_no_future_state_leak():
    for arm in ARMS:
        model=build(arm).eval();sample=ids();changed=sample.clone();changed[:,45:]=(changed[:,45:]+1)%32
        with torch.no_grad():a=model(sample);b=model(changed)
        assert torch.allclose(a[:,:45],b[:,:45],atol=1e-6,rtol=1e-5)


def test_kv_cache_equivalence():
    for arm in ARMS:
        model=build(arm).eval();sample=ids(125)
        with torch.no_grad():reference=model(sample)
        session=Session(model);actual=session.prefill(sample);session.finish()
        assert torch.allclose(reference,actual,atol=3e-6,rtol=3e-5),arm
        split=Session(model);partitioned=torch.cat([split.prefill(chunk) for chunk in sample.split(8,1)],1);split.finish()
        assert torch.equal(actual,partitioned) and torch.equal(session.cache['persistent'],split.cache['persistent'])


def test_state_read_ablation():
    for arm in ARMS:
        model=build(arm).eval();model.read_enabled=False;sample=ids(64)
        base=initial_backbone(**TINY)
        with torch.no_grad():assert torch.allclose(model(sample),base(sample),atol=2e-6,rtol=2e-5)


def test_state_write_ablation():
    model=build('B2').eval();model.write_enabled=False;session=Session(model)
    session.prefill(ids(70));session.finish();assert torch.equal(session.cache['persistent'],model.persistent.init(1))


def test_reset_every_chunk():
    model=build('B3').eval();model.reset_every_chunk=True;session=Session(model)
    session.prefill(ids(70));session.finish();assert torch.equal(session.cache['persistent'],model.persistent.init(1))


def test_gradient_across_chunks(tmp_path):
    for arm in ARMS:
        model=build(arm);model.collect_gradients=True;sample=ids(64)
        logits=model(sample);F.cross_entropy(logits[:,40:].reshape(-1,32),sample[:,40:].reshape(-1)).backward()
        assert model.persistent.update.weight_ih.grad is not None and bool(model.persistent.update.weight_ih.grad.abs().sum()>0)
        assert model.persistent.bridge.weight.grad is not None
        assert model.persistent.init(1).dtype==torch.float32
        if arm=='B1':assert not model.path_tensors['ffn']
        if arm=='B2':assert not model.path_tensors['attention']


    # One synthetic optimizer update through the actual trainer; no real corpus training.
    import json,time
    import numpy as np
    from types import SimpleNamespace
    from a2state.training import run
    from a2state.audit import load
    stream=tmp_path/'tokens.bin';np.arange(512,dtype=np.uint16).__mod__(32).tofile(stream)
    plan=copy.deepcopy(load()['training']);plan['profiles']['dev']={'tokens':1000,'warmup':1}
    entry={'synthetic_fixture':True};model=build('B3')
    run(model,SimpleNamespace(train_path=stream,validation_path=stream),((0,),),plan,torch.device('cpu'),tmp_path,time.monotonic()+30,entry)
    assert entry['training_status']=='COMPLETE' and entry['optimizer_updates']==1 and entry['training_tokens']==1000
    row=json.loads((tmp_path/'training-metrics.json').read_text())['metrics'][0]
    assert row['state_update_gradient_norm']>0 and row['state_read_projection_gradient_norm']>0
    saved=tmp_path/'final-model.pt';shared.native._atomic_save({'model':model.state_dict()},saved)
    restored=build('B3');restored.load_state_dict(torch.load(saved,weights_only=True)['model'])
    assert shared.model_hash(restored)==shared.model_hash(model)


def test_snapshot_restore():
    model=build('B3').eval();sample=ids(100);session=Session(model,'fixture')
    session.prefill(sample[:,:39]);torch.rand(2,generator=session.generator);saved=session.snapshot()
    a=session.prefill(sample[:,39:]);random_a=torch.rand(2,generator=session.generator)
    restored=Session(model,'fixture').restore(saved);b=restored.prefill(sample[:,39:]);random_b=torch.rand(2,generator=restored.generator)
    assert torch.equal(a,b) and torch.equal(random_a,random_b)
    session.finish();closed=session.snapshot();restored.restore(closed);assert restored.cache['closed']


def test_parameter_accounting():
    counts=[]
    for arm in ARMS:
        model=InteractionModel(arm);counts.append(sum(p.numel() for p in model.parameters()))
        assert not any('router' in name for name,_ in model.named_parameters())
    assert len(set(counts))==1 and counts[0]>1892220


def test_flops_accounting():
    rows={arm:report(InteractionModel(arm)) for arm in ARMS}
    assert rows['B1']['parts']['state_read_ffn_path']==0 and rows['B2']['parts']['state_read_attention_path']==0
    assert rows['B3']['forward_flops']>max(rows['B1']['forward_flops'],rows['B2']['forward_flops'])
    assert len({row['parts']['state_write'] for row in rows.values()})==1
