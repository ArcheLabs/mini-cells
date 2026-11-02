"""Required causal/lifecycle/sparsity checks on tiny fixtures; full X4 accounting once."""
import copy
import pytest
import torch
from torch.nn import functional as F
from common import phase_c,PhaseBLM,native,N1_ALPHA
from architecture import build
from runtime import Session
from flops import report

torch.set_num_threads(1)
TINY={'dim':8,'heads':2,'ffn':16,'vocab':32}


def test_x4_reference_equivalence():
    torch.manual_seed(713)
    original=PhaseBLM(dim=188,heads=4,ffn_dim=639,vocab_size=2048,norm_kind='rmsnorm',activation='swiglu',use_rope=True,update='residual-only').eval()
    model=build('A0').eval();model.load_state_dict(original.state_dict(),strict=True)
    ids=torch.randint(0,2048,(1,40))
    with torch.no_grad():assert torch.allclose(original(ids),model(ids),atol=2e-6,rtol=2e-5)
    assert model.token.weight is model.head.weight
    assert len({id(s.ff.up.weight) for s in model.stages})==3 and N1_ALPHA!=.1


def test_iteration_configs():
    for name,depth in (('A0',[4,4,4]),('A1',[2,2,2]),('A2',[1,1,1]),('A3',[1,2,1])):
        model=build(name,**TINY)
        assert [s.iterations for s in model.stages]==depth
        assert [len(s.step) for s in model.stages]==depth


def test_causality():
    model=build('B2',(1,1,1),**TINY).eval();ids=torch.randint(0,32,(1,70));changed=ids.clone();changed[:,45:]=(changed[:,45:]+1)%32
    with torch.no_grad():assert torch.allclose(model(ids)[:,:45],model(changed)[:,:45],atol=1e-6,rtol=1e-5)


def test_kv_cache_equivalence():
    for name,base in (('A0',None),('A3',None),('D_ATTN1_FFN4',None),('D_ATTN4_FFN1',None),('B0',(1,1,1)),('B2',(1,1,1)),('C2',(1,1,1))):
        model=build(name,base,**TINY).eval();ids=torch.randint(0,32,(1,140))
        with torch.no_grad():reference=model.forward_extended(ids)
        cached=Session(model).prefill(ids)
        assert torch.allclose(reference,cached,atol=3e-6,rtol=3e-5),name


def test_chunk_partition_equivalence():
    model=build('B2',(1,1,1),**TINY).eval();ids=torch.randint(0,32,(1,128))
    states=[];outputs=[]
    for width in (128,32,8,1):
        session=Session(model);outputs.append(torch.cat([session.prefill(chunk) for chunk in ids.split(width,1)],1));states.append(session.cache)
    assert all(torch.equal(outputs[0],x) for x in outputs[1:])
    assert all(torch.equal(states[0]['persistent'],s['persistent']) and s['position']==128 and s['chunk_tokens']==0 for s in states[1:])


def test_state_persistence():
    model=build('B2',(1,1,1),**TINY).eval();session=Session(model);session.prefill(torch.randint(0,32,(1,32)))
    before=session.cache['persistent'].clone();assert bool(before.abs().sum()>0)
    session.prefill(torch.randint(0,32,(1,8)))
    assert torch.equal(before,session.cache['persistent']) and session.cache['chunk_tokens']==8


def test_state_reset():
    model=build('B2',(1,1,1),**TINY).eval();a=Session(model);b=Session(model)
    a.prefill(torch.randint(0,32,(1,35)));assert not bool(b.cache['persistent'].any())
    a.reset();assert not bool(a.cache['persistent'].any()) and a.cache['position']==0 and not a.cache['pending']


def test_snapshot_restore():
    model=build('B2',(1,1,1),**TINY).eval();ids=torch.randint(0,32,(1,72));session=Session(model,'fixture')
    session.prefill(ids[:,:39]);snapshot=session.snapshot();continuous=session.prefill(ids[:,39:])
    restored=Session(model,'fixture').restore(snapshot);resumed=restored.prefill(ids[:,39:])
    assert torch.equal(continuous,resumed) and torch.equal(restored.cache['persistent'],session.cache['persistent'])
    assert torch.equal(restored.generator.get_state(),session.generator.get_state())
    with pytest.raises(ValueError):Session(model,'different').restore(snapshot)


def test_state_read_ablation():
    model=build('B2',(1,1,1),**TINY).eval();ids=torch.randint(0,32,(1,64))
    with torch.no_grad():a=model(ids);model.read_enabled=False;b=model(ids)
    assert not torch.equal(a[:,32:],b[:,32:])


def test_state_write_ablation():
    model=build('B2',(1,1,1),**TINY).eval();model.write_enabled=False;session=Session(model)
    session.prefill(torch.randint(0,32,(1,64)));assert not bool(session.cache['persistent'].any())


def test_flat_vs_cellular_state():
    for name,shape in (('B1',(2,2048)),('B2',(2,16,128))):
        model=build(name,(1,1,1),**TINY);ids=torch.randint(0,32,(2,64))
        logits,state=model.forward_extended(ids,return_session=True);assert state['persistent'].shape==shape
        F.cross_entropy(logits.reshape(-1,32),ids.reshape(-1)).backward()
        assert model.persistent.write_value.weight.grad is not None


def test_sparse_cell_execution():
    model=build('C2',(1,1,1),**TINY);ids=torch.randint(0,32,(2,32));rows=[]
    hook=model.stages[1].ff.shared.register_forward_pre_hook(lambda module,args:rows.append(args[0].shape[0]))
    logits=model(ids);logits.sum().backward();hook.remove()
    assert rows==[2*32*2] and model.stages[1].ff.executed_rows==128
    assert model.stages[1].ff.down.grad is not None
    assert not hasattr(model.stages[1].ff,'dense_ffn')


def test_parameter_accounting():
    a=build('A0');assert sum(p.numel() for p in a.parameters())==1893912
    counts={n:sum(p.numel() for p in build(n,(1,1,1)).parameters()) for n in ('B0','B1','B2','C1','C2','C3')}
    assert abs(counts['B1']-counts['B2'])<5000
    assert abs(counts['C2']/counts['C3']-1)<.001


def test_flops_accounting():
    a=build('A0');assert report(a)['forward_flops']==phase_c.estimate_forward_flops(phase_c.X4_NAME,125)
    assert report(build('A2'))['forward_flops']<.8*report(a)['forward_flops']
    c=report(build('C2',(1,1,1)))
    assert all(c['parts'][k]>0 for k in ('state_read','state_write','router','cell_ffn'))


def test_training_gradients(tmp_path,monkeypatch):
    model=build('B2',(1,1,1),**TINY);ids=torch.randint(0,32,(1,64))
    logits=model(ids);loss=F.cross_entropy(logits[:,40:].reshape(-1,32),ids[:,40:].reshape(-1));loss.backward()
    grad=model.persistent.write_value.weight.grad
    assert grad is not None and bool(grad.isfinite().all()) and bool(grad.abs().sum()>0)

    # Exercise the real trainer once on a synthetic CPU corpus (not real data or a benchmark).
    import json,numpy as np,time
    from train import train_language
    from common import protocol
    train_file=tmp_path/'train.u16';validation_file=tmp_path/'validation.u16'
    values=np.arange(2000,dtype=np.uint16)%32;values.tofile(train_file);values.tofile(validation_file)
    corpus=native.Corpus(train_file,validation_file,tmp_path/'unused-tokenizer',{})
    job=protocol();job['training']['profiles']['smoke']={'tokens':1000,'warmup':1}
    job.update(profile='smoke',match='tokens');metadata={'model':'CPU_FIXTURE','training_status':'IN_PROGRESS'}
    folder=tmp_path/'run';folder.mkdir()
    train_language(model,corpus,((0,),),job,torch.device('cpu'),folder,time.monotonic()+60,metadata)
    assert metadata['training_status']=='COMPLETE' and metadata['training_tokens']==1000 and metadata['optimizer_updates']==1
    assert json.loads((folder/'flops-report.json').read_text())['runtime_profile']['status']=='MEASURED_TRAINING_UPDATE'
    from mechanisms import episodes
    train_tasks,used=episodes(model,713,2);test_tasks,_=episodes(model,714,2,used)
    assert not ({r['combination'] for r in train_tasks}&{r['combination'] for r in test_tasks})
    assert all(r['early_fact_gap']>r['receptive_field_bound'] for r in test_tasks)

    # Auxiliary profiler failure cannot erase a committed optimizer update.
    import train as trainer
    def broken_profile(*args):raise RuntimeError('intentional profiler export failure')
    monkeypatch.setattr(trainer,'profile_record',broken_profile)
    other=tmp_path/'profile-failure';other.mkdir();meta={'model':'CPU_FIXTURE','training_status':'IN_PROGRESS'}
    trainer.train_language(model,corpus,((0,),),job,torch.device('cpu'),other,time.monotonic()+60,meta)
    assert meta['training_status']=='COMPLETE' and meta['training_tokens']==1000 and meta['optimizer_updates']==1
    assert json.loads((other/'flops-report.json').read_text())['runtime_profile']['status']=='FAILED'
