"""One small CPU integration test covers the essential state contracts."""
import copy
import torch
from model import Core, Config
from runtime import Runtime

def test_contracts():
    torch.set_num_threads(1); torch.manual_seed(11)
    core=Core(Config(vocab=32,cells=4,dim=8)); runtime=Runtime(core)
    initial=core.init_state(2); tokens=torch.tensor([[1,2],[3,4],[5,6],[7,8]])
    state=initial
    outputs=[]
    for token in tokens:
        state,logits=runtime.step(state,token); outputs.append(logits.clone())
    reference=state.clone(); before=state.clone(); core.read(state)
    assert torch.equal(before,state)
    runtime2=Runtime(core); state=initial
    for token in tokens[:2]: state,_=runtime2.step(state,token)
    snapshot=runtime2.snapshot(state); state=runtime2.restore(snapshot)
    for token in tokens[2:]: state,_=runtime2.step(state,token)
    assert torch.equal(reference,state)
    solo=core.init_state(1)
    for token in tokens[:,0]: solo,_=core.step(solo,token[None])
    torch.testing.assert_close(solo,state[:1],atol=1e-6,rtol=1e-6)
    assert torch.count_nonzero(core.reset(state))==0
    bad=copy.deepcopy(snapshot); bad['state'][0,0,0]+=1
    try: runtime.restore(bad)
    except ValueError: pass
    else: raise AssertionError('corrupt snapshot accepted')
    state=core.init_state(1)
    for token in [1,3]: state,logits=core.step(state,[token])
    logits.sum().backward()
    assert core.embedding.weight.grad[1].abs().sum()>0
    assert core.left.weight.grad.abs().sum()>0
    assert core.embedding.weight.grad[7].abs().sum()==0
    try: core.evolve(torch.full_like(state,float('inf')))
    except FloatingPointError: pass
    else: raise AssertionError('non-finite state accepted')
    import tempfile,time
    from pathlib import Path
    from trainer import train
    documents=[{'id':'a','tokens':[1,2,3,4,5,6,7,8,9]},{'id':'b','tokens':[9,8,7,6,5,4,3,2,1]}]
    with tempfile.TemporaryDirectory() as directory:
        one,two=Core(Config(vocab=32,cells=4,dim=8)),Core(Config(vocab=32,cells=4,dim=8))
        two.load_state_dict(one.state_dict()); identity={'fixture':'resume'}
        kwargs=dict(documents=documents,identity=identity,target_tokens=16,deadline=time.monotonic()+60,seed=12,batch=2,unroll=2)
        train(one,path=Path(directory)/'one.pt',**kwargs)
        train(two,path=Path(directory)/'two.pt',stop_after_updates=1,**kwargs)
        train(two,path=Path(directory)/'two.pt',**kwargs)
        for a,b in zip(one.parameters(),two.parameters()): torch.testing.assert_close(a,b,atol=0,rtol=0)
    for arm in ('C0','C1','F0','L0'):
        model=Core(Config(vocab=32,cells=4,dim=8,arm=arm))
        state=model.init_state(1)
        for token in [1,2,3,4,5,6]: state,logits=model.step(state,[token])
        logits.square().mean().backward()
        assert torch.isfinite(logits).all()
        if arm=='L0':
            other=model.init_state(1)
            for token in [9,9,3,4,5,6]: other,_=model.step(other,[token])
            assert torch.equal(state,other)
    # A single update cannot propagate beyond radius one.
    model=Core(Config(vocab=32,cells=8,dim=8))
    zero=model.init_state(1); changed=model.inject(zero,[1])
    delta=model.evolve(changed,1)-model.evolve(zero,1)
    assert torch.count_nonzero(delta[:,2:7])==0
