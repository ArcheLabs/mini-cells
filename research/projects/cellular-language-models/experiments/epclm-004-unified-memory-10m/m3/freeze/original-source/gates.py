"""Six ordered engineering gates; reports contain measured numerical evidence."""
import copy
import gc
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from memory import NeuralMemory, memory_forward, state_bytes
from mag_model import MAGLite, Config, E1
from e0 import save_checkpoint, load_checkpoint, masked_loss


def finite(parameters):
    values=list(parameters)
    assert values and all(p is not None and torch.isfinite(p).all() for p in values)


def gate1(out):
    torch.manual_seed(41031)
    m=NeuralMemory(4,4,6).double()
    x=torch.randn(1,6,4,dtype=torch.float64)
    target=torch.randn(1,6,4,dtype=torch.float64)
    def objective():
        y,_,_=m.stream(x,chunk=2)
        return (y-target).square().mean()
    loss=objective()
    named=dict(m.named_parameters())
    grads=torch.autograd.grad(loss,tuple(named.values()))
    finite(grads)
    errors=[]
    # Initialization and all Q/K/V/control projections must receive exact meta-gradients.
    for name in ('initial.w1','initial.ln_weight','key.weight','value.weight','query.weight','controls.weight'):
        p=named[name];g=grads[list(named).index(name)]
        index=int(g.detach().abs().argmax());analytic=float(g.reshape(-1)[index])
        original=float(p.detach().reshape(-1)[index]);eps=1e-5
        with torch.no_grad():p.reshape(-1)[index]=original+eps
        plus=float(objective().detach())
        with torch.no_grad():p.reshape(-1)[index]=original-eps
        minus=float(objective().detach())
        with torch.no_grad():p.reshape(-1)[index]=original
        numerical=(plus-minus)/(2*eps);error=abs(numerical-analytic)
        tolerance=2e-6+2e-4*abs(numerical)
        assert error<=tolerance,(name,error,tolerance)
        assert abs(analytic)>1e-10,name
        errors.append(dict(parameter=name,index=index,autodiff=analytic,finite_difference=numerical,
                           absolute_error=error,tolerance=tolerance))
    # Independent explicit recurrence for three chunk updates, including nonzero past surprise.
    cell=m.create();oracle={k:v.clone() for k,v in cell.fast.items()};moment={k:v.clone() for k,v in cell.momentum.items()}
    max_error=0.;limits=[]
    for hidden in x.split(2,1):
        keys,values,_=m.project(hidden)
        objective_inner=((memory_forward(keys,oracle)-values)**2).sum(-1).mean(1).sum()
        gradient=torch.autograd.grad(objective_inner,tuple(oracle.values()))
        eta,theta,alpha=m.controls(keys.mean(1)).sigmoid().unbind(-1)
        theta=theta*m.max_lr;alpha=alpha*m.max_forget
        new_moment={};new_oracle={}
        for (name,p),g in zip(oracle.items(),gradient):
            shape=(-1,)+(1,)*(p.ndim-1)
            new_moment[name]=eta.view(shape)*moment[name]-theta.view(shape)*g
            new_oracle[name]=(1-alpha.view(shape))*p+new_moment[name]
        cell,d=m.learn(keys,values,cell)
        for name in oracle:
            max_error=max(max_error,float((cell.fast[name]-new_oracle[name]).detach().abs().max()),
                          float((cell.momentum[name]-new_moment[name]).detach().abs().max()))
        oracle,moment=new_oracle,new_moment
        limits.append([float(d['eta']),float(d['theta']),float(d['alpha'])])
    assert max_error<1e-12
    assert all(0<=eta<=1 and 0<=theta<=m.max_lr and 0<=alpha<=m.max_forget for eta,theta,alpha in limits)
    return {'status':'PASS','finite_difference':errors,'recurrence_max_abs_error':max_error,
            'eta_theta_alpha':limits,'precision':'float64','gradient':'complete higher-order, three chunks'}


def gate2(out):
    torch.manual_seed(41032);m=NeuralMemory(4,4,8).double();x=torch.randn(2,9,4,dtype=torch.float64)
    y,cell,_=m.stream(x,chunk=3)
    modified=x.clone();modified[:,4:]+=torch.randn_like(modified[:,4:])*2
    z,_,_=m.stream(modified,chunk=3)
    future_error=float((y[:,:4]-z[:,:4]).detach().abs().max());assert future_error==0
    initial=m.create(2);k,v,q=m.project(x[:,:3]);before=m.recall(q,initial,'meta_train')
    after,_=m.learn(k,v,initial);written=m.recall(q,after,'meta_train')
    error=float((y[:,:3]-before).detach().abs().max());assert error==0
    assert float((before-written).detach().abs().max())>0
    # Different streams never share fast state or chunk gradients.
    independent=torch.cat([m.stream(row[None],chunk=3)[0] for row in x],0)
    batch_error=float((y-independent).detach().abs().max());assert batch_error<1e-12
    return {'status':'PASS','future_error':future_error,'read_before_write_error':error,
            'batch_isolation_error':batch_error,'chunk':3,'order':'all reads use preceding chunk state; then one chunk update'}


def gate3(out):
    torch.manual_seed(41033);m=NeuralMemory(4,4,8);x=torch.randn(1,4,4)
    cell=m.create(mode='online_adapt');snapshot=cell.detached();static=copy.deepcopy(m.state_dict())
    k,v,q=m.project(x);same,_=m.learn(k,v,cell,'read_only');m.recall(q,cell)
    assert same is cell
    assert all(torch.equal(cell.fast[n],snapshot.fast[n]) and torch.equal(cell.momentum[n],snapshot.momentum[n]) for n in cell.fast)
    learned,_=m.learn(k,v,cell,'online_adapt')
    assert all(torch.equal(p,static[n]) for n,p in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())
    compatibility={'static':'tiny-fixture-v1','tokenizer':'latent-v1','interface':1}
    stats=m.checkpoint(learned,Path(out)/'cell',compatibility);restored=m.restore(Path(out)/'cell',compatibility)
    err=float((m.recall(q,learned)-m.recall(q,restored)).abs().max());assert err==0
    for name in learned.fast:
        assert torch.equal(learned.fast[name],restored.fast[name])
        assert torch.equal(learned.momentum[name],restored.momentum[name])
    # Corruption of either independently serialized component must fail closed.
    manifest=__import__('json').loads((Path(out)/'cell/manifest.json').read_text())
    component=Path(out)/'cell'/manifest['generation']/'momentum.pt'
    original=component.read_bytes();component.write_bytes(original+b'corruption')
    try:
        m.restore(Path(out)/'cell',compatibility)
    except ValueError:pass
    else:raise AssertionError('corrupted momentum was accepted')
    component.write_bytes(original)
    fork=m.fork(learned)
    assert all(fork.fast[n].data_ptr()!=learned.fast[n].data_ptr() and fork.momentum[n].data_ptr()!=learned.momentum[n].data_ptr() for n in learned.fast)
    with torch.no_grad():fork.fast['w1'].add_(1)
    assert torch.equal(learned.fast['w1'],restored.fast['w1'])
    try:m.recall(q.expand(2,-1,-1),learned)
    except ValueError:pass
    else:raise AssertionError('Cell batch mismatch was silently broadcast')
    try:m.learn(k.expand(2,-1,-1),v.expand(2,-1,-1),learned,'online_adapt')
    except ValueError:pass
    else:raise AssertionError('write batch mismatch was silently broadcast')
    return {'status':'PASS','restore_max_abs_error':err,'static_unchanged':True,'fork_tensor_alias':False,**stats}


def tiny_model():
    torch.manual_seed(41017)
    return MAGLite(Config(vocab=64,dim=16,layers=1,heads=2,kv_heads=2,ffn=32,window=4),4,8,2)


def gate4(out):
    m=tiny_model();ids=torch.randint(0,64,(2,8));labels=torch.randint(0,64,(2,8))
    reference=E1(m.config);reference.load_state_dict({n:p for n,p in m.state_dict().items() if n in reference.state_dict()})
    only_swa,_,g=m.forward_state(ids,disable='memory');assert torch.equal(only_swa,reference(ids));assert not g.any()
    only_memory,_,g=m.forward_state(ids,disable='swa');assert g.eq(1).all() and torch.isfinite(only_memory).all()
    logits,cell,g=m.forward_state(ids);assert g.ge(0).all() and g.le(1).all()
    batch_error=float((m(ids[:1])-logits[:1]).detach().abs().max());assert batch_error<1e-6
    altered=ids.clone();altered[:,5:]=(altered[:,5:]+1)%64
    causal_error=float((logits[:,:5]-m(altered)[:,:5]).detach().abs().max());assert causal_error<1e-6
    loss=masked_loss(logits,labels);loss.backward();finite(p.grad for p in m.parameters())
    for prefix in ('memory.initial','memory.key','memory.value','memory.query'):
        assert sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(prefix))>0,prefix
    # Exercise the exact inherited transaction/checkpoint trainer, not a second toy loop.
    from training import fit
    rows=[]
    for i in range(6):
        tokens=[(j*3+i)%64 for j in range(8)]
        rows.append({'id':f'tiny-{i}','input_ids':tokens,'labels':[-100]*7+[(i+5)%64],
                     'query_mask':[False]*7+[True]})
    config={'seed':41017,'batch':2,'lr':.001,'updates':2,'precision':'fp32',
            'tf32':False,'optimizer':'AdamW','scheduler':'constant-engineering-smoke',
            'checkpoint':{'every_updates':8,'every_seconds':120}}
    before=m.token_embedding.weight.detach().clone();identity={'fixture':'M3-tiny-resume-v1'}
    first=fit(m,rows[:4],rows[4:],config,Path(out)/'training',identity,stop_after=1)
    assert first['ledger']['committed_updates']==1
    resumed=tiny_model()
    final=fit(resumed,rows[:4],rows[4:],config,Path(out)/'training',identity)
    assert final['status']=='COMPLETE' and final['ledger']['committed_updates']==2
    assert final['history'][0]==first['history'][0] and len(resumed.events)==1
    saved=load_checkpoint(Path(out)/'training/checkpoint.pt')
    assert all(k in saved for k in ('model','optimizer','rng','sampler','ledger'))
    assert saved['ledger']==final['ledger'] and saved['sampler']['cursor']==4
    assert all(int(v['step'])==2 for v in saved['optimizer']['state'].values())
    fresh=tiny_model();fresh.load_state_dict(saved['model']);fresh.eval();resumed.eval()
    with torch.no_grad():
        restore_error=float((fresh(ids)-resumed(ids)).abs().max())
    assert restore_error==0
    delta=float((resumed.token_embedding.weight.detach()-before).norm());assert delta>0
    persistent=cell.detached();saved=persistent.fast['w1'].clone();m.reset_runtime();assert torch.equal(saved,persistent.fast['w1'])
    online=tiny_model().set_mode('online_adapt');static=copy.deepcopy(online.state_dict());_,state,_=online.forward_state(ids)
    assert all(torch.equal(p,static[n]) for n,p in online.state_dict().items()) and state.update_step==4
    readonly=online.set_mode('read_only');_,s,_=readonly.forward_state(ids,state)
    assert s is state and all(not p.requires_grad for p in readonly.parameters())
    return {'status':'PASS','loss':float(loss.detach()),'embedding_delta':delta,'causal_error':causal_error,'batch_isolation_error':batch_error,
            'gate_range':[float(g.detach().min()),float(g.detach().max())],
            'static_parameters':sum(p.numel() for p in m.parameters()),'dynamic':state_bytes(cell),
            'outer_updates':2,'probe_inner_updates':cell.update_step,
            'training_inner_updates':m.events[-1]['inner_updates']+resumed.events[-1]['inner_updates'],
            'runtime_reset_preserves_cell':True,
            'checkpoint_restore_error':restore_error,'resumed_updates_only':1,'ledger':final['ledger']}


def recall_control(out,device='cpu',memory=None,steps=128,context=None):
    """Latent associative positive control, NOT a language/MQAR quality claim."""
    torch.manual_seed(41035)
    m=memory if memory is not None else NeuralMemory(8,8,16)
    m=m.to(device);m.requires_grad_(False);dim=m.dim
    keys=torch.eye(dim,device=device)[:4][None]
    permutation=torch.tensor([2,0,3,1],device=device)
    values=F.layer_norm(keys[:,permutation],(dim,))
    cell=m.create(mode='online_adapt');initial=cell.detached();static=copy.deepcopy(m.state_dict())
    def score(c):
        read=m.recall(keys,c)
        predictions=F.cosine_similarity(read[:,:,None],values[:,None],dim=-1).argmax(-1)
        return float(predictions.eq(torch.arange(4,device=device)).float().mean())
    initial_accuracy=score(cell);assert initial_accuracy<1
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    sync();started=time.perf_counter()
    for _ in range(steps):cell,_=m.learn(keys,values,cell,'online_adapt')
    sync();update_seconds=time.perf_counter()-started
    accuracy=score(cell);assert accuracy==1,('recall positive control',initial_accuracy,accuracy)
    expected=m.recall(keys,cell);compatibility={'static_state': __import__('data').digest({n:__import__('hashlib').sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in static.items()}),'interface':1,'task':'latent-association-control-v1','context':context or {'scope':'latent-no-tokenizer'}}
    stats=m.checkpoint(cell,Path(out)/'recall-cell',compatibility)
    fast_delta=sum(float((cell.fast[n].detach()-initial.fast[n]).square().sum()) for n in initial.fast)**.5
    momentum=sum(float(v.square().sum()) for v in cell.momentum.values())**.5
    del cell,initial;gc.collect()
    # New session contains only fixed static parameters plus the restored persistent Cell.
    fresh=NeuralMemory(m.input_dim,m.dim,m.hidden,m.max_lr,m.max_forget).to(device)
    fresh.load_state_dict(static);fresh.requires_grad_(False)
    restored=fresh.restore(Path(out)/'recall-cell',compatibility)
    sync();started=time.perf_counter();actual=fresh.recall(keys,restored);sync();recall_seconds=time.perf_counter()-started
    error=float((actual-expected).abs().max());assert error==0
    assert all(torch.equal(p,static[n]) for n,p in m.state_dict().items())
    predictions=F.cosine_similarity(actual[:,:,None],values[:,None],dim=-1).argmax(-1)
    assert predictions.eq(torch.arange(4,device=device)).all()
    assert fast_delta>0 and momentum>0
    return {'status':'PASS','task':'four latent keys -> nonidentity permuted values; no query answers supplied',
            'scope':'Memory positive control; not language MQAR or H3 evidence','initial_accuracy':initial_accuracy,
            'restored_accuracy':accuracy,'restore_max_abs_error':error,'inner_updates':steps,
            'fast_delta':fast_delta,'momentum_norm':momentum,'static_unchanged':True,
            'memory_update_seconds':update_seconds,'recall_seconds':recall_seconds,**state_bytes(restored),**stats}


def gate5(out):
    torch.manual_seed(41035)
    return recall_control(out,memory=NeuralMemory(256,32,64))
