import itertools,torch
from exp006.controller import Controller
from exp006.baseline import make_model
from exp006.training import objective

def test_score_function_matches_exact_two_step_expectation(c):
    cfg={**c['model'],'cell_count':1};actor=Controller(cfg,'R2').double();h=torch.linspace(-.7,.7,16)[None].double();expected=0.;estimate=0.
    for a,b in itertools.product(range(2),repeat=2):
        s0,_=actor(h,torch.tensor([2]),0);log0=s0.log_softmax(-1)[0,a];state=h+(a+1)*.2
        s1,_=actor(state,torch.tensor([a]),1);log1=s1.log_softmax(-1)[0,b];prob=(log0+log1).exp();cost=torch.tensor(2*a+b+(a-b)**2,dtype=torch.double)
        expected=expected+prob*cost
        # Independent state baselines have zero expected score, including at the second state.
        estimate=estimate+prob.detach()*((cost-.4)*log0+(cost-(a+1)*.1)*log1)
    params=list(actor.scorer.parameters())+[actor.source.weight,actor.target.weight]
    exact=torch.autograd.grad(expected,params,retain_graph=True);pg=torch.autograd.grad(estimate,params)
    for a,b in zip(exact,pg):assert torch.allclose(a,b,atol=2e-7,rtol=2e-5)
    assert sum(float(a.abs().sum()) for a in pg)>0

def test_actor_advantage_and_critic_do_not_leak_labels(c):
    model=make_model(c['model'],'R2',18);x=torch.arange(16).reshape(2,8);out=model(x,policy='sampled');labels=(x+3)%32
    advantage=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),labels.flatten(),reduction='none').reshape(labels.shape).detach()[...,None]-out.values.detach()
    actor=(advantage*out.log_prob).sum(-1).mean()
    grads=torch.autograd.grad(actor,list(model.controller.critic.parameters()),allow_unused=True,retain_graph=True)
    assert all(g is None for g in grads)
    critic=(out.values-torch.ones_like(out.values)).square().mean()
    assert torch.autograd.grad(critic,model.embedding.weight,allow_unused=True)[0] is None
