import pytest,torch
from exp006.baseline import make_model
from exp006.training import objective
@pytest.mark.parametrize('arm',['R0','R1','R2','M0'])
def test_real_lm_backward(arm,c):
    model=make_model(c['model'],arm,601);x=torch.arange(16).reshape(2,8);out=model(x,policy='sampled',generator=torch.Generator().manual_seed(3))
    loss,_=objective(out,(x+1)%32,c['training']);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.embedding.weight.grad.abs().sum()>0
    if arm!='M0':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.controller.scorer.parameters()) if arm!='R0' else model.controller.static.grad.abs().sum()>0

def test_only_selected_cell_executes_and_repeats(c):
    model=make_model(c['model'],'R2',1);calls=[0]*4
    hooks=[cell.register_forward_hook(lambda m,i,o,a=a:calls.__setitem__(a,calls[a]+1)) for a,cell in enumerate([*model.cells,model.context])]
    x=torch.ones(2,8,dtype=torch.long);path=torch.zeros(2,8,2,dtype=torch.long)
    out=model(x,replay=path)
    assert calls==[2,0,0,0] and torch.equal(out.actions,path)
    out.logits.sum().backward();assert model.cells[0].up.weight.grad is not None and model.cells[1].up.weight.grad is None
    for h in hooks:h.remove()

def test_shared_weights_named_initialization(c):
    models=[make_model(c['model'],a,21) for a in ['R0','R1','R2']]
    for n,p in models[0].named_parameters():
        if not n.startswith('controller'):assert all(torch.equal(p,dict(m.named_parameters())[n]) for m in models[1:])
    for n,p in models[1].controller.scorer.named_parameters():assert torch.equal(p,dict(models[2].controller.scorer.named_parameters())[n])

def test_no_fixed_transformer_backbone_or_stop(c):
    model=make_model(c['model'],'R2');assert not hasattr(model,'blocks')
    with pytest.raises(ValueError):make_model(c['model'],'R3')
