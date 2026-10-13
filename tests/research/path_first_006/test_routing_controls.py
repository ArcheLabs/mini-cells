import numpy as np,torch
from exp006.baseline import make_model
from exp006.evaluation import intervention_suite

def test_r0_static_r1_predecessor_independent_r2_signal(c):
    h=torch.randn(4,16);i=torch.tensor([0,1,2,3]);j=torch.tensor([3,2,1,0])
    a=make_model(c['model'],'R0').controller.eval();assert torch.equal(a(h,i,0)[0],a(h+3,j,0)[0])
    a=make_model(c['model'],'R1').controller.eval();assert torch.equal(a(h,i,0)[0],a(h,j,0)[0]);assert not torch.equal(a(h,i,0)[0],a(h+3,i,0)[0])
    a=make_model(c['model'],'R2').controller.eval();assert torch.equal(a(h,i,0,'zero')[0],a(h,j,0,'zero')[0]);assert not torch.equal(a(h,i,0)[0],a(h,j,0)[0])

def test_interventions_same_weights_order_multiset(c):
    model=make_model(c['model'],'R2',123).eval();before={n:p.clone() for n,p in model.state_dict().items()};tokens=np.arange(32,dtype=np.int32)
    suite=intervention_suite(model,tokens,[0,8],[0,1],c)
    assert set(suite['interventions'])=={'connection_removed','connection_shuffled','same_multiset_order_permuted'}
    for n,p in model.state_dict().items():assert torch.equal(before[n],p)
    natural=np.asarray(suite['natural']['actions']);permuted=np.asarray(suite['interventions']['same_multiset_order_permuted']['actions'])
    assert np.array_equal(np.sort(natural,-1),np.sort(permuted,-1))
