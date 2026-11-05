"""Small CPU functional/gradient proof, no optimizer or language training."""
import torch
from repair_models import ARMS,build,select_rank
from runtime import Runtime
from frozen_model import BaselineIntegrityError


def preflight(plan,full=True):
    count=torch.get_num_threads();torch.set_num_threads(1)
    try:
        expected={'tokens':250000,'seed':173018021,'batch':4,'tbptt':64,'precision':'fp32','lr':3e-4,'betas':[.9,.95],'weight_decay':.1,'clip':1.,'scheduler':'existing warmup/cosine'}
        if any(plan['training'].get(k)!=v for k,v in expected.items()):raise BaselineIntegrityError('protocol differs from supported unchanged Trainer')
        if plan['event']['coordinate_shift']!=8 or plan['event']['cells']!=16 or plan['event']['dim']!=128:raise BaselineIntegrityError('frozen event encoding differs')
        sizing=select_rank()
        if (sizing['rank']!=plan['parameter_matching']['rank'] or not sizing['within_tolerance']
                or plan['parameter_matching']['target_active_parameters']!=966784
                or plan['parameter_matching']['expected_active_parameters']!=sizing['parameters']):
            raise BaselineIntegrityError('frozen active-parameter sizing differs: '+str(sizing))
        if not full:return {'status':'CONFIGURATION_PASS','rank_selection':sizing,'gradient_proof':'MATCHING_BATCH_A_REQUIRED_NOT_REPEATED','optimizer_updates':0}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(713)
            independent=build('IC-ALL-SELF');broadcast=build('C1-BROADCAST');broadcast.load_state_dict(independent.state_dict())
            state=independent.init_state(1);a=independent.inject(state,[7]);b=broadcast.inject(state,[7])
            if any(not torch.equal(a[0,i],independent.embedding.weight[7].roll(8*i)) for i in range(16)):raise BaselineIntegrityError('event permutation does not match frozen roll rule')
            if not torch.equal(a,b) or torch.unique(a[0],dim=0).shape[0]!=16:raise BaselineIntegrityError('event distribution differs/degenerates on generic fixture')
            changed=a.clone();changed[:,4]+=1
            delta=(independent.evolve(a,1)-independent.evolve(changed,1)).norm(dim=-1)[0]
            if torch.count_nonzero(delta)!=1 or delta[4]==0:raise BaselineIntegrityError('independent cell leaked into another cell')
            for arm in ARMS:
                model=build(arm);state=model.init_state(1)
                for token in (7,9):state,logits=model.step(state,[token])
                torch.nn.functional.cross_entropy(logits,torch.tensor([11])).backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in model.parameters()):
                    raise BaselineIntegrityError('synthetic active-gradient gate failed')
                runtime=Runtime(model);snapshot=runtime.snapshot(state)
                if not torch.equal(runtime.restore(snapshot),state):raise BaselineIntegrityError('all-cell snapshot differs')
        return {'status':'PASS','rank_selection':sizing,'event_input_match':True,'independent_isolation':True,'all_parameter_gradients':'PASS','snapshot':'PASS','optimizer_updates':0}
    finally:torch.set_num_threads(count)
