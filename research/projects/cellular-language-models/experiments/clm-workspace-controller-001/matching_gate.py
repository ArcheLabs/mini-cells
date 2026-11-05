"""Pre-outcome executable conditional controls; report both incompatible budgets."""
from copy import deepcopy
from models import Model
from costs import flops


def freeze(c):
    target=Model('C2',c);total=sum(p.numel() for p in target.parameters());reference=flops('C2',c)['total']
    base=Model('C1',c);base_count=sum(p.numel() for p in base.parameters())
    old_router=sum(p.numel() for p in base.router.parameters());candidates=[]
    for width in range(8,4097,8):
        q={**c,'matched_router_width':width}
        params=base_count-old_router+(c['dim']+1)*width+(width+1)*c['cells']
        active=flops('C1',c)['total']-4*c['dim']*c['cells']+4*width*(c['dim']+c['cells'])
        candidates.append({'config':q,'parameters':params,'active_forward_flops':active,'parameter_error':params/total-1,'flops_error':active/reference-1})
    # Only declared arithmetic is searched; actual chosen models are checked below.
    chosen={name:min(candidates,key=lambda x:abs(x[key])) for name,key in (('P','parameter_error'),('F','flops_error'))}
    for entry in chosen.values():
        assert sum(p.numel() for p in Model('C1-MatchedRouter',entry['config']).parameters())==entry['parameters']
    return {'H2':'CONTROL_DESIGN_FROZEN','H3':'CONTROL_DESIGN_FROZEN','H2_controls':{'C0-PoolFixed':{'pair':[0,1],'active_training_experts':2,'note':'only two experts trained; not an effective-capacity match'},'C0-PoolBalanced':{'schedule':'(absolute input position + 2*step) mod8, adjacent partner; independent of input representation','active_training_experts':8,'router_parameters':0,'pool':'same independently initialized eight SwiGLU experts as C1'}},'H3_controls':chosen,'matching_note':'P matches total parameters and reports extra FLOPs; F matches active matrix cost and reports parameter deficit; do not claim both matched when impossible','trigger':{'H2':'C1 pooled mean NLL improves over C0 by >=1% and >=2/3 seeds favor C1, OR normal-vs-random/fixed route gain >=1% with >=2/3 positive seeds','H3':'C2 pooled mean NLL improves over C1 by >=1% with >=2/3 positive seeds; require State-Off consistency before claiming controller utility'},'review_budget':{'targets':'same separately frozen Main target count; never auto-reduced','seeds':[17,29,43],'activation':'conditional explicit execution only, not run during this gate task'},'decision_rule':'effect >=1% relative NLL, >=2/3 seed sign agreement, paired document intervals reported separately; matched control must not explain away effect; no scientific claim from freeze alone','status':'PASS'}
