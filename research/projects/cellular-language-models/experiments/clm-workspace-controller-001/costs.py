"""Declared MAC=2 forward reference; match only within this explicit convention."""
from models import Model

def flops(arm,c):
    d=c['dim'];m=c['state_dim'];n=c['cells'];w=c['window'];f=c['ffn']
    attention=8*d*d+4*d*w
    parts={'head':2*d*c['vocab'],'attention':0,'ffn':0,'router':0,'controller':0}
    if arm.startswith('T0'):
        parts['attention']=c['layers']*attention;parts['ffn']=c['layers']*6*d*c['encoder_ffn']
    else:
        parts['attention']=attention+4*d*d+(1 if arm=='C3' else 2)*(4*d*d+4*d*w)
        parts['ffn']=6*d*c['encoder_ffn']+24*d*f
        if arm!='C0':parts['router']=4*d*n
        if arm in ('C2','C3'):parts['router']+=4*m*n;parts['controller']=6*m*(3*d+n+m)
    return {'total':sum(parts.values()),'parts':parts,'convention':'MAC=2, full-window forward matmul reference, includes actually executed one effective controller update; excludes elementwise/norm/dispatch/softmax and backward; not measured hardware FLOPs'}

def match(config,seed=17):
    from copy import deepcopy
    target=Model('C2',config,seed);parameters=sum(p.numel() for p in target.parameters());cost=flops('C2',config)['total']
    candidates=[]
    for width in range(8,4097,8):
        # Three SwiGLU blocks; parameters linear in width (bias-free primitive).
        c={**config,'encoder_ffn':width};d=c['dim']
        count=2*c['vocab']*d+2*d+c['layers']*(4*d*d+4*d+3*d*width)
        candidates.append((c,count,flops('T0',c)['total']))
    p=min(candidates,key=lambda x:abs(x[1]/parameters-1));f=min(candidates,key=lambda x:abs(x[2]/cost-1))
    result={}
    for name,item,budget in (('T0-P',p,'parameters'),('T0-F',f,'flops')):
        model=Model(name,item[0],seed);actual=sum(v.numel() for v in model.parameters());error=abs(actual/parameters-1) if budget=='parameters' else abs(flops(name,item[0])['total']/cost-1)
        result[name]={'config':item[0],'parameters':actual,'flops':flops(name,item[0]),'relative_matching_error':error,'matched':error<=.01,'matching_convention':budget}
    for name in ('C0','C1','C2','C3'):
        model=Model(name,config,seed);result[name]={'config':deepcopy(config),'parameters':sum(v.numel() for v in model.parameters()),'flops':flops(name,config)}
    return result
