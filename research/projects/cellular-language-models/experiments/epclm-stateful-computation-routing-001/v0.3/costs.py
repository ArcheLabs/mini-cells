"""Declared matrix FLOPs reference, separate from measured CUDA timing."""
def analytic(model):
    c=model.config;d=c['dim'];s=c['state_dim'];r=c['router_dim'];n=c['cells'];t=c['chunk'];window=c['window'];f=c['ffn']
    attention=8*d*d+4*d*window
    parts={'attention':attention*(c['layers'] if model.arm=='T0' else 1),'router':0,'cell_core':0,'state_read':0,'state_write':0,'state_update':0,'head':2*d*c['vocab']}
    if model.arm=='T0':parts['ffn']=c['layers']*6*d*f
    else:
        parts['router']=2*(2*d*r+2*n*r)+2*n*d*r/t
        parts['cell_core']=2*(6*d*f+2*d*d)
        if model.arm in ('M1','M2'):
            parts['state_read']=2*2*s*d
            parts['state_write']=2*2*3*d*s
            parts['state_update']=n*6*s*(s+s)/t
            if model.arm=='M2':parts['router']+=2*n*s*r/t
    return {'status':'ANALYTIC_REFERENCE_NOT_HARDWARE_MEASUREMENT','parts_flops_per_token':parts,'total_flops_per_token':sum(parts.values()),'active_flops_per_token':sum(parts.values()),'convention':'MAC=2, steady full-window attention reference, dense masked execution may cost more; excludes norm/softmax/gates, no backward estimate; shared GRU evaluates all slots then masks inactive writes; selected Core evaluated twice only'}
