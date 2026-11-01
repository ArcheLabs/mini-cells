"""Dispatch-level arithmetic inventory. Unknown floating operators block compute matching."""
from collections import Counter
import math
import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._pytree import tree_flatten
from torch.nn import functional as F
from cached_transformer import CachedTransformer
from repair_models import build

# A scalar-equivalent arithmetic convention, not hardware instruction counts.
POINTWISE={'add':1,'sub':1,'mul':1,'div':1,'neg':1,'pow':1,'sqrt':1,'rsqrt':1,'reciprocal':1,'exp':1,'log':1,'sin':1,'cos':1,'abs':1,'sign':1,'silu':4,'silu_backward':10,'sigmoid':3,'sigmoid_backward':3,'tanh':1,'tanh_backward':3,'lerp':3,'addcmul':2,'addcdiv':2,'maximum':1,'minimum':1,'clamp':2,'clamp_min':1,'clamp_max':1,'where':0}
ZERO={'view','_unsafe_view','reshape','transpose','t','permute','as_strided','expand','slice','select','detach','clone','copy','_to_copy','to','alias','contiguous','empty','empty_like','empty_strided','zeros','zeros_like','ones','ones_like','new_zeros','new_empty','fill','zero','cat','stack','split','split_with_sizes','unbind','unsqueeze','squeeze','roll','index','index_select','gather','scatter','masked_fill','arange','scalar_tensor','lift_fresh','resolve_conj','resolve_neg','slice_backward'}


class ArithmeticCounter(FlopCounterMode):
    def __init__(self):
        super().__init__(display=False);self.phase='forward';self.totals=Counter();self.unknown=Counter();self.inventory=Counter()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};out=func(*args,**kwargs);packet=func._overloadpacket
        name=str(packet).split('.')[-1].rstrip('_');self.inventory[self.phase+':'+str(packet)]+=1
        inputs=[v for v in tree_flatten((args,kwargs))[0] if torch.is_tensor(v) and v.is_floating_point()]
        outputs=[v for v in tree_flatten(out)[0] if torch.is_tensor(v) and v.is_floating_point()]
        n=sum(v.numel() for v in outputs)
        if packet in self.flop_registry:
            count=self.flop_registry[packet](*args,**kwargs,out_val=out)
        elif name in ZERO or not inputs:count=0
        elif name in POINTWISE:count=n*POINTWISE[name]
        elif name in ('sum','mean'):
            count=max(0,inputs[0].numel()-n)+(n if name=='mean' else 0)
        elif name in ('_softmax','_safe_softmax','_log_softmax'):
            count=5*inputs[0].numel() # max comparison, shift, exp, reduction, normalization.
        elif name in ('_softmax_backward_data','_log_softmax_backward_data'):count=4*inputs[0].numel()
        elif name=='native_layer_norm':
            x=inputs[0];width=math.prod(args[1]);count=8*x.numel()+3*(x.numel()//width)
        elif name=='native_layer_norm_backward':count=16*inputs[0].numel()
        elif name in ('embedding','embedding_dense_backward','index_add','scatter_add'):
            count=inputs[0].numel() if name!='embedding' else 0
        elif name=='amax':count=0 # comparisons, not floating arithmetic
        elif name=='index_put':count=args[2].numel() if (len(args)>3 and args[3]) or kwargs.get('accumulate',False) else 0
        elif name=='linalg_vector_norm' and (len(args)<2 or args[1]==2):count=2*inputs[0].numel()
        elif name=='nll_loss_forward':count=inputs[0].shape[0]*2
        elif name=='nll_loss_backward':count=inputs[0].numel()+n
        elif name in ('eq','ne','gt','ge','lt','le','isfinite','isinf','isnan','all','any','_assert_async','_local_scalar_dense'):count=0
        else:self.unknown[self.phase+':'+str(packet)]+=1;count=0
        self.totals[self.phase]+=int(count)
        return out


def measure(name,config,device='cpu',batch=4,unroll=64,history=128):
    torch.manual_seed(713)
    model=(CachedTransformer(config) if name=='T1-PM-KV' else build(name)).to(device)
    inputs=torch.randint(4,2048,(batch,unroll),device=device);targets=torch.randint(4,2048,(batch,unroll),device=device)
    caches=None
    if name=='T1-PM-KV' and history:
        with torch.no_grad():caches=[model._forward(torch.randint(4,2048,(1,history),device=device),save_cache=True)[1] for _ in range(batch)]
    # lr=0 profiles AdamW arithmetic on a disposable synthetic fixture. Never load
    # a trained checkpoint, never consume corpus data, never change model weights.
    before=[p.detach().clone() for p in model.parameters()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.,foreach=False)
    counter=ArithmeticCounter()
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH),counter:
        if name=='T1-PM-KV':
            from training_path import teacher_forced_lanes
            parts=[[{'reset':not bool(history),'inputs':inputs[lane].tolist(),'targets':targets[lane].tolist()}] for lane in range(batch)]
            loss,_,_=teacher_forced_lanes(model,parts,caches)
        else:
            state=model.init_state(batch);losses=[]
            for t in range(unroll):
                state,logits=model.step(state,inputs[:,t]);losses.append(F.cross_entropy(logits,targets[:,t],reduction='none').sum())
            loss=sum(losses)/(batch*unroll)
        counter.phase='backward';loss.backward()
        counter.phase='gradient_clipping';torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False,error_if_nonfinite=True)
        counter.phase='optimizer';optimizer.step()
    unchanged=all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
    if not unchanged:raise ValueError('zero-LR synthetic arithmetic fixture changed weights')
    return {'status':'REGISTERED_ARITHMETIC_COVERED' if not counter.unknown else 'BLOCKED_UNSUPPORTED_ARITHMETIC_OPERATORS',
        'operator_inventory_complete':not bool(counter.unknown),'formal_compute_matching_ready':False,'model':name,'device':device,'batch':batch,'unroll':unroll,'synthetic_prediction_events':batch*unroll,'history_tokens':history if name=='T1-PM-KV' else 'persistent recurrent state',
        'committed_training_tokens':0,'real_model_optimizer_updates':0,'synthetic_optimizer_calls':1,'model_weights_unchanged':unchanged,
        'phase_flops':dict(counter.totals),'total_registered_flops':sum(counter.totals.values()),'unsupported_operators':dict(counter.unknown),'operator_inventory':dict(counter.inventory),
        'convention':'semantic scalar operation equivalents v1; FMA=2; transcendental=1; normalization/activation reference formulas; memory/comparison operations excluded; not hardware FLOP instructions',
        'caveats':['operator registration is framework/version dependent','full-state/steady-context synthetic fixture; actual variable document/reset shapes require live per-update counting','token/FLOPs counters must advance only for accepted optimizer updates','compute matching cannot use totals while any floating operator is unsupported']}


def profile(config,device='cpu',batch=4,unroll=64):
    reports={name:measure(name,config,device,batch,unroll) for name in ('C1-BROADCAST','IC-ALL-SELF','F0-PM-BROADCAST','T1-PM-KV')}
    return {'status':'OPERATOR_POLICY_VALIDATED' if all(r['operator_inventory_complete'] for r in reports.values()) else 'UNSUPPORTED_OPERATORS_BLOCK_FORMAL_COMPUTE_MATCHING',
            'models':reports,'training_tokens_consumed':0,'hardware_flop_measurement':False,'counting_attention_backend':'MATH_REFERENCE_OPERATOR_CONVENTION',
            'formal_compute_matching_ready':False,'remaining_gate':'validate live actual-update counting across document boundaries, masks and target hardware; then freeze numerical budget checkpoints'}
