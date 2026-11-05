"""Small isolated GPU timing pilot and read-only state inspector."""
import copy
import io
import time
import torch
from torch.nn import functional as F
from runtime import Runtime


def pilot(model):
    candidate=copy.deepcopy(model); values={}
    for k in (2,4,8):
        samples=[]
        for repeat in range(4):
            candidate.zero_grad(set_to_none=True); state=candidate.init_state(1)
            torch.cuda.synchronize(); start=time.perf_counter()
            for token in range(8):
                state,logits=candidate.step(state,[token],k)
            F.cross_entropy(logits,torch.tensor([8],device=state.device)).backward()
            torch.cuda.synchronize()
            if repeat: samples.append((time.perf_counter()-start)/8)
        values[str(k)]={'forward_backward_seconds_per_token':max(samples),'timing_statistic':'max_of_3_after_warmup'}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        candidate.step(candidate.init_state(1),[2])
        torch.cuda.synchronize()
    kernels=sum(str(event.device_type).endswith('CUDA') for event in profile.events())
    c=model.config
    recurrent_flops=(4 if c.arm=='F0' else 22)*c.cells*c.dim*c.dim
    flops=c.steps*recurrent_flops+2*c.cells*c.dim*c.dim+2*c.dim*c.vocab
    return {'kernel_launch_count':kernels,'forward_linear_flops_estimate':flops,'flops_excludes':'norm/activation/memory','K':values,'scientific_training_tokens':0,'capacity_tokens_in_5400_seconds':int(5400/values['4']['forward_backward_seconds_per_token'])}


@torch.no_grad()
def inspector(model,output):
    state=model.inject(model.init_state(1),[2]); trace=[]; probabilities=[]
    for _ in range(8):
        state=model.evolve(state,1,trace=trace)
        probabilities.append(float(model.read(state).softmax(-1).max()))
    runtime=Runtime(model); runtime.events=1; runtime.internal_steps=8
    start=time.perf_counter(); snapshot=runtime.snapshot(state)
    buffer=io.BytesIO(); torch.save(snapshot,buffer); save_seconds=time.perf_counter()-start
    start=time.perf_counter(); restored=runtime.restore(snapshot); restore_seconds=time.perf_counter()-start
    torch.testing.assert_close(restored,state,rtol=0,atol=0)
    torch.cuda.synchronize(); start=time.perf_counter()
    for _ in range(16): state,_=model.step(state,[2])
    torch.cuda.synchronize(); latency=(time.perf_counter()-start)/16
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for axis,key in zip(axes,('state_norm','delta_norm','message_norm')):
        axis.imshow([step[key][0] for step in trace],aspect='auto'); axis.set_title(key); axis.set_xlabel('Cell'); axis.set_ylabel('Internal step')
    fig.tight_layout(); fig.savefig(output/'inspector.png'); plt.close(fig)
    c=model.config
    # Multiply/add estimate, excluding normalization, activation and memory traffic.
    communication=6*c.cells*c.dim*c.dim
    update=c.cells*(4*c.dim*c.dim+12*c.dim*c.dim)
    readout=2*c.cells*c.dim*c.dim+2*c.dim*c.vocab
    return {'trace':trace,'max_probability_per_step':probabilities,'snapshot_bytes':buffer.tell(),
            'snapshot_save_seconds':save_seconds,'snapshot_restore_seconds':restore_seconds,
            'snapshot_restore':'PASS','decode_seconds_per_token':latency,
            'forward_flops_per_event_estimate':c.steps*(communication+update)+readout,
            'flops_scope':'C1 dense linear multiply/add estimate; excludes norms, activations, memory and kernel launches'}


@torch.no_grad()
def history_test(model,tokenizer):
    """Counterbalanced development stories with identical suffix and different earlier facts."""
    rows=[]
    for name in ('Alice','Bob','Lily','Tom'):
        histories=[]
        for color in ('red','blue'):
            prefix=f"{name} put a {color} ball in the box. Then everyone went outside. "
            state=model.init_state(1)
            for token in tokenizer.encode(prefix).ids: state,_=model.step(state,[token])
            histories.append(state)
        suffix=tokenizer.encode(f"Later, {name} opened the box. The ball was").ids
        for index,color in enumerate(('red','blue')):
            target=tokenizer.encode(' '+color).ids
            scores={}
            for label,state in [('original',histories[index]),('reset',model.reset(histories[index])),('swapped',histories[1-index])]:
                for token in suffix: state,logits=model.step(state,[token])
                loss=0.
                for token in target:
                    loss+=float(F.cross_entropy(logits,torch.tensor([token],device=state.device)))
                    state,logits=model.step(state,[token])
                scores[label]=loss/len(target)
            rows.append({'name':name,'color':color,'nll':scores})
    return {'task':'counterbalanced color-binding development stories','pairs':rows,
            'mean_nll':{key:sum(row['nll'][key] for row in rows)/len(rows) for key in ('original','reset','swapped')},
            'scientific_status':'EXPLORATORY_SINGLE_SEED_NOT_CONFIRMED'}
