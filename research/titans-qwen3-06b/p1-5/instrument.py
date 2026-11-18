"""Observe the pinned single-chunk update without replacing its computation."""
from contextlib import contextmanager
import torch
from replay import capture, same_tree


@contextmanager
def observe(wrapper):
    memory = wrapper.memory
    records = []
    commit = wrapper.commit

    def traced():
        assert wrapper.state is None
        initial = tuple(t.clone() for t in memory.init_weights(1))
        observed = {}
        original = memory.memory_grads

        def grads(weights, keys, lr, values):
            result = original(weights, keys, lr, values)
            observed.update(keys=keys.detach(), loss=result[1].detach())
            return result

        memory.memory_grads = grads
        try:
            with capture(memory, write=True) as inner:
                commit()
        finally:
            del memory.memory_grads
        state = wrapper.state
        fast = tuple(t[:, -1] for t in state.updates)
        same_tree(fast, state.states[0])
        momentum = tuple(t[0] for t in state.states[1])
        records.append(dict(**observed, **inner, initial=initial, fast=fast,
                            fast_delta=tuple(a-b for a,b in zip(fast, initial)),
                            momentum=momentum,
                            previous_momentum=tuple(torch.zeros_like(t) for t in momentum),
                            decay_anchor=tuple(initial_t * inner['decay'][:, 0].reshape(
                                -1, *([1] * (initial_t.ndim-1))) for initial_t in initial),
                            decay_carry=tuple(initial_t * (1-inner['decay'][:, 0]).reshape(
                                -1, *([1] * (initial_t.ndim-1))) for initial_t in initial)))

    wrapper.commit = traced
    try:
        yield records
    finally:
        del wrapper.commit


def compare(a, b, c):
    a,b,c = (x.double().reshape(x.shape[0], -1) for x in (a,b,c))
    for x in (a,b,c):
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite diagnostic")
    def cosine(x,y):
        return torch.nn.functional.cosine_similarity(x,y,dim=-1).tolist()
    def stats(x):
        return dict(norm=x.norm(dim=-1).tolist(), rms=x.square().mean(-1).sqrt().tolist())
    d,e=b-a,c-a
    return dict(baseline=stats(a), swap=stats(b), random=stats(c),
                swap_delta=stats(d), random_delta=stats(e),
                swap_relative=(d.norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-30)).tolist(),
                random_relative=(e.norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-30)).tolist(),
                swap_cosine=cosine(a,b), random_cosine=cosine(a,c),
                delta_cosine=cosine(d,e),
                swap_random_norm_ratio=(d.norm(dim=-1)/e.norm(dim=-1).clamp_min(1e-30)).tolist())


def summarize(records, tensors):
    a,b,c=records
    for key in ('keys','lr','beta','decay','initial'):
        same_tree(a[key],b[key])
        same_tree(a[key],c[key])
    result={}
    for key in ('raw_gradient','weighted_gradient','surprise','momentum','fast','fast_delta',
                'initial','previous_momentum','decay_anchor','decay_carry','loss'):
        vals=[r[key] for r in records]
        if not isinstance(vals[0],tuple):
            vals=[(v,) for v in vals]
        result[key]=[compare(*v) for v in zip(*vals)]
    result['read']=[compare(*(t.transpose(0,1) for t in tensors['reads'].values()))]
    result['controls']={k:dict(values=a[k].tolist(),mean=a[k].mean().item(),
                               min=a[k].min().item(),max=a[k].max().item()) for k in ('lr','beta','decay')}
    return result
