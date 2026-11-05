"""Readout-only greedy/temperature/top-k/top-p sampling with a private CPU RNG."""
import math
import torch


def sample(logits, generator, temperature=0., top_k=0, top_p=1., forbidden=()):
    if not math.isfinite(temperature) or temperature<0 or not isinstance(top_k,int) or top_k<0 or not 0<top_p<=1:
        raise ValueError('invalid sampling configuration')
    scores=logits.detach().double().reshape(-1).cpu().clone()
    if not torch.isfinite(scores).all():raise FloatingPointError('non-finite generation logits')
    for token in forbidden:
        if token is not None:scores[token]=-torch.inf
    if not torch.isfinite(scores).any():raise ValueError('all tokens masked')
    if temperature==0:return int(scores.argmax()),0.
    scores=(scores-scores.max())/temperature
    if top_k:
        count=min(top_k,len(scores));selected=torch.topk(scores,count).indices
        keep=torch.zeros_like(scores,dtype=torch.bool);keep[selected]=True;scores[~keep]=-torch.inf
    if top_p<1:
        ordered,indices=scores.sort(descending=True);cumulative=ordered.softmax(-1).cumsum(-1)
        remove=cumulative>top_p;remove[1:]=remove[:-1].clone();remove[0]=False
        scores[indices[remove]]=-torch.inf
    probabilities=scores.softmax(-1)
    entropy=float(-(probabilities*probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum())
    return int(torch.multinomial(probabilities,1,generator=generator)),entropy
