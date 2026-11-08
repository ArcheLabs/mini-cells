"""M2 baselines reuse frozen M1 modules; E1 changes only visibility."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e0 import E0, Config, CachedTransformer


class E1(CachedTransformer):
    """Causal SWA: max(0,t-W+1)..t, current token included; W keys at most.

    Inherit the explicit mask, bounded cloned KV and differentiable uncached
    forward. Never use inference-only cached() as a training path.
    """
    pass


def build(arm, config, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm == 'E0':return E0(Config(**config))
        if arm == 'E1':return E1(Config(**config))
    raise ValueError('unsupported M2 arm')
