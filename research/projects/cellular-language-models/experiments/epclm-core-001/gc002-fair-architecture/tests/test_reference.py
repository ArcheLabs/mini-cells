"""Two required engineering contracts; no training loop or CPU benchmark."""
import torch
from cached_transformer import Config
from preparation import equivalence,parameter_match


def test_kv_cache_equivalence():
    torch.set_num_threads(1)
    report=equivalence(Config())
    assert report['status']=='PASS' and len(report['cases'])==3
    assert report['memory']['persistent_state_bytes_per_stream']==262144


def test_parameter_matching():
    torch.set_num_threads(1)
    config,report=parameter_match()
    assert abs(report['relative_difference_percent'])<=1 and report['optimizer_updates']==0
    assert equivalence(config)['status']=='PASS'
