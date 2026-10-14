"""Load the real historical implementation without changing its files."""

import subprocess
import types
import copy
import numpy as np
from pd001.configs import ROOT, BASE, load
from pd001.runner import fixture_config

BASELINE = "2e77369b0df1ce704863a815dacc48301736dde0"


def original(name):
    relative = (BASE / "implementation/pd001" / (name + ".py")).relative_to(ROOT)
    source = subprocess.check_output(
        ["git", "show", BASELINE + ":" + str(relative)], cwd=ROOT, text=True
    )
    module = types.ModuleType("pd001.baseline_" + name)
    module.__package__ = "pd001"
    exec(compile(source, BASELINE + "/" + str(relative), "exec"), module.__dict__)
    return module


def fixture(proposal, arm="P50-F16", steps=4):
    c = copy.deepcopy(proposal)
    c["training"].update(
        steps=steps,
        effective_batch=2,
        microbatch=1,
        sequence_length=3,
        tokens=steps * 6,
        checkpoint_every=1,
    )
    m = fixture_config(c, arm)
    m.update(budget=5)
    tokens = np.arange(128, dtype=np.int32) % m["vocab_size"]
    assets = {
        "train": tokens,
        "train_starts": [0, 3, 6, 9],
        "manifest_sha256": "f" * 64,
        "tokenizer_sha256": "e" * 64,
    }
    return c, load()[1], m, assets


def final_loader(*a, **kw):
    assert kw["frozen_checkpoint"]["complete"]
    return {
        "tokens": np.arange(128, dtype=np.int32) % 23,
        "starts": [0, 3],
        "groups": [0, 1],
        "evaluation_sha": "d" * 64,
    }


def tensors_equal(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            tensors_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            tensors_equal(x, y)
    else:
        assert a == b
