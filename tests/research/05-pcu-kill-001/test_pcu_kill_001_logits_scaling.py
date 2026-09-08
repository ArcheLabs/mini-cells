"""Regression tests for Granite CausalLM logits scaling in cached-tail execution."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from minicells.pcu_kill_001.backends import make_toy_model
from minicells.pcu_kill_001.cache import CachedTailRunner


def _scaled_toy_model(scale: float = 6.0):
    model = make_toy_model()
    model.config.logits_scaling = float(scale)
    model.logits_scaling = float(scale)
    raw_forward = model.forward

    def scaled_forward(*args, **kwargs):
        output = raw_forward(*args, **kwargs)
        return SimpleNamespace(logits=output.logits / float(scale))

    model.forward = scaled_forward
    return model


def test_cached_tail_applies_granite_logits_scaling() -> None:
    model = _scaled_toy_model(6.0)
    runner = CachedTailRunner(model, "model.layers.0")
    input_ids = torch.randint(0, model.vocab_size, (8, 8))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    cache = runner.capture(input_ids, attention_mask)
    tail_logits = runner.forward(cache)

    assert runner.logits_scaling == 6.0
    torch.testing.assert_close(tail_logits, full_logits, rtol=1e-5, atol=1e-6)
    assert runner.verify(cache, full_logits=full_logits).passed


def test_cached_training_path_uses_same_scaled_logits_contract() -> None:
    model = _scaled_toy_model(6.0)
    runner = CachedTailRunner(model, "model.layers.0")
    input_ids = torch.randint(0, model.vocab_size, (4, 8))
    attention_mask = torch.ones_like(input_ids)
    cache = runner.capture(input_ids, attention_mask)

    inference_logits = runner.forward(cache)
    training_logits = runner.forward_with_experts(cache, runner.moe.experts)

    torch.testing.assert_close(training_logits, inference_logits, rtol=1e-5, atol=1e-6)
