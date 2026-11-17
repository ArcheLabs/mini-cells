import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from p1_1_evaluation import evaluate_p1_1


class EvaluationTest(unittest.TestCase):
    def test_interventions_and_diagnostics_restore_state(self):
        native = object()
        memory = nn.Module()
        memory.history_mix_proj = nn.Linear(1, 1, bias=False)
        memory.history_mix = nn.Parameter(torch.zeros(1))
        memory.read_only = lambda x: (x, None)
        wrapper = SimpleNamespace(attention=native, state=None, pending=None,
                                  read_override=None, memory=memory)
        writes = []
        def commit():
            writes.append(wrapper.pending)
            wrapper.state = wrapper.pending
        wrapper.commit = commit
        block = SimpleNamespace(self_attn=wrapper)

        class Model(nn.Module):
            def forward(self, input_ids, position_ids, use_cache):
                self.assert_no_cache(use_cache)
                value = self.weight.expand(1, input_ids.shape[1], 1)
                if block.self_attn is wrapper:
                    wrapper.pending = input_ids[0, 0].item()
                    if wrapper.state is not None:
                        read, _ = memory.read_only(value * wrapper.state)
                        gate = memory.history_mix_proj(value).sigmoid()
                        value = (1 - gate) * value + gate * (
                            torch.zeros_like(read) if wrapper.read_override == "zero" else read)
                return SimpleNamespace(logits=torch.cat((value, -value), dim=-1))

        model = Model()
        model.assert_no_cache = self.assertFalse
        model.weight = nn.Parameter(torch.ones(1))
        model.model = SimpleNamespace(layers=[None] * 3 + [block])
        del memory.read_only
        type(memory).read_only = lambda _self, x: (x, None)
        try:
            with patch("p1_1_evaluation.CHUNK", 2):
                result = evaluate_p1_1(model, wrapper, [[0, 1] * 4, [1, 0] * 4],
                                       {0: [1], 1: [0]})
        finally:
            del type(memory).read_only
        self.assertEqual(len(writes), 6)
        row = result["documents"][1]
        self.assertEqual(row["empty"], row["chunked_attention"])
        self.assertNotEqual(row["zero_read"], row["empty"])
        self.assertGreater(result["diagnostics"][0]["read_rms_delta"][0], 0)
        self.assertIs(block.self_attn, wrapper)
        self.assertIsNone(wrapper.state)
        self.assertIsNone(wrapper.read_override)
