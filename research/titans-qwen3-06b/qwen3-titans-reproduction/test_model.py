import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from model import install, hidden_states, SEGMENT, LAYERS


class AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen3Config(hidden_size=1024, intermediate_size=128, vocab_size=64,
                              num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                              head_dim=128, max_position_embeddings=40960)
        cls.base = Qwen3ForCausalLM(config).eval()
        cls.ids = torch.randint(0, 64, (1, 19))
        cls.mask = torch.ones_like(cls.ids, dtype=torch.bool)
        with torch.no_grad():
            cls.original = cls.base.model(cls.ids, attention_mask=cls.mask, use_cache=False).last_hidden_state
        install(cls.base)

    def test_native_and_training_boundary(self):
        with torch.no_grad():
            self.assertTrue(torch.equal(self.original, hidden_states(self.base, self.ids, self.mask, 'native')))
            seg = hidden_states(self.base, self.ids, self.mask, 'segmented')
            titans = hidden_states(self.base, self.ids, self.mask, 'titans')
        torch.testing.assert_close(seg, titans, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(~titans.isfinite()).item(), 0)
        for layer_idx in LAYERS:
            layer = self.base.model.layers[layer_idx].self_attn
            self.assertFalse(any(p.requires_grad for p in layer.native.parameters()))
            for i, w in enumerate(layer.memory.memory_model_parameters):
                expected = torch.eye(128).expand(8, 128, 128) * (4 if i == 2 else 1)
                torch.testing.assert_close(w, expected, rtol=0, atol=0)

    def test_segment_causality_and_value_handoff(self):
        q = torch.randint(0, 64, (1, SEGMENT+3))
        changed = q.clone()
        changed[:, SEGMENT-1] = (changed[:, SEGMENT-1]+1) % 64
        with torch.no_grad():
            a = hidden_states(self.base, q, torch.ones_like(q), 'segmented')
            b = hidden_states(self.base, changed, torch.ones_like(q), 'segmented')
        torch.testing.assert_close(a[:, :SEGMENT-1], b[:, :SEGMENT-1], rtol=0, atol=0)
        # The untouched upper Qwen layers can carry information across the
        # boundary. Test the segmented cut at layer 3 itself via a hook.
        layer = self.base.model.layers[3].self_attn
        outputs = []
        handle = layer.register_forward_hook(lambda m, args, out: outputs.append(out[0]))
        try:
            with torch.no_grad():
                for ids in (q, changed):
                    hidden_states(self.base, ids, torch.ones_like(ids), 'segmented')
        finally:
            handle.remove()
        torch.testing.assert_close(outputs[0][:, SEGMENT:], outputs[1][:, SEGMENT:], rtol=0, atol=0)

    def test_memory_cut_and_reset(self):
        memory = self.base.model.layers[0].self_attn.memory
        gen = torch.Generator().manual_seed(19)
        def sample(heads):
            return .03 * torch.randn(1, heads, SEGMENT+2, 128, generator=gen)
        q, k, v = sample(16), sample(8), sample(8)
        context = torch.zeros(1, SEGMENT+2, 1024)
        a, mix, state = memory(q, k, v, context=context)
        b, _, _ = memory(q, k, -v, context=context)
        c, _, _ = memory(q, k, v, context=context)
        self.assertEqual(int(state.seq_index), SEGMENT)
        self.assertEqual(a[:, :SEGMENT].count_nonzero().item(), 0)
        self.assertEqual(mix[:, :SEGMENT].count_nonzero().item(), 0)
        self.assertGreater(float((a[:, SEGMENT:] - b[:, SEGMENT:]).abs().max()), 0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
        a[:, SEGMENT:].square().mean().backward()
        self.assertTrue(all(w.grad is not None and w.grad.isfinite().all()
                            for w in memory.memory_model_parameters))


if __name__ == '__main__':
    unittest.main()
