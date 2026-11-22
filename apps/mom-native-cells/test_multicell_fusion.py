from types import SimpleNamespace
import unittest

import torch
from torch import nn

from multicell_fusion import CellReadResult, FusionConfig, FusionGate, FusionReader, fuse, permutation_errors


class FusionTests(unittest.TestCase):
    def setUp(self):
        rng = torch.Generator().manual_seed(42)
        self.hidden = torch.randn(1, 3, 8, generator=rng)
        self.probs = torch.rand(1, 3, 4, generator=rng).softmax(-1)
        top, self.indices = self.probs.topk(2, dim=-1)
        self.top = top / top.sum(-1, keepdim=True)
        self.reads = [
            CellReadResult(torch.randn(1, 3, 4, 2, 4, generator=rng), self.probs, self.indices,
                           self.top, torch.randn(1, 3, 2, 4, generator=rng))
            for _ in range(3)
        ]
        self.selected = [
            CellReadResult(r.routed_outputs.gather(2, self.indices[..., None, None].expand(1, 3, 2, 2, 4)),
                           r.router_probabilities, r.topk_indices, r.topk_weights, r.shared_output)
            for r in self.reads
        ]
        self.gate = FusionGate(8, 16)

    def test_factorization_and_shared(self):
        a = fuse(self.hidden, self.selected, FusionConfig("hierarchical_equal"))[0]
        b = fuse(self.hidden, self.selected, FusionConfig("global_factorized"))[0]
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        expected = torch.stack([r.shared_output for r in self.reads]).mean(0)
        for read in self.selected:
            read.routed_outputs.zero_()
        actual = fuse(self.hidden, self.selected, FusionConfig("global_factorized"))[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_factorization_matches_at_fp16_rounding_boundary(self):
        routed = torch.tensor([[-1.404296875, -0.1383056640625],
                               [2.6484375, -0.95849609375]], dtype=torch.float16)
        shared = torch.tensor([0.10211181640625, -0.818359375], dtype=torch.float16)
        weights = torch.tensor([0.77490234375, 0.2252197265625], dtype=torch.float16)
        tokens = torch.zeros(1, 1, 1, dtype=torch.float16)
        reads = [CellReadResult(routed[i].reshape(1, 1, 2, 1, 1), None, None,
                                weights.reshape(1, 1, 2), shared[i].reshape(1, 1, 1, 1))
                 for i in range(2)]
        product = routed.float() * weights.float()
        old_grouped = (product.sum(-1) + shared).mean().half()
        old_global = ((product * 0.5).sum() + shared.float().mean()).half()
        self.assertNotEqual(old_grouped, old_global)
        grouped = fuse(tokens, reads, FusionConfig("hierarchical_equal"))[0]
        global_pool = fuse(tokens, reads, FusionConfig("global_factorized"))[0]
        expected = ((routed.double() * weights.double()).sum() + shared.double().sum()) * 0.5
        torch.testing.assert_close(grouped, global_pool, rtol=0, atol=0)
        torch.testing.assert_close(grouped.reshape(()), expected.half(), rtol=0, atol=0)

    def test_permutation_and_joint_normalization(self):
        for mode in ("hierarchical_gate", "global_topk_gate", "global_all4_gate"):
            reads = self.reads if mode == "global_all4_gate" else self.selected
            config = FusionConfig(mode, router_prior=float(mode == "global_all4_gate"), null_candidate=True)
            normal = fuse(self.hidden, reads, config, self.gate)
            reverse = fuse(self.hidden, reads[::-1], config, self.gate)
            out, alpha, beta = normal
            rev, swapped_alpha, swapped_beta = reverse
            errors = permutation_errors(normal, reverse, len(reads), reads[0].routed_outputs.shape[2])
            self.assertLessEqual(max(errors.values()), 1e-6)
            # A genuine candidate-weight mismatch must remain visible.
            changed = permutation_errors(normal, (rev, swapped_alpha + 2e-5, swapped_beta),
                                         len(reads), reads[0].routed_outputs.shape[2])
            self.assertGreater(changed["cell"], 1e-6)
            torch.testing.assert_close(out, rev, rtol=0, atol=1e-6)
            torch.testing.assert_close(alpha[..., :-1], swapped_alpha[..., :-1].flip(-1))
            if beta is not None:
                self.assertEqual(beta.shape[-1], 3 * reads[0].routed_outputs.shape[2] + 1)
                torch.testing.assert_close(beta.sum(-1), torch.ones_like(beta[..., 0]))
                torch.testing.assert_close(beta[..., :-1],
                                           swapped_beta[..., :-1].unflatten(-1, (3, -1)).flip(-2).flatten(-2))

    def test_stability_and_empty_path(self):
        config = FusionConfig("global_all4_gate", router_prior=1)
        self.probs[..., 0] = 0
        self.reads[0].routed_outputs.mul_(1e7)
        out, alpha, beta = fuse(self.hidden, self.reads, config, self.gate)
        self.assertTrue(all(torch.isfinite(t).all() for t in (out, alpha, beta)))
        backend = SimpleNamespace(
            stream=lambda tensors: ["native", tensors],
            step=lambda tokens, caches, weights: (tokens, caches, weights),
        )
        self.assertEqual(FusionReader(backend).step([2], [], config), ([2], [["native", {}]], [1.0]))

    def test_only_gate_trainable(self):
        backbone = nn.Linear(8, 8).requires_grad_(False)
        hidden = backbone(self.hidden)
        out, _, _ = fuse(hidden, self.selected, FusionConfig("global_topk_gate"), self.gate)
        out.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.gate.parameters()), 0)

    def test_native_read_adapter_pack_and_recurrence(self):
        from multicell_fusion import Diagnostics, read_layer

        def transform(hidden, mask, banks, selected, capacity):
            return hidden.unsqueeze(0) * mask.permute(2, 0, 1).unsqueeze(-1), None, selected, hidden.shape[1], None

        def reconstruct(outputs, indices, selected, batch, length, k, routing_weights, mask):
            values = outputs.permute(1, 2, 0, 3, 4)
            ix = selected[..., None, None].expand(batch, length, k, *values.shape[-2:])
            return (values.gather(2, ix) * routing_weights[..., None, None]).sum(2)

        class Router:
            calls = 0

            def __call__(self, hidden):
                self.calls += 1
                return hidden.new_tensor([0., 1., 2., 3.]).expand(*hidden.shape[:2], 4)

        class Cache(dict):
            def update(self, *, layer_idx, recurrent_state, conv_state, offset):
                self[layer_idx] = dict(recurrent_state=recurrent_state, conv_state=conv_state, offset=offset)

        calls = []

        def recurrent(q, k, v, g, beta, initial_state, **kwargs):
            calls.append(id(initial_state))
            return v + initial_state, initial_state + 1

        def conv(x, cache, output_final_state):
            return x, x[..., -1:].clone()

        project = nn.Identity()
        scalar = nn.Linear(8, 2)
        router = Router()
        attn = SimpleNamespace(num_heads=2, num_memories=4, topk=2, capacity=1., gate=router,
                               q_proj=project, shared_k=project, shared_v=project, shared_b=scalar,
                               shared_a=scalar, single_kv_proj=True, A_log=torch.zeros(2),
                               dt_bias=torch.zeros(2), q_conv1d=conv, k_conv1d=conv, v_conv1d=conv,
                               layer_idx=0, use_gate=False, o_norm=project, o_proj=project)
        ops = SimpleNamespace(transform=transform, reconstruct=reconstruct, l2_norm=project,
                              fused_recurrent_gated_delta_rule=recurrent)

        def caches():
            return [Cache({0: dict(recurrent_state=[torch.tensor(float(10*c+i)) for i in range(5)],
                                   conv_state=([None, None],) * 3, offset=0)}) for c in range(2)]

        work = caches()
        trace = Diagnostics(check_permutation=True)
        result, _ = read_layer(attn, ops, self.hidden, work, FusionConfig("hierarchical_equal"), None, trace)
        self.assertEqual(router.calls, 1)
        self.assertEqual(trace.layers[0]["fixed_input_permutation_max_abs"], {"cell": 0})
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(set(calls)), 6)
        self.assertEqual([int(state) for state in work[0][0]["recurrent_state"]], [0, 1, 3, 4, 5])
        self.assertEqual([int(state) for state in work[1][0]["recurrent_state"]], [10, 11, 13, 14, 15])
        self.assertEqual(work[0][0]["offset"], self.hidden.shape[1])
        self.assertIsNot(work[0][0]["conv_state"][0][0], work[1][0]["conv_state"][0][0])
        factor, _ = read_layer(attn, ops, self.hidden, caches(), FusionConfig("global_factorized"), None, None)
        torch.testing.assert_close(result, factor, atol=4e-6, rtol=0)
        calls.clear()
        read_layer(attn, ops, self.hidden, caches(), FusionConfig("global_all4_gate"), self.gate, None)
        self.assertEqual(len(calls), 10)
