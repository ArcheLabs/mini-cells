import unittest
from types import SimpleNamespace

import torch

from recall_diagnostic import score


class ScoringTest(unittest.TestCase):
    def test_alignment_and_candidate_isolation(self):
        seen = []

        def step(tokens, caches, weights):
            self.assertEqual(caches, [[]])
            caches[0].extend(tokens)
            seen.append(tokens)
            return torch.nn.functional.one_hot((torch.tensor([tokens]) + 1) % 6, 6).float()

        backend = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, **kw: [int(c) for c in text]),
            model=SimpleNamespace(lm_head=lambda hidden: hidden),
            step=step,
        )
        result = score(backend, "12", ["34", "5"], lambda: [[]])
        self.assertEqual(seen, [[1, 2, 3], [1, 2]])
        expected = torch.tensor([0., 0., 0., 1., 0., 0.]).log_softmax(0)
        torch.testing.assert_close(torch.tensor(result["candidates"][0]["token_logp"]),
                                   expected[3].repeat(2))
        self.assertEqual(result["mean_logp"]["gold_rank"], 1)
        self.assertGreater(result["mean_logp"]["margin"], 0)
        self.assertNotEqual(result["candidates"][0]["mean_logp"],
                            result["candidates"][0]["total_logp"])
