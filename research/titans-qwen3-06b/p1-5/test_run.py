import unittest
import run
import torch
from instrument import compare


class StatisticsTest(unittest.TestCase):
    def test_per_head_deltas(self):
        a = torch.ones(8, 2)
        result = compare(a, 2*a, 3*a)
        self.assertEqual(result["swap_delta"]["rms"], [1.] * 8)
        self.assertEqual(result["swap_random_norm_ratio"], [.5] * 8)
