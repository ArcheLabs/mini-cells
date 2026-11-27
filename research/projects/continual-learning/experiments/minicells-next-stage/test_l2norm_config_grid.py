"""CPU checks for native grid coverage and exact replay interpretation."""
from types import SimpleNamespace
import unittest

import torch

import m1_l2norm_config_grid as exp


class ConfigGridTest(unittest.TestCase):
    def test_all_25_configs_are_applied_and_restore_natural_candidates(self):
        def l2norm_fwd_kernel():
            pass

        choices = [SimpleNamespace(kwargs={"BT": bt}, num_warps=warps,
                                   num_stages=3, num_ctas=1, pre_hook=None)
                   for bt, warps in exp.GRID]
        kernel = SimpleNamespace(configs=choices, fn=l2norm_fwd_kernel)
        module = SimpleNamespace(l2norm_fwd_kernel=kernel)
        input = torch.ones((2, 256), dtype=torch.float16)

        def norm(x):
            self.assertEqual(len(kernel.configs), 1)
            kernel.best_config = kernel.configs[0]
            return x.clone()

        rows = exp.run_axis(norm, module, input, {}, input, (32, 16), input, (32, 4))
        self.assertEqual([(row["BT"], row["num_warps"]) for row in rows], list(exp.GRID))
        self.assertIs(kernel.configs, choices)
        self.assertEqual(exp.outcome(rows), "CONFIG_GRID_BITWISE_EQUAL")

        def different(x):
            out = norm(x)
            if kernel.best_config.num_warps == 8:
                out[0, 0] += 1
            return out

        rows = exp.run_axis(different, module, input, {}, input, (32, 16), input, (32, 4))
        self.assertEqual(exp.outcome(rows), "CONFIG_DEPENDENT_OUTPUT")
        rows = exp.run_axis(different, module, input, {}, input, (8, 8), input, (32, 4))
        self.assertEqual(exp.outcome(rows), "ARCHIVED_CONTEXT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
