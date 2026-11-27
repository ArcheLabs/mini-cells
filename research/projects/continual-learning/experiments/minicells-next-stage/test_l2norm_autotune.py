"""Tests for the diagnostic's singleton intervention and causal decision."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from a31_norm import FIELDS
import m1_l2norm_autotune as exp


class AutotuneTest(unittest.TestCase):
    def test_singleton_applies_only_to_two_measured_calls_and_restores_candidates(self):
        def l2norm_fwd_kernel():
            pass

        options = [SimpleNamespace(kwargs={"BT": bt}, num_warps=warps, pre_hook=None,
                                   num_stages=3, num_ctas=1)
                   for bt in (8, 16, 32, 64, 128) for warps in (1, 2, 4, 8, 16)]
        kernel = SimpleNamespace(configs=options, fn=l2norm_fwd_kernel, best_config=options[0])
        module = SimpleNamespace(l2norm_fwd_kernel=kernel)
        chosen = exp.fixed_config(kernel)
        used = []

        def norm(x, eps=1e-6):
            used.append(tuple(kernel.configs))
            kernel.best_config = kernel.configs[0]
            return x.clone()

        observer = exp.PinnedBoundary(norm, module, chosen)
        observer.active = True
        for _ in range(3):
            observer.norm(torch.ones((1, 1, 256), dtype=torch.float16))
            self.assertIs(kernel.configs, options)
        self.assertEqual(used[:2], [(chosen,), (chosen,)])
        self.assertEqual(used[2], tuple(options))
        exp.validate_calls(observer.calls, fixed=True)

    def test_pin_uses_exact_outputs_and_retains_inconclusive_if_natural_configs_match(self):
        def controls():
            fields = {name: dict(bitwise_equal=True, layout_equal=True) for name in FIELDS}
            fields.update({axis: dict(config_equal=True, norm_arguments_equal=True,
                                     same_first_routed_slot=True) for axis in "QK"})
            return fields

        lanes = {"natural": controls(), "fixed": controls()}
        pairs = {"P0_vs_P2": lanes}
        lanes["natural"]["K_POST_NORM"]["bitwise_equal"] = False
        lanes["natural"]["K_RECURRENT"]["bitwise_equal"] = False
        self.assertEqual(exp.classify(pairs, True)[0], "INCONCLUSIVE")
        lanes["natural"]["K"]["config_equal"] = False
        self.assertEqual(exp.classify(pairs, True)[0], "AUTOTUNE_HYPOTHESIS_SUPPORTED")
        self.assertEqual(exp.classify(pairs, False)[0], "PRE_NORM_INPUT_DIVERGED")
        lanes["fixed"]["K_RECURRENT"]["bitwise_equal"] = False
        self.assertEqual(exp.classify(pairs, True)[0], "DIVERGENCE_WITH_FIXED_CONFIG")

    def test_missing_original_bundle_blocks_before_any_gpu_jobs(self):
        with TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            with (patch.object(sys, "argv", ["probe", str(out)]),
                  patch.object(exp, "recover_bundle", side_effect=exp.BundleUnavailable("missing")),
                  patch.object(exp.subprocess, "run") as spawn):
                exp.main()
            spawn.assert_not_called()
            decision = json.loads((out / "decision.json").read_text())
            self.assertEqual(decision["status"], "BLOCKED")
            self.assertEqual(decision["state_equivalence"], "NOT_RUN_BY_SCOPE")


if __name__ == "__main__":
    unittest.main()
