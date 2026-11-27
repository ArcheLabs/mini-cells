"""CPU checks for fixed policy scope and the original full-equivalence gate."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import torch

from common import write
from m1_l2norm_config_grid import GRID
import m1_l2norm_fixed_e2e as exp


class FixedE2ETest(unittest.TestCase):
    def test_scoped_policy_audits_write_and_query_and_restores_original_config(self):
        def l2norm_fwd_kernel():
            pass

        options = [SimpleNamespace(kwargs={"BT": bt}, num_warps=warps,
                                   num_stages=3, num_ctas=1, pre_hook=None,
                                   all_kwargs=lambda bt=bt, warps=warps: {"BT": bt, "num_warps": warps})
                   for bt, warps in GRID]
        kernel = SimpleNamespace(configs=options, fn=l2norm_fwd_kernel)

        def norm(x):
            self.assertEqual(len(kernel.configs), 1)
            kernel.best_config = kernel.configs[0]
            return x.clone()

        module = SimpleNamespace(l2norm_fwd_kernel=kernel, l2norm_fwd=norm)
        calls = Counter()
        segment = "write"
        with exp.fixed_forward(module, calls, lambda: segment) as chosen:
            self.assertEqual(chosen, {"BT": 32, "num_warps": 4})
            module.l2norm_fwd(torch.ones((1, 256), dtype=torch.float16))
            segment = "query/P0"
            module.l2norm_fwd(torch.ones((1, 256), dtype=torch.float16))
            with self.assertRaisesRegex(RuntimeError, "audited"):
                module.l2norm_fwd(torch.ones((1, 256)))
        self.assertIs(kernel.configs, options)
        self.assertIs(module.l2norm_fwd, norm)
        self.assertEqual({key[0] for key in calls}, {"write", "query/P0"})

    def test_any_failed_original_pair_remains_fail(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            for pid, phase in enumerate(exp.PHASES, 1):
                write(out / f"policy-{phase}.json", dict(pid=pid, gpu_uuid="GPU-1", fixed_config={"BT": 32}))
            pairs = {f"{a}_vs_{b}": {"passed": True} for a, b in exp.a3.formal_pairs()}
            key = next(iter(pairs))
            for fails in (False, True):
                pairs[key]["passed"] = not fails
                (out / "layer_comparison.jsonl").write_text("".join(
                    json.dumps(dict(pair=name, status="PASS" if row["passed"] else "FAIL",
                                    max_absolute=0 if row["passed"] else 1)) + "\n"
                    for name, row in pairs.items()))
                result = {}
                with patch.object(exp.a3, "formal", return_value=pairs):
                    exp.evaluate(out, result)
                self.assertEqual(result["status"], "FAIL" if fails else "PASS")
                receipt = exp.load_json(out / "fixed_config_summary.json")
                self.assertEqual(receipt["pairs_passed"], 26 if fails else 27)


if __name__ == "__main__":
    unittest.main()
