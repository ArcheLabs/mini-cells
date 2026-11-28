"""CPU coverage for the A3.3 diagnostic interpretations and frozen gate."""
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from unittest.mock import patch

from common import write
import m1_process_history as exp


class ProcessHistoryTest(unittest.TestCase):
    def test_group_interpretations(self):
        labels = ("P0", "A1", "A2", "B1", "B2")
        cases = (
            ((0, 0, 0, 0, 0), "ALL_EXACT"),
            ((0, 1, 1, 0, 0), "WRITE_HISTORY_REPRODUCES_LIVE"),
            ((0, 0, 0, 1, 1), "WRITE_HISTORY_SHIFTS_AWAY_FROM_LIVE"),
            ((0, 1, 1, 1, 1), "ARMS_MATCH_DIFFER_FROM_LIVE"),
            ((0, 1, 1, 2, 2), "ARMS_DIFFER_NEITHER_MATCHES_LIVE"),
            ((0, 1, 2, 0, 0), "WITHIN_ARM_DIFFERENCE"),
        )
        for values, expected in cases:
            groups = dict(zip(labels, values, strict=True))
            pairs = {f"{a}_vs_{b}": {"passed": True, "exact": groups[a] == groups[b]}
                     for a, b in combinations(labels, 2)}
            with self.subTest(expected=expected):
                self.assertEqual(exp.interpretation(pairs), expected)

    def test_start_state_gate_then_complete_ten_pair_comparison(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            work = out / "work"
            work.mkdir()
            archive = work / "prefix.cell"
            archive.write_bytes(b"fixture")
            digest = exp.digest_file(archive)
            state = dict(structure_sha256="structure", cursor=29, layer_count=24,
                         tensor_sha256={str(i): "same" for i in range(263)})
            write(work / "shared.json", dict(bundle_sha256=digest, pre=state))
            for pid, label in enumerate(exp.ORDER, 1):
                actual = dict(state, cursor=28) if label == "B2" else state
                write(out / f"policy-{label}.json", dict(pid=pid, gpu_uuid="GPU-fixture",
                      arm=exp.ARMS[label], fixed_config={"BT": 32}, source="fixed"))
                write(work / f"{label}.json", dict(pid=pid, state=actual, start_failures=[],
                      bundle_unchanged=True, bundle_sha256=digest, input_sha256="shared-input"))
            with patch.object(exp, "load_trace") as trace:
                with self.assertRaisesRegex(RuntimeError, "strict start-state"):
                    exp.evaluate(out, {})
                trace.assert_not_called()

            receipt = exp.load_json(work / "B2.json")
            write(work / "B2.json", dict(receipt, state=state))
            write(out / exp.BOUNDS_FILE, {"query.layer.0": 0.125})
            structures = dict(pre={}, post={})

            def synthetic_trace(path):
                offset = 0.25 if path.name in {"trace-A1", "trace-A2"} else 0.0
                return structures, {"query.layer.0": torch.tensor([[offset]])}

            result = {}
            with patch.object(exp, "load_trace", side_effect=synthetic_trace):
                exp.evaluate(out, result)
            report = exp.load_json(out / "process_history.json")
            self.assertEqual(len(report["pairs"]), 10)
            self.assertEqual(sum(pair["passed"] for pair in report["pairs"].values()), 4)
            self.assertEqual(report["pattern"], "WRITE_HISTORY_REPRODUCES_LIVE")
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["engineering_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
