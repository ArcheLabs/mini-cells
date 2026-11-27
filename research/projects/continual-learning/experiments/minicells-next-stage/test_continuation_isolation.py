"""CPU checks of matched-start continuation and the immutable R1 thresholds."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from minicells_native import MiniCells
import m1_continuation_isolation as a2
from common import write
from r1_equivalence import load_trace, trace
from test_full_state import Backbone


class ContinuationTest(unittest.TestCase):
    @torch.inference_mode()
    def test_original_sdk_cache_is_the_primary_query_cache(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with MiniCells._from_backbone(Backbone(), root / "writer") as mc:
                backbone = mc._backbone
                generate = a2.FullState.capture
                cache_seen = []

                def hook(owner, cache):
                    cache_seen.append(cache)
                    return generate(owner, cache)

                with (patch.object(a2, "PRE_LAYERS", 1), patch.object(a2, "PRE_TENSORS", 9),
                      patch.object(a2.FullState, "capture", side_effect=hook)):
                    cell, live, _ = a2.produce(mc, ["plain fact"], root)
                self.assertIs(live, cache_seen[-1])
                frozen = deepcopy(live)
                original_step = backbone.step
                seen = []

                def observe(tokens, caches, weights):
                    seen.append(caches[0])
                    return original_step(tokens, caches, weights)

                query = dict(tokens=[4, 6], candidates={"a": [7], "b": [8]})
                with patch.object(backbone, "step", side_effect=observe):
                    a2.ensure_readonly(mc, cell, lambda: trace(
                        backbone, lambda: deepcopy(frozen), query, primary_cache=live))
                self.assertIs(seen[0], live)
                self.assertTrue(all(cache is not live for cache in seen[1:]))
                self.assertGreater(live.get_seq_length(), frozen.get_seq_length())

    def test_independent_cpu_workers_export_all_three_pairs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "work").mkdir()
            for name in a2.INSTANCES:
                subprocess.run([sys.executable, __file__, temp, name], check=True)
            _, tensors = load_trace(root / "work" / "trace_A")
            write(root / a2.BOUNDS_FILE, {key: 0.0 for key in tensors})
            result = {}
            a2.evaluate(root, result)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["a2_pairs"]), 3)
            self.assertTrue((root / "query_traces.safetensors").is_file())

    def test_start_is_exact_even_when_historical_pre_bound_is_nonzero(self):
        structure = dict(sequence_length=29, layer_count=24)
        before = {"a": torch.tensor([1.0])}
        altered = {"a": torch.tensor([1.125])}
        self.assertEqual(a2.start_differences(a2.fingerprint(structure, before),
                                              a2.fingerprint(structure, altered)), ["a"])
        traces = (
            (dict(pre=structure, post=structure), {"pre.a": before["a"], "post.a": before["a"]}),
            (dict(pre=structure, post=structure), {"pre.a": altered["a"], "post.a": altered["a"]}),
        )
        result, rows = a2.compare_pair(*traces, {"pre.a": .2, "post.a": .125})
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"], ["pre.a"])
        self.assertEqual([row["status"] for row in rows], ["FAIL", "PASS"])

    def test_decisions_distinguish_restored_worker_drift(self):
        for passes, expected in (
            ((True, True, True), "MATCHED_STATE_CONTINUATION_PASS"),
            ((False, True, False), "LIVE_RESTORED_DIVERGED"),
            ((True, False, False), "RESTORED_WORKERS_DIVERGED"),
        ):
            results = {f"{a}_vs_{b}": {"passed": hit}
                       for (a, b), hit in zip(a2.PAIRS, passes, strict=True)}
            self.assertEqual(a2.interpretation(results)[0], expected)


def fixture_worker(root, instance):
    query = dict(tokens=[4, 6], candidates={"a": [7], "b": [8]})
    with (patch.object(a2.MiniCells, "open_preview",
                       side_effect=lambda path: MiniCells._from_backbone(Backbone(), path)),
          patch.object(a2, "check_environment", return_value={"device": "cpu"}),
          patch.object(a2, "dataset"),
          patch.object(a2, "probe_inputs", return_value=(["plain fact"], [[7, 6]], query)),
          patch.object(a2, "PRE_LAYERS", 1), patch.object(a2, "PRE_TENSORS", 9)):
        a2.worker(root, instance)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        fixture_worker(Path(sys.argv[1]), sys.argv[2])
    else:
        unittest.main()
