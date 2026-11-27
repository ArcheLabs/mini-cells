"""CPU tests of observation, replay and original-bound comparisons."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from a3_operators import Observer, alignment, freeze, replay, thaw
from common import write
from minicells_native import MiniCells
import m1_live_vs_restored as a3
from m1_live_vs_restored import formal_pairs
from r1_equivalence import load_trace
from test_full_state import Backbone


class A3Test(unittest.TestCase):
    def test_first_divergence_requires_identical_inputs_and_detects_token(self):
        a, b = Observer(), Observer()
        a.active = b.active = True
        sample = torch.ones((1, 2, 3))
        delta = torch.zeros_like(sample)
        delta[:, 1, 0] = 1
        a.call("matmul", lambda x: x, (sample,), {})
        b.call("matmul", lambda x: x + delta, (sample,), {})
        a.call("residual", lambda x: x, (sample,), {})
        b.call("residual", lambda x: x, (sample + delta,), {})
        first, rows = alignment(a.rows, b.rows, a.tensors, b.tensors,
                                {"query.layer.0": .125}, query_length=2)
        self.assertEqual((first["operator"], first["query_token_index"]), ("matmul", 1))
        self.assertEqual(first["first_tensor_index"], [0, 1, 0])
        self.assertEqual(rows[1]["status"], "INPUT_DIFFERENCE")

    def test_bound_for_internal_node_is_exact_and_layer_bound_is_unchanged(self):
        a, b = Observer(), Observer()
        a.active = b.active = True
        sample = torch.ones((1, 2, 3))
        a.call("block.output", lambda x: x, (sample,), {})
        b.call("block.output", lambda x: x + .0625, (sample,), {})
        signal, rows = alignment(a.rows, b.rows, a.tensors, b.tensors,
                                 {"query.layer.0": .125})
        self.assertIsNone(signal)
        self.assertEqual(rows[0]["status"], "PASS")

    def test_dispatch_replay_preserves_nested_data_strides_and_topk_result(self):
        observer = Observer()
        observer.active = True
        value = torch.arange(12.).reshape(3, 4).T
        rows = {}
        spec = freeze((value, {"dtype": torch.float32}), rows, "roundtrip")
        out, kwargs = thaw(spec, rows, "cpu")
        torch.testing.assert_close(out, value)
        self.assertEqual(out.stride(), value.stride())
        self.assertEqual(kwargs["dtype"], torch.float32)
        observer.call("topk", torch.topk, (value, 2), {})
        self.assertEqual(observer.rows[0]["output"]["kind"], "tuple")
        with observer:
            total = value + value
        op = observer.rows[-1]
        _, replayed = replay(SimpleNamespace(device="cpu"), op["operator"], op["input"], observer.tensors)
        torch.testing.assert_close(replayed["replay.output"], total)

    def test_formal_protocol_repeats_from_one_bundle_in_three_cpu_processes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "work").mkdir()
            for phase in ("P0", "P2", "P3"):
                subprocess.run([sys.executable, __file__, tmp, phase], check=True)
            _, tensors = load_trace(root / "work" / "trace-P0")
            write(root / a3.BOUNDS_FILE, {name: 0.0 for name in tensors})
            result = a3.formal(root)
            self.assertEqual(len(result), 27)
            self.assertTrue(all(pair["passed"] for pair in result.values()))
            self.assertTrue((root / "query-P3-3.safetensors").is_file())

    def test_predefined_formal_pairs_cover_all_required_repetitions(self):
        pairs = list(formal_pairs())
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len(pairs), 27)
        for process in ("P1", "P2", "P3"):
            for trial in (1, 2, 3):
                self.assertIn(("P0", f"{process}-{trial}"), pairs)
            self.assertIn((f"{process}-1", f"{process}-3"), pairs)
        self.assertIn(("P2-3", "P3-3"), pairs)


def fixture_worker(root, phase):
    query = dict(tokens=[4, 6], candidates={"a": [7], "b": [8]})
    create = a3.produce

    def produce(mc, texts, work, **kwargs):
        with (patch("m1_continuation_isolation.PRE_LAYERS", 1),
              patch("m1_continuation_isolation.PRE_TENSORS", 9)):
            return create(mc, texts, work, **kwargs)

    with (patch.object(a3.MiniCells, "open_preview",
                       side_effect=lambda path: MiniCells._from_backbone(Backbone(), path)),
          patch.object(a3, "check_environment", return_value={"device": "cpu"}),
          patch.object(a3, "dataset"), patch.object(a3, "warm"),
          patch.object(a3, "produce", side_effect=produce),
          patch.object(a3, "probe_inputs", return_value=(["plain fact"], [[7, 6]], query))):
        a3.worker(root, phase)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        fixture_worker(Path(sys.argv[1]), sys.argv[2])
    else:
        unittest.main()
