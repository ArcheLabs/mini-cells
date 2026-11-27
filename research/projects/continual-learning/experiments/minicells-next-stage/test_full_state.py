"""Full-state schema and cross-process SDK behavior with a stateful CPU fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from minicells_native import MiniCells
from minicells_native.errors import CellArchiveCorruptedError
from minicells_native.full_state import FullState
from m1_reference_capacity import digest_file
import m1_full_state_bridge as r1
from m1_full_state_bridge import ARMS, PATHS, summarize
from r1_equivalence import compare, envelope, trace
from test_next_stage import Cache, TinyBackbone


class FixtureCache(Cache):
    def to_legacy_cache(self):
        return tuple(self.layers)

    def get_seq_length(self, layer_idx=0):
        return self.position if layer_idx < len(self.layers) else 0

    def get_max_length(self):
        return None

    @classmethod
    def from_legacy_cache(cls, layers, seen_tokens):
        return super().from_legacy_cache(layers if isinstance(layers, list) else [], seen_tokens)


class Backbone(TinyBackbone):
    cache_type = FixtureCache

    def __init__(self):
        super().__init__()
        self.model.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))

    def step(self, ids, caches, weights, **kwargs):
        cache = caches[0]
        previous = cache.layers[0]["conv_state"][0][0] if cache.layers else None
        feature = 0 if previous is None else float(previous.flatten()[0])
        hidden = super().step(ids, caches, weights, **kwargs)
        cache.position += 1  # Packed cursor may differ from the text-token counter.
        base = torch.arange(6, dtype=torch.float16).reshape(2, 3).T + sum(ids) + feature
        cache.layers[0]["conv_state"] = tuple([base.T, base.clone()] for _ in range(3))
        cache.layers[0]["ffn_state"] = ([base.clone()], None)
        hidden[..., 8] += feature
        hidden[..., 9] += cache.position
        return self.model.model.layers[0](hidden)


def subprocess_operation(root, phase):
    with MiniCells._from_backbone(Backbone(), root / phase) as mc:
        bundle = root / "prefix.cell"
        cell = mc.cells.import_cell(bundle)
        if phase == "append":
            mc.cells.append(cell.id, "next fact", resume=False)
            mc.cells.export(cell.id, root / "appended.cell")
        else:
            before = digest_file(mc.cells._store.path(cell.id))
            out = mc.query("plain", cells=[cell.id], max_new_tokens=1)
            assert before == digest_file(mc.cells._store.path(cell.id))
            (root / "query.json").write_text(json.dumps(dict(text=out.text, pid=os.getpid())))


class FullStateTest(unittest.TestCase):
    def test_v2_layout_cursor_and_independent_ablations(self):
        backbone = Backbone()
        cache = backbone.stream({})
        backbone.step([2, 10], [cache], [1.0])
        state = FullState.capture(backbone, cache)
        work = state.restore(backbone)
        node = work.layers[0]["conv_state"][0][0]
        self.assertEqual(node.stride(), cache.layers[0]["conv_state"][0][0].stride())
        self.assertEqual(state.structure["per_layer_seen_tokens"], [3])
        self.assertIs(type(work.layers[0]["conv_state"]), tuple)
        self.assertIs(type(work.layers[0]["conv_state"][0]), list)
        self.assertIs(type(work.layers[0]["ffn_state"]), tuple)
        torch.testing.assert_close(work.layers[0]["ffn_state"][0][0], cache.layers[0]["ffn_state"][0][0])
        torch.testing.assert_close(node, cache.layers[0]["conv_state"][0][0])
        node.zero_()
        self.assertNotEqual(state.restore(backbone).layers[0]["conv_state"][0][0].count_nonzero(), 0)
        self.assertEqual(state.restore(backbone, cursor=False).get_seq_length(), 0)
        self.assertIsNone(state.restore(backbone, conv=False).layers[0]["conv_state"][0][0])

    def test_sdk_cross_process_append_query_and_fork(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with MiniCells._from_backbone(Backbone(), root / "writer") as mc:
                old = mc.cells.create("legacy")
                cell = mc.cells.create("v2", full_state=True)
                mc.cells.write(cell.id, "first fact", resume=False)
                mc.cells.export(cell.id, root / "prefix.cell")
                self.assertEqual(old.state_completeness, "recurrent_only")
                self.assertEqual(cell.state_completeness, "full_state")
                continued = mc.cells.append(cell.id, "next fact", resume=False)
                expected = mc.cells._state(continued.id)
                expected_answer = mc.query("plain", cells=[continued.id], max_new_tokens=1).text
                fork = mc.cells.fork(cell.id)
                path = mc.cells._store.path(cell.id)
                digest = digest_file(path)
                mc.cells.append(fork.id, "extra", resume=False)
                self.assertEqual(digest_file(path), digest)
                mc.cells.delete(fork.id)
            subprocess.run([sys.executable, __file__, tmp, "append"], check=True)
            with MiniCells._from_backbone(Backbone(), root / "verify") as mc:
                imported = mc.cells.import_cell(root / "appended.cell")
                actual = mc.cells._state(imported.id)
                self.assertEqual(actual.structure, expected.structure)
                for key, tensor in expected.tensors.items():
                    torch.testing.assert_close(actual.tensors[key], tensor, atol=0, rtol=0)
                self.assertEqual(imported.tokens_written, 4)
                self.assertEqual(expected.structure["sequence_length"], 6)
            (root / "prefix.cell").unlink()
            (root / "appended.cell").rename(root / "prefix.cell")
            subprocess.run([sys.executable, __file__, tmp, "query"], check=True)
            query = json.loads((root / "query.json").read_text())
            self.assertNotEqual(query["pid"], os.getpid())
            self.assertEqual(query["text"], expected_answer)

    def test_full_state_resume_and_bundle_integrity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with MiniCells._from_backbone(Backbone(), root / "writer") as mc:
                cell = mc.cells.create("v2", full_state=True)
                calls = iter([False, True])
                text = "one two three four"
                self.assertIsNone(mc.cells.write(cell.id, text, chunk_size=2, cancelled=lambda: next(calls)))
                self.assertEqual(mc.cells.get(cell.id).tokens_written, 0)
                mc.cells.write(cell.id, text, chunk_size=2)
                actual = mc.cells._state(cell.id)
                ref = mc.cells.create("fresh", full_state=True)
                mc.cells.write(ref.id, text, chunk_size=2, resume=False)
                self.assertEqual(actual.structure, mc.cells._state(ref.id).structure)
                for key, tensor in actual.tensors.items():
                    torch.testing.assert_close(tensor, mc.cells._state(ref.id).tensors[key])
                export = mc.cells.export(cell.id, root / "export.cell")
                bad = root / "bad.cell"
                with zipfile.ZipFile(export) as source, zipfile.ZipFile(bad, "w") as output:
                    for name in source.namelist():
                        output.writestr(name, source.read(name) + (b" " if name == "cache_structure.json" else b""))
                with self.assertRaises(CellArchiveCorruptedError):
                    mc.cells.import_cell(bad)

    def test_failed_state_gate_does_not_launch_bridge(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"

            def spawn(argv):
                self.assertEqual(argv[-1], "write")
                (out / "state_equivalence.json").write_text(json.dumps({
                    "same_process_roundtrip": {"passed": False, "failures": ["pre.0.conv_state.0.0"]}
                }))
                return SimpleNamespace(returncode=1)

            with (patch.object(sys, "argv", ["r1", str(out)]), patch.object(r1, "baseline"),
                  patch.object(r1.subprocess, "run", side_effect=spawn) as run):
                r1.main()
            run.assert_called_once()
            decision = json.loads((out / "decision.json").read_text())
            self.assertEqual(decision["state_equivalence"], "FAIL")
            self.assertEqual(decision["bridge_status"], "NOT_RUN")
            self.assertEqual(decision["capacity_status"], "NOT_RUN_BY_SCOPE")
            self.assertIn("pre.0.conv_state.0.0", decision["diagnosis"])

    def test_bridge_keeps_original_combined_threshold(self):
        cases = []
        for i in range(12):
            arms = {}
            for arm in ARMS:
                rates = {key: {"orig": [{"margin": 1 if arm != "combined" or i < 6 else -1,
                                          "gold_nll": 1.0, "logp": {"orig": -1.0}}]} for key in PATHS}
                arms[arm] = dict(paths=rates, no_memory=[dict(margins={"orig": -1})])
            cases.append(dict(arms=arms))
        gates = summarize(cases)["gates"]
        self.assertEqual(gates["reference_E"], "PASS")
        self.assertEqual(gates["combined_E"], "FAIL")
        self.assertEqual(gates["combined_A"], "FAIL")
        self.assertEqual(gates["combined_no_memory"], "PASS")

    @torch.inference_mode()
    def test_equivalence_uses_only_live_noise_and_reports_first_difference(self):
        backbone = Backbone()
        query = dict(tokens=[4, 6], candidates={"a": [7], "b": [8]})

        def run(tokens):
            def factory():
                cache = backbone.stream({})
                backbone.step(tokens, [cache], [1.0])
                return cache
            return trace(backbone, factory, query)

        ref, repeated, other = run([3, 5]), run([3, 5]), run([5, 6])
        limits = envelope(ref, repeated)
        self.assertTrue(all(v == 0 for v in limits.values()))
        self.assertTrue(compare(ref, repeated, limits)[0]["passed"])
        report, rows = compare(ref, other, limits)
        self.assertFalse(report["passed"])
        self.assertIsNotNone(report["first_divergent_token"])
        self.assertEqual(report["first_divergent_layer"], 0)
        self.assertTrue(any("conv_state" in row["field"] and row["status"] == "FAIL" for row in rows))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        subprocess_operation(Path(sys.argv[1]), sys.argv[2])
    else:
        unittest.main()
