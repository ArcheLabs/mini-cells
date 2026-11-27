"""Test both M1-002A launch decisions and read-only scoring on a CPU fixture."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minicells_native import MiniCells

import m1_bridge as bridge
from test_next_stage import TinyBackbone


class BridgeTest(unittest.TestCase):
    def test_scoring_persists_each_value_and_uses_independent_queries(self):
        case = dict(labels=["one", "two"], texts={"one": "Test one", "two": "Test two"},
                    wordings=[dict(text="query", tokens=[3], candidates={"one": [4], "two": [6]})] * 2)
        narrowed = {**case, "wordings": [
            {**row, "candidates": {k: [v[0], 8] for k, v in row["candidates"].items()}}
            for row in case["wordings"]
        ]}
        item = dict(index=0, arms={"short": case, "long": narrowed})
        with TemporaryDirectory() as temp:
            with MiniCells._from_backbone(TinyBackbone(), Path(temp) / "registry") as mc:
                row = bridge.evaluate(mc, item)
                self.assertEqual(mc.cells.list(), [])
        for arm in row["arms"].values():
            self.assertEqual(set(arm["cell"]), {"one", "two"})
            for wordings in arm["cell"].values():
                self.assertEqual(len(wordings), 2)
                self.assertEqual(wordings[0], wordings[1])
            self.assertTrue(all(len(value) == 64 for value in arm["state_sha256"].values()))
        self.assertEqual(row["arms"]["short"]["state_sha256"], row["arms"]["long"]["state_sha256"])

    def test_capacity_only_launches_after_passing_controls(self):
        def wordings(key, margin):
            return [{"margins": {key: margin}}] * 2

        def fixture(fail, name):
            live = {key: wordings(key, 1.0) for key in ("one", "two")}
            return dict(full=live, cell={**live, "two": wordings("two", -1.0)}
                        if fail and name == "combined" else live, no_memory=wordings("one", -1.0))

        def completed(_mc, _out, result):
            result.update(status="PASS", diagnosis="Capacity passed", next_action="Review",
                          capacity_summary={"sentinel": 1})

        mc = SimpleNamespace(_backbone=SimpleNamespace(tokenizer=None))
        indices = [dict(index=i) for i in range(2)]
        for fail in (True, False):
            with self.subTest(fail=fail), TemporaryDirectory() as temp:
                out = Path(temp)
                rows = [dict(index=idx, arms={n: fixture(fail, n) for n in bridge.ARMS})
                        for idx in range(2)]
                result = {}
                with (patch.object(bridge, "build_dataset", return_value=indices),
                      patch.object(bridge, "DATASET_SHA256", bridge.canonical_hash(indices)),
                      patch.object(bridge, "evaluate", side_effect=rows),
                      patch.object(bridge.capacity, "run", side_effect=completed) as capacity_run):
                    bridge.run(mc, out, result)
                self.assertEqual(capacity_run.call_count, int(not fail))
                self.assertEqual(result["capacity_status"], "SKIPPED" if fail else "PASS")
                self.assertEqual(len((out / "bridge_cases.jsonl").read_text().splitlines()), 2)
                stats = json.loads((out / "bridge_statistics.json").read_text())
                comparison = stats["contrasts"]["combined_vs_wording"]["cell"]
                self.assertEqual(comparison, dict(gains=0, losses=2 if fail else 0))


if __name__ == "__main__":
    unittest.main()
