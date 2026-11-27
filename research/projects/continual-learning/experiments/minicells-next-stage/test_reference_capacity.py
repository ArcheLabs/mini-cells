"""CPU regression tests for all-target scoring, the mandatory gate and reporting."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from minicells_native import MiniCells

import m1_reference_capacity as exp
from test_next_stage import TinyBackbone


class ReferenceCapacityTest(unittest.TestCase):
    @staticmethod
    def synthetic_facts(tokenizer, count):
        def fact(i):
            texts = {"A": f"Project {i} has amber", "B": f"Project {i} has green"}
            return dict(project=str(i), values={"A": "amber", "B": "green"}, texts=texts,
                        write_tokens={k: tokenizer.encode(v, add_special_tokens=False)
                                      for k, v in texts.items()},
                        queries={q: dict(query_ids=[3, 4], candidates={"A": [8, 10], "B": [11, 12]})
                                 for q in ("q1", "q2")})
        return [fact(i) for i in range(count)]

    @torch.inference_mode()
    def test_every_target_gets_two_fresh_cells(self):
        backbone = TinyBackbone()
        # Use facts whose token representation actually differs on swap.
        with patch.object(backbone.tokenizer, "encode",
                          side_effect=lambda text, **kw: [8 if word == "amber" else len(word) + 2
                                                           for word in text.split()]):
            facts = self.synthetic_facts(backbone.tokenizer, 3)
            group = exp.assemble(facts, backbone.tokenizer)
            full = exp.live_factory(backbone, group["original"]["tokens"])
            first = full()
            first.layers[0]["recurrent_state"][0].zero_()
            self.assertNotEqual(full().layers[0]["recurrent_state"][0].count_nonzero(), 0)
            self.assertEqual(full().get_seq_length(), len(group["original"]["tokens"]))
            with TemporaryDirectory() as temp:
                with MiniCells._from_backbone(backbone, Path(temp) / "registry") as mc:
                    rows = [exp.evaluate_target(mc, 1, 3, group, pos, full) for pos in range(3)]
                    self.assertEqual(mc.cells.list(), [])
        self.assertEqual([row["position"] for row in rows], [1, 2, 3])
        states = [tuple(row["state"].values()) for row in rows]
        self.assertEqual(len({state["cell_id"] for pair in states for state in pair}), 6)
        self.assertEqual(len({a["state_sha256"] for a, _ in states}), 1)
        self.assertTrue(all(a["state_sha256"] != b["state_sha256"] for a, b in states))
        self.assertEqual([row["target_token_offset"] for row in rows], [0, 4, 8])

    def test_stage_zero_gate_and_all_target_counts(self):
        mc = SimpleNamespace(_backbone=SimpleNamespace(tokenizer=None))
        dataset = {str(n): [dict(original=dict(tokens=[1]))] * 12 for n in exp.LEVELS}
        for gate_fails in (True, False):
            def row(_mc, index, level, _group, position, _original):
                hit = not (level == 1 and index <= 3 and gate_fails)
                word = {lane + "_" + condition: dict(gold_nll=1) for lane in ("full", "cell")
                        for condition in ("original", "swapped")}
                return dict(position=position+1, full_flip=True, cell_flip=hit, wordings={"q": word})

            with self.subTest(gate_fails=gate_fails), TemporaryDirectory() as temp:
                out = Path(temp)
                result = {}
                with (patch.object(exp, "build_dataset", return_value=dataset),
                      patch.object(exp, "DATASET_SHA256", exp.canonical_hash(dataset)),
                      patch.object(exp, "live_factory"),
                      patch.object(exp, "evaluate_target", side_effect=row),
                      patch("builtins.print")):
                    exp.run(mc, out, result)
                expected = [1] if gate_fails else [1, 3, 8, 16]
                self.assertEqual(result["levels_completed"], expected)
                self.assertEqual(result["status"], "SINGLE_FACT_GATE_FAILED" if gate_fails else "PASS")
                self.assertEqual(len((out / "cases.jsonl").read_text().splitlines()), 12 * sum(expected))
                stats = json.loads((out / "statistics.json").read_text())
                self.assertEqual(stats["1"]["cell"]["groups_passed"], 9 if gate_fails else 12)

    def test_one_failed_target_fails_group_and_full_control_takes_precedence(self):
        cells = [{"position": n, "cell_flip": n != 3, "full_flip": True,
                  "wordings": {"q": {f"{lane}_{cond}": {"gold_nll": 1}
                                     for lane in ("cell", "full") for cond in ("original", "swapped")}}}
                 for n in (1, 2, 3)]
        stats = exp.summarize([cells])
        self.assertEqual(stats["cell"]["facts_passed"], 2)
        self.assertEqual(stats["cell"]["groups_passed"], 0)
        all_levels = {str(n): {"full": {"groups_passed": 12}, "cell": {"groups_passed": 12}}
                      for n in exp.LEVELS}
        all_levels["8"]["cell"]["groups_passed"] = 9
        self.assertEqual(exp.decide(all_levels)[0], "CAPACITY_GATE_FAILED")
        all_levels["8"]["full"]["groups_passed"] = 9
        self.assertEqual(exp.decide(all_levels)[0], "FULL_CONTEXT_GATE_FAILED")


if __name__ == "__main__":
    unittest.main()
