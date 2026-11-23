import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cells import Cells
from compositional_recall import read, write
from gdr_diagnostics import FIELDS, reference, statistics
from residual_metrics import curve, group_bootstrap, metrics, rank_signal
from run_multicell_residual import (LABELS, PRIMARY, SEEDS, dataset, replay_fixed_inputs, restored_check,
                                    score_trace, trace_drift, verify_trace)


class ResidualExperimentTest(unittest.TestCase):
    def test_seed_isolation_and_double_dependency(self):
        groups = [row for seed in SEEDS for row in dataset(seed, 32)]
        self.assertEqual(len(groups), 96)
        samples = [x for group in groups for x in group["identifiers"]]
        self.assertEqual(len(samples), len(set(samples)))
        for g in groups:
            left, right, _, _, joint = g["queries"]
            self.assertEqual(set(joint["relevant"]), set(left["relevant"] + right["relevant"]))
            self.assertNotIn(joint["gold"].strip(), g["facts"][left["relevant"][0]])
            project = g["facts"][left["relevant"][0]].split()[1]
            self.assertNotIn(project, g["facts"][right["relevant"][0]])

    def test_aggregate_masks_and_permutation(self):
        def trace(order):
            return SimpleNamespace(
                banks=[(i, 0, bank, torch.tensor([0, 1]), torch.full((2, 2, len(FIELDS)), float(v + bank)))
                       for i, v in enumerate(order) for bank in (0, 2, 4)],
                routers=[{"layer": 0, "topk_mass": [.7, .8], "topk_indices": [[0, 2], [0, 2]]}])
        q = dict(id="fixture")
        with patch("run_multicell_residual.LAYERS", 1):
            scores = score_trace(trace([0, 1]), [1, 2], ["A", "B"], q)
        self.assertEqual(scores[PRIMARY], {"A": -1, "B": -2})
        self.assertEqual(scores["shared_relative_residual_norm"], {"A": -4, "B": -5})
        self.assertEqual(len(set(scores["router_mass"].values())), 1)
        self.assertEqual(verify_trace(trace([0, 1]), trace([1, 0]), {"A": "A", "B": "B"},
                                     ["A", "B"], ["B", "A"], atol=0, rtol=0), 0)

    def test_group_bootstrap_and_research_gate(self):
        rows = [dict(group=f"{seed}-{block}", seed=seed, relevant=["A", "B"],
                     score={"A": .9, "B": .8, "C": .1, "D": .05})
                for seed in SEEDS for block in range(4)]
        results = metrics(rows, "score")
        results["ci95"] = group_bootstrap(rows, "score", repeats=30)
        self.assertEqual(results["full_coverage_at_2"], 1)
        self.assertEqual(results["ci95"]["auroc"], [1, 1])
        self.assertEqual(curve([(1., True), (1., False)]), (.5, .5))
        self.assertEqual(rank_signal(results, {str(seed): results for seed in SEEDS}, [0.9], True, .5),
                         "RESEARCH_PASS")
        self.assertNotEqual(rank_signal(results, {str(seed): results for seed in SEEDS}, [.9], False, .5),
                            "RESEARCH_PASS")


    def test_restore_separates_fixed_input_parity_from_rollout_drift(self):
        q = k = v = torch.ones((1, 2, 1, 1))
        g = beta = torch.zeros((1, 2, 1))
        state = torch.ones((1, 1, 1, 1))
        operands = (q, k, v, g, beta)
        native, _, prediction, error = reference(*operands, state)
        expected = statistics(native, prediction, error, v, beta, g)[0, 1:]
        banks = [(0, 0, 0, torch.tensor([0]), expected)]

        upstream = SimpleNamespace(fused_recurrent_gated_delta_rule=lambda *args, initial_state, **kw:
                                   reference(*args, initial_state)[:2])
        with patch.dict("sys.modules", {"mom": SimpleNamespace(),
                                       "mom.layers": SimpleNamespace(mom_gated_deltanet=upstream)}):
            result = replay_fixed_inputs(SimpleNamespace(device="cpu"), {"0.0": operands},
                                         {"banks": banks, "packing": {"0.0": torch.tensor([1])}},
                                         {"A": {"0.0": state}}, reference)
        self.assertEqual(result, {"banks": 1, "max_abs": 0})
        drifted = expected.clone()
        drifted[..., FIELDS.index("prediction_cosine")] -= 0.03
        report = trace_drift(banks, [(0, 0, 0, torch.tensor([0]), drifted)])
        self.assertEqual(report["fields"]["prediction_cosine"]["outside_old_tolerance"], 1)
        self.assertEqual(report["fields"]["relative_residual_norm"]["outside_old_tolerance"], 0)


    def test_restore_worker_reads_archive_and_replays_saved_operands(self):
        upstream = SimpleNamespace(fused_recurrent_gated_delta_rule=lambda *args, initial_state, **kw:
                                   reference(*args, initial_state)[:2])
        backbone = SimpleNamespace(device="cpu", identity={"model": "test"})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            store = Cells(root / "original", backbone)
            meta = {label: store.create(label) for label in LABELS}
            write(output / "restore-cells.json", meta)

            query_id = "fixture-T1"
            q = k = v = torch.ones((1, 2, 4, 1))
            g = beta = torch.zeros((1, 2, 4))
            operands = (q, k, v, g, beta)
            native, _, predicted, residual = reference(*operands)
            values = statistics(native, predicted, residual, v, beta, g)[0, 1:].tolist()
            with gzip.open(output / "sample-residual-records.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(query_id=query_id, cell="A", layer=0, bank=0,
                                             token_index=[0], values=values)) + "\n")
            torch.save({"inputs": {"0.0": operands}, "packing": {"0.0": torch.tensor([1])}},
                       store.root / f"{query_id}-inputs.pt")
            row = {"id": query_id, PRIMARY: dict.fromkeys(LABELS, -1.0)}
            write(output / "smoke-cell-scores.json", [row])
            args = SimpleNamespace(scratch=root, output=output)
            assets = {"smoke": [{"queries": [{"id": query_id}]}]}
            rollout = dict(rows=[{**row, "rollout_trace": {"common_positions": 1}}], parity={}, integrity={})
            with patch.dict("sys.modules", {"mom": SimpleNamespace(),
                                           "mom.layers": SimpleNamespace(mom_gated_deltanet=upstream)}):
                with patch("run_multicell_residual.run_group", return_value=rollout) as run:
                    restored_check(backbone, args, assets, reference)
            self.assertEqual(len(run.call_args.kwargs["original_traces"][query_id]), 1)
            self.assertEqual(read(output / "restore-fixed-input.json")["queries"][query_id],
                             {"banks": 1, "max_abs": 0})
            self.assertEqual(read(output / "restore.json")["status"], "PASS")
