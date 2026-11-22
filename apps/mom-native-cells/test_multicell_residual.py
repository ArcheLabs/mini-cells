from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gdr_diagnostics import FIELDS
from residual_metrics import curve, group_bootstrap, metrics, rank_signal
from run_multicell_residual import PRIMARY, SEEDS, dataset, score_trace, verify_trace


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
