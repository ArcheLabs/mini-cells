import unittest

from decision import classify
from protocol import SEEDS


class DecisionTest(unittest.TestCase):
    def test_scientific_gates_and_smoke(self):
        for donor, expected in ((2.02, "PASS"), (2.005, "INCONCLUSIVE"), (1.98, "FAIL")):
            row = {"correct": 2., "empty": 2.1, "donor_mean": donor, "zero_read": 2.03}
            rows = {seed: [row] * 64 for seed in SEEDS}
            self.assertEqual(classify(rows, engineering_pass=True, replicates=100)["status"],
                             expected)
        self.assertEqual(classify({}, engineering_pass=True, smoke=True)["scientific_status"],
                         "NOT_EVALUATED")
        self.assertEqual(classify({}, engineering_pass=False)["status"], "RUN_FAILED")
