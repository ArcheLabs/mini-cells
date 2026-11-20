import unittest
from collections import Counter

from protocol import donor_assignments, paired_summary


class ProtocolTest(unittest.TestCase):
    def test_balanced_donor_derangements(self):
        rows = donor_assignments()
        self.assertEqual(rows, donor_assignments())
        for key, donors in rows.items():
            self.assertEqual(len(set(donors)), 3)
            self.assertNotIn(key, donors)
        for index in range(3):
            self.assertEqual(Counter(row[index] for row in rows.values()),
                             Counter(range(64)))

    def test_paired_summary(self):
        result = paired_summary([[.02] * 64] * 3, 100)
        self.assertEqual(result["pairs"], 192)
        self.assertEqual(result["ci95"], [.02, .02])
        self.assertEqual(result["win_rate"], 1)
