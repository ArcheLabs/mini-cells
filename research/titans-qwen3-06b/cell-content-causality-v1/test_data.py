import unittest

from data import SPLITS, episodes, manifest


class DataTests(unittest.TestCase):
    def test_counterfactual_changes_only_the_queried_value(self):
        for episode in episodes("test"):
            for query in episode["queries"]:
                facts = episode["facts"]
                if "counterfactual" not in query:
                    self.assertNotIn(query["entity"], dict(facts + episode["distractors"]))
                    continue
                changed = query["counterfactual_facts"]
                self.assertEqual([a for a, _ in facts], [a for a, _ in changed])
                self.assertEqual(sum(a != b for a, b in zip(facts, changed)), 1)
                self.assertEqual(dict(changed)[query["entity"]], query["counterfactual"])

    def test_train_balance_and_split_isolation(self):
        train = list(episodes("train"))
        self.assertEqual(len(train), 100)
        self.assertTrue(all(len(row["queries"]) == 1 for row in train))
        self.assertEqual(sum(row["queries"][0]["answer"] == "unknown" for row in train), 50)
        seen = set()
        for split in SPLITS:
            values = {item for row in episodes(split)
                      for pair in row["facts"] + row["distractors"] for item in pair}
            self.assertFalse(seen & values)
            seen.update(values)
            self.assertEqual(manifest(split), manifest(split))
