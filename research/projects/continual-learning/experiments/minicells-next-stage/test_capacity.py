"""CPU regression checks for the frozen M1-002 execution design."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from minicells_native import MiniCells

import m1_capacity as capacity
from test_next_stage import TinyBackbone


class WordTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [self.vocab.setdefault(word, len(self.vocab)) for word in re.findall(r"[A-Za-z]+|[^\s]", text)]


class CapacityTest(unittest.TestCase):
    def test_all_arms_run_with_real_storage_and_publish(self):
        backbone = TinyBackbone()
        backbone.tokenizer = WordTokenizer()
        with patch.object(capacity, "EPISODES", 1):
            dataset = capacity.build_dataset(backbone.tokenizer)
            self.assertEqual(capacity.build_dataset(backbone.tokenizer), dataset)
            for orders in dataset[0]["orders"].values():
                for sample in orders.values():
                    self.assertEqual(len(sample["tokens"]), len(sample["neutral_tokens"]))
                    self.assertEqual(sample["tokens"], sum(sample["batch_ids"], []))
                    self.assertNotEqual(sample["schedules"]["factual"], sample["schedules"]["split32"])

            with TemporaryDirectory() as temp:
                out = Path(temp) / "run"
                opened = MiniCells._from_backbone(backbone, Path(temp) / "registry")
                with (patch.object(MiniCells, "open_preview", return_value=opened),
                      patch.object(capacity, "environment", return_value={"gpu": "fixture"}),
                      patch.object(capacity, "DATASET_SHA256", capacity.canonical_hash(dataset)),
                      patch.object(sys, "argv", ["m1_capacity.py", str(out)])):
                    capacity.main()
                self.assertEqual(list((Path(temp) / "registry").rglob("*.cell")), [])
                decision = json.loads((out / "decision.json").read_text())
                self.assertEqual(decision["engineering_status"], "PASS")
                self.assertEqual(decision["format"], capacity.FORMAT)
                stats = json.loads((out / "statistics.json").read_text())
                self.assertEqual(set(stats["levels"]), {"3", "8", "16"})
                row = json.loads((out / "cases.jsonl").read_text())
                for orders in row["levels"].values():
                    for arm in orders.values():
                        self.assertEqual(arm["factual"]["wordings"][0]["start_cursor"], 0)
                        self.assertEqual(arm["factual"]["wordings"][1]["start_cursor"], 0)
                        self.assertEqual(len(arm["append"]["bundle_sha256"]), 64)
                        self.assertEqual(arm["append"]["wordings"],
                                         arm["continuous_append_schedule"]["wordings"])
                self.assertIn("| 16 |", (out / "report.md").read_text())
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertEqual(set(manifest["artifacts_sha256"]), {
                    "cases.jsonl", "dataset.json", "decision.json", "report.md", "result.json", "statistics.json"})
                for name, digest in manifest["artifacts_sha256"].items():
                    self.assertEqual(sha256((out / name).read_bytes()).hexdigest(), digest)

    def test_gates_use_both_orders_without_counting_wordings_as_replicates(self):
        hit = dict(recalled=True, mean_margin=2, mean_gold_nll=3)
        miss = dict(recalled=False, mean_margin=-2, mean_gold_nll=5)
        row = dict(baseline=dict(no_memory=miss, full_context=hit, target_only=hit), levels={})
        for level in ("3", "8", "16"):
            arms = dict(factual=hit, neutral=hit, split32=hit, append=hit, continuous_append_schedule=hit)
            row["levels"][level] = dict(target_first=arms, target_last=dict(arms, factual=miss))
        stats = capacity.summarize([row])
        self.assertEqual(stats["gates"]["robust_capacity"], "FAIL")
        self.assertEqual(stats["levels"]["16"]["robust_recalled"], 0)
        self.assertEqual(stats["levels"]["16"]["arms"]["factual"]["accuracy"], .5)
        self.assertEqual(stats["levels"]["16"]["contrasts"]["extra_facts_at_equal_length"]["recall_losses"], 1)


if __name__ == "__main__":
    unittest.main()
