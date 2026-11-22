import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import cells
from cells import Cells
from compositional_recall import (
    ARMS, PAIRS, dataset, gold_for, preflight, restore_cells, run_group, summarize, write_cells,
)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


class CompositionalRecallTest(unittest.TestCase):
    def test_dataset_disjoint_and_counterbalanced(self):
        tokenizer = CharacterTokenizer()
        groups = dataset(tokenizer)
        self.assertEqual(groups, dataset(tokenizer))
        self.assertEqual(len(groups), 16)
        self.assertEqual(len({group["seed"] for group in groups}), 4)
        for key in ("projects", "keys", "codes"):
            self.assertEqual(len({value for group in groups for value in group[key]}), 32)
        for group in groups:
            for i in range(2):
                self.assertEqual([gold_for(group, i, pair) for pair in PAIRS],
                                 [group["codes"][j ^ i] for j in (0, 1, 1, 0)])
            for swap in (0, 1):
                a, b = (group["facts"][f"{cell}{swap}"] for cell in "AB")
                for code in group["codes"]:
                    self.assertNotIn(code.strip(), a)
                for project in group["projects"]:
                    self.assertNotIn(project, b)

    @patch("compositional_recall.prompt_for", side_effect=lambda project: "Question " + "x" * 30 + project + ":")
    def test_cells_roundtrip_and_candidate_isolation(self, _):
        tokenizer = CharacterTokenizer()
        group = dataset(tokenizer)[0]
        group["projects"] = ["X", "Y"]
        preflight(tokenizer, [group])
        candidate_calls = []

        def stream(tensors):
            return [tensors["0.4"].clone()] if tensors else []

        def step(tokens, caches, weights):
            if len(tokens) > 32:
                # Candidate queries receive independent cache instances.
                if caches[0]:
                    self.assertLess(float(caches[0][0].item()), 1000)
                    caches[0][0].add_(1000)
                candidate_calls.append((len(caches), weights))
            else:
                if not caches[0]:
                    caches[0].append(torch.zeros((1, 1, 1, 1)))
                caches[0][0].add_(sum(tokens))
            return torch.zeros((1, len(tokens), 256))

        backbone = SimpleNamespace(tokenizer=tokenizer, identity={"fixture": True},
            stream=stream, step=step, persistent=lambda cache: {"0.4": cache[0].clone()},
            model=SimpleNamespace(lm_head=lambda h: h))
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(cells, LAYERS=1, HEADS=1, WIDTH=1):
            original, imported = (Cells(Path(tmp)/name, backbone) for name in ("write", "restore"))
            metadata, state = write_cells(backbone, original, group)
            self.assertEqual(len({meta["cell_id"] for meta in metadata.values()}), 4)
            restored = restore_cells(original, imported, metadata)
            for label in state:
                # Keep fixture state values below the corruption sentinel.
                state[label]["0.4"].div_(100)
                restored[label]["0.4"].div_(100)
                torch.testing.assert_close(restored[label]["0.4"], state[label]["0.4"], rtol=0, atol=0)
            local_rows = run_group(backbone, group, state)
            fresh_rows = run_group(backbone, group, restored, restoring=True)
        self.assertEqual(len(candidate_calls), 2 * 2 * (len(ARMS) + len(PAIRS)))
        self.assertIn((2, [0.5, 0.5]), candidate_calls)
        summary, _ = summarize([group], local_rows, fresh_rows)
        self.assertEqual(summary["local_all_four"], 0)  # Ties never count as wins.
        self.assertEqual(summary["max_abs_restore_margin_delta"], 0)

    def test_summary_scores_by_intervention(self):
        group = dataset(CharacterTokenizer())[0]
        local, restored = [], []
        for i, project in enumerate(group["projects"]):
            scores = {}
            for arm in ARMS:
                gold = gold_for(group, i, arm) if arm in PAIRS else group["codes"][0]
                scores[arm] = [dict(answer=answer, total_logp=float(answer == gold) - 2,
                                    mean_logp=(float(answer == gold) - 2) / 8)
                               for answer in group["candidates"]]
            row = dict(group=group["group"], project_index=i, project=project, prompt=project)
            local.append({**row, "scores": scores})
            restored.append({**row, "scores": {arm: scores[arm] for arm in PAIRS}})
        summary, comparisons = summarize([group], local, restored)
        self.assertEqual(summary["conditions"]["no_cell"], dict(top1=4, total=8))
        self.assertEqual(summary["conditions"]["fresh_process"], dict(top1=8, total=8))
        self.assertEqual(summary["both_all_four"], 2)
        self.assertEqual(summary["fresh_complete_groups"], 1)
        self.assertTrue(all(row["margin_delta"] == 0 for row in comparisons))
