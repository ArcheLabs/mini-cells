"""CPU integration for the independent M0 and M1 delivery lanes."""
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from minicells_native import MiniCells
from minicells_native.backend import Backbone
from minicells_native.errors import IncompatibleModelError, InvalidCellSelectionError
from minicells_native.recurrent import PREVIEW_PROFILE, no_padding_transitions
from common import write
import kaggle_next_stage as launcher
import m0_acceptance as m0
import m1_value_swap as m1


class Cache:
    def __init__(self):
        self.layers, self.position = [], 0

    @classmethod
    def from_legacy_cache(cls, layers, seen_tokens):
        cache = cls()
        cache.layers, cache.position = layers, seen_tokens
        return cache

    def get_seq_length(self):
        return self.position

    def to_legacy_cache(self):
        return self.layers

    def __len__(self):
        return len(self.layers)


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        return [len(word) % 241 + 2 for word in text.split()]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class TinyBackbone:
    identity = {"execution_profile": PREVIEW_PROFILE, "model_revision": "fixture"}
    device = "cpu"
    cache_type = Cache
    layer_count, bank_count = 1, 2
    recurrent_shape = (1, 1, 2, 2)
    native_bf16 = False
    tokenizer = Tokenizer()
    stream = Backbone.stream
    persistent = Backbone.persistent

    def __init__(self):
        self.model = SimpleNamespace(lm_head=lambda hidden: hidden)

    def step(self, ids, caches, weights, *, shared_observer=None):
        cache, = caches
        if not 1 <= len(ids) <= 64 or weights != [1.0]:
            raise AssertionError("Expected a bounded native stream")
        before = cache.layers[0]["recurrent_state"][0] if cache.layers else torch.zeros(self.recurrent_shape)
        bank = before + sum(ids)
        cache.layers = [dict(recurrent_state=[bank, bank.clone()], conv_state=([None, None],)*3,
                             attn_state=None, ffn_state=None)]
        cache.position += len(ids)
        if shared_observer is not None:
            shared_observer(bank)
        return torch.arange(256, dtype=torch.float32).repeat(1, len(ids), 1)


class PreviewTest(unittest.TestCase):
    def test_preview_lifecycle_and_public_query(self):
        with TemporaryDirectory() as temp:
            work = Path(temp)
            with MiniCells._from_backbone(TinyBackbone(), work / "a", selection_top_k=1) as mc:
                saved = m0.writer(mc, work)
                self.assertFalse(mc.query("plain", max_new_tokens=1).diagnostics["selection_experimental"])
                self.assertTrue(mc.query("plain", cells="auto", max_new_tokens=1).diagnostics["selection_experimental"])
                with self.assertRaises(InvalidCellSelectionError):
                    mc.query("plain", cells=[cell.id for cell in mc.cells.list()])
            write(work / "writer.json", saved)
            with MiniCells._from_backbone(TinyBackbone(), work / "b") as mc,                     patch.object(m0.os, "getpid", return_value=saved["pid"] + 1):
                imported = m0.reader(mc, work)
                self.assertEqual(imported["import_hash"], saved["bundle_sha256"])
            broken = work / "broken.cell"
            with zipfile.ZipFile(work / "a.cell") as source, zipfile.ZipFile(broken, "w") as target:
                manifest = json.loads(source.read("manifest.json"))
                manifest["backbone"]["execution_profile"] = "legacy"
                target.writestr("manifest.json", json.dumps(manifest))
                target.writestr("recurrent_states.safetensors", source.read("recurrent_states.safetensors"))
            with MiniCells._from_backbone(TinyBackbone(), work / "c") as mc:
                with self.assertRaises(IncompatibleModelError):
                    mc.cells.import_cell(broken)
                self.assertEqual(mc.cells.list(), [])

    def test_padding_policy_masks_only_padded_updates(self):
        g = torch.tensor([[[-1., -2.], [-3., -4.]]])
        beta = torch.ones_like(g)
        q = torch.ones(1, 2, 2, 4)
        masked = no_padding_transitions(dict(g=g, beta=beta, q=q), torch.tensor([False, True]))
        self.assertTrue(torch.equal(masked["g"][:, 1], g[:, 1]))
        self.assertTrue(torch.equal(masked["beta"][:, 1], beta[:, 1]))
        self.assertEqual(masked["g"][:, 0].count_nonzero(), 0)
        self.assertEqual(masked["beta"][:, 0].count_nonzero(), 0)
        self.assertIs(masked["q"], q)


class ValueSwapTest(unittest.TestCase):
    def test_stats_are_fact_level_and_deterministic(self):
        def row(i):
            arms = {name: dict(order_equal=True, margin=margin) for name, margin
                    in zip(m1.ARMS, (1 if i % 2 else -1, 3, -3, 2, -2), strict=True)}
            second = {name: dict(arm) for name, arm in arms.items()}
            if i == 0:
                second["cell_swapped"]["margin"] = 1
            return {"wordings": [{"arms": arms}, {"arms": second}]}
        rows = [row(i) for i in range(30)]
        summary = m1.summarize(rows)
        self.assertEqual(summary["flip_facts"], 29)
        self.assertEqual(summary["full_context_flip_facts"], 30)
        self.assertEqual(summary, m1.summarize(rows))
        self.assertTrue(all(value == "PASS" for value in summary["gates"].values()))

    def test_value_swap_writes_case_artifacts_with_real_cell_storage(self):
        items = [dict(index=i, texts={"A": "Original code AB", "B": "Alternate code CD"},
                      write_tokens={"A": [4, 6], "B": [5, 7]},
                      queries={f"query {j}": {"query_ids": [2, 3], "candidates": {"A": [4, 8], "B": [6, 9]}}
                               for j in range(2)})
                 for i in range(2)]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with (MiniCells._from_backbone(TinyBackbone(), root / "registry") as mc,
                  patch.object(m1, "make_dataset", return_value=items),
                  patch.object(m1, "DATASET_SHA256", m1.canonical_hash(items))):
                result = {}
                m1.run(mc, root, result)
            rows = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(len(row["wordings"]) == 2 for row in rows))
            self.assertTrue(all(len(set(row["cell_ids"].values())) == 2 for row in rows))
            self.assertEqual(result["status"], "INVALID")
            self.assertEqual(json.loads((root / "statistics.json").read_text())["facts"], 2)

    def test_scoring_reverses_independent_caches(self):
        backbone = TinyBackbone()
        query = dict(query_ids=[2, 3], candidates={"A": [4, 8], "B": [6, 9]})
        caches = []

        def factory():
            cache = Cache()
            caches.append(cache)
            return cache

        scores = m1.paired_scores(backbone, factory, query, "A")
        self.assertEqual(len({id(cache) for cache in caches}), 4)
        self.assertTrue(scores["order_equal"])
        self.assertGreater(scores["gold_nll"], 0)
        self.assertAlmostEqual(scores["margin"], scores["candidates"]["A"]["logp"] -
                               scores["candidates"]["B"]["logp"])


class LauncherTest(unittest.TestCase):
    def test_both_jobs_finish_before_publisher_switches_checkout(self):
        events = []

        def subprocess_run(*args, **kwargs):
            if args[:2] == ("git", "clone"):
                source = Path(args[-1]) / launcher.ROOT
                source.mkdir(parents=True)
                for _, script, _ in launcher.JOBS.values():
                    (source / script).touch()
            elif "-m" in args and "wheel" in args:
                (Path(args[-1]) / "minicells_native_sdk-0.1.0-test.whl").touch()
            elif "-u" in args:
                script, out = map(Path, args[-2:])
                self.assertTrue(script.exists())
                key = next(key for key, (_, filename, _) in launcher.JOBS.items() if filename == script.name)
                out.mkdir(parents=True, exist_ok=True)
                fmt = launcher.JOBS[key][2]
                write(out / "decision.json", dict(format=fmt, status="PASS", engineering_status="PASS"))
                events.append(key)

        def push(repo, bundle, label, branch, *_):
            events.append(branch.split("/")[-1].rsplit("-", 1)[0])
            shutil.rmtree(repo / launcher.ROOT, ignore_errors=True)

        publisher = SimpleNamespace(
            ExperimentSpec=lambda *args: args,
            load_github_token=lambda _: None,
            prepare_artifacts=lambda repo, label, spec, version_id: repo / spec[1],
            push_results=push,
        )
        with TemporaryDirectory() as temp, patch.dict(sys.modules, {
            "scripts.research.publish_experiment_results": publisher,
        }), patch.dict(os.environ, {"MINICELLS_NEXT_STAGE": "both"}), \
                patch.object(launcher, "OUTPUT", Path(temp)), \
                patch.object(launcher, "run", side_effect=subprocess_run), \
                patch.object(launcher, "full_suite"), patch.object(launcher, "t4_uuid", return_value="mock"), \
                patch.object(launcher, "version", return_value="mock"), \
                patch.object(launcher.subprocess, "check_output", return_value="candidate"):
            launcher.main()
        self.assertEqual(events, ["m0", "m1", launcher.JOBS["m0"][0], launcher.JOBS["m1"][0]])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
