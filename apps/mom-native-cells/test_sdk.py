from dataclasses import asdict
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from minicells_native import cells as native
from minicells_native.errors import (
    CellAlreadyExistsError, CellArchiveCorruptedError, CellLimitError, CellNotFoundError,
    CellWriteError, IncompatibleModelError, InvalidCellSelectionError, ModelLoadError,
    RuntimeClosedError, StorageBusyError, UnsupportedChunkSizeError,
)
from minicells_native.sdk import MiniCells


class FakeCache:
    def __init__(self, value, position=0):
        self.r = value
        self.position = position

    def to_legacy_cache(self):
        return (dict(recurrent_state=[self.r], conv_state=[[None, None]],
                     attn_state=None, ffn_state=None),)

    def get_seq_length(self):
        return self.position

    @classmethod
    def from_legacy_cache(cls, layers, seen_tokens):
        return cls(layers[0]["recurrent_state"][0], seen_tokens)


class Fixture:
    identity = {"model": "fixture"}
    device = "cpu"
    native_bf16 = False
    parameters = 0
    cache_type = FakeCache

    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1 + ord(x) % 9 for x in text],
                                         decode=lambda tokens, **kwargs: "".join(map(str, tokens)),
                                         eos_token_id=0)
        self.model = SimpleNamespace(lm_head=lambda h: h)

    def stream(self, state):
        return FakeCache(state.get("0.4", torch.zeros(1, 1, 2, 2)).clone())

    def step(self, tokens, caches, weights):
        self.calls.append(("native", list(tokens), [float(c.r.flatten()[0]) for c in caches]))
        for cache in caches:
            cache.r.add_(sum(tokens))
            cache.position += len(tokens)
        return torch.tensor([[[0., 1.]]]).expand(1, len(tokens), 2)

    def persistent(self, cache):
        return {"0.4": cache.r.clone()}


class Reader:
    def __init__(self, backbone):
        self.backbone = backbone

    def start(self, states):
        return [self.backbone.stream(state) for state in states]

    def step(self, tokens, caches, config, gate=None, diagnostics=None):
        self.backbone.calls.append(("score" if diagnostics else "fuse", list(tokens),
                                    [float(c.r.flatten()[0]) for c in caches]))
        if diagnostics:
            diagnostics.shared_norm(caches[0].r.reshape(1, 1, 2, 2))
        for cache in caches:
            cache.r.add_(100)
        return torch.tensor([[[0., 1.]]]).expand(1, len(tokens), 2)


def fixture_patch():
    return patch.multiple(native, LAYERS=1, HEADS=1, WIDTH=2)


class SDKTest(unittest.TestCase):
    def setUp(self):
        dims = fixture_patch()
        dims.start()
        self.addCleanup(dims.stop)
        reader = patch("minicells_native.query.FusionReader", Reader)
        reader.start()
        self.addCleanup(reader.stop)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Fixture()
        self.mc = MiniCells._from_backbone(self.model, self.root / "store")
        self.addCleanup(self.mc.close)

    def test_lifecycle_isolation_and_resume(self):
        a = self.mc.cells.create("a")
        b = self.mc.cells.create("a")
        self.assertNotEqual(a.id, b.id)
        progress = []
        self.assertIsNone(self.mc.cells.write(a.id, "abcdefgh", chunk_size=2,
                          progress=lambda n, total: progress.append(n),
                          cancelled=lambda: bool(progress)))
        self.assertEqual(self.mc.cells.get(a.id).revision, 0)
        finished = self.mc.cells.append(a.id, "abcdefgh", chunk_size=2)
        self.assertEqual(finished.tokens_written, 8)
        self.assertEqual(self.mc.cells.get(b.id), b)
        self.assertEqual([r["revision"] for r in self.mc.cells.revisions(a.id)], [0, 1])
        branch = self.mc.cells.fork(a.id, "branch")
        self.assertEqual(branch.parent_cell_id, a.id)
        before = self.mc.cells._store.path(a.id).read_bytes()
        self.mc.cells.write(branch.id, "child")
        self.assertEqual(self.mc.cells._store.path(a.id).read_bytes(), before)
        self.mc.cells.delete(a.id)
        self.assertEqual(self.mc.cells.get(branch.id).parent_cell_id, a.id)
        with self.assertRaises(CellNotFoundError):
            self.mc.cells.get(a.id)

    def test_mode_scoring_and_fresh_generation(self):
        self.assertEqual(self.mc.query("hello", max_new_tokens=1).fallback_reason, "no_registered_cells")
        a = self.mc.cells.create("a").id
        b = self.mc.cells.create("b").id
        self.assertEqual(self.mc.query("hello", max_new_tokens=1).fallback_reason, "no_match")
        self.mc.cells.write(a, "a")
        self.mc.cells.write(b, "bb")
        self.model.calls.clear()
        hashes = [self.mc.cells._get(cell)["tensor_sha256"] for cell in (a, b)]
        auto = self.mc.query("hello", max_new_tokens=1)
        self.assertEqual(auto.candidate_cell_ids, sorted((a, b)))
        self.assertEqual(auto.selected_cell_ids, [b, a])
        self.assertEqual(auto.text, "1")
        scoring, other, final = self.model.calls
        self.assertEqual(scoring[0], "score")
        self.assertEqual(other[1], scoring[1])
        self.assertEqual(sorted(final[2]), sorted(scoring[2] + other[2]))
        self.assertEqual(hashes, [self.mc.cells._get(cell)["tensor_sha256"] for cell in (a, b)])
        self.model.calls.clear()
        self.assertEqual(self.mc.query("manual", [a], max_new_tokens=1).selected_cell_ids, [a])
        self.assertEqual([call[0] for call in self.model.calls], ["fuse"])
        self.model.calls.clear()
        self.assertEqual(self.mc.query("q", [], max_new_tokens=1).fallback_reason, "explicit_no_cells")
        self.assertEqual([call[0] for call in self.model.calls], ["native"])
        with self.assertRaises(InvalidCellSelectionError):
            self.mc.query("q", None)
        with self.assertRaises(CellNotFoundError):
            self.mc.query("q", ["bogus"])

    def test_limit_threshold_chunk_and_lock(self):
        with self.assertRaises(StorageBusyError):
            MiniCells._from_backbone(Fixture(), self.root / "store")
        with MiniCells._from_backbone(Fixture(), self.root / "limited", max_cells=1,
                                     selection_threshold=10000) as limited:
            a = limited.cells.create("a")
            limited.cells.write(a.id, "hello")
            self.assertEqual(limited.query("q", max_new_tokens=1).fallback_reason, "no_match")
            limited.cells.create("b")
            with self.assertRaises(CellLimitError):
                limited.query("q")
            with self.assertRaises(UnsupportedChunkSizeError):
                limited.cells.append(a.id, "hello", chunk_size=65)
            self.assertEqual(limited.cells.get(a.id).revision, 1)
        with self.assertRaises(RuntimeClosedError):
            limited.cells.list()
        self.model.calls.clear()
        self.mc.query("x" * 150, [], max_new_tokens=1)
        self.assertEqual([len(call[1]) for call in self.model.calls][:2], [64, 64])

    def test_archives_conflicts_and_cross_process(self):
        a = self.mc.cells.create("a")
        self.mc.cells.write(a.id, "abc")
        b = self.mc.cells.fork(a.id, "b")
        single = self.mc.cells.export(a.id, self.root / "one.zip")
        collection = self.mc.cells.export_collection(self.root / "all.zip", cell_ids=[a.id, b.id])
        with MiniCells._from_backbone(Fixture(), self.root / "other") as imported:
            self.assertEqual(imported.cells.import_cell(single).revision, 1)
            with self.assertRaises(CellAlreadyExistsError):
                imported.cells.import_collection(collection)
            self.assertEqual([c.id for c in imported.cells.list()], [a.id])
            corrupt = self.root / "corrupt.zip"
            with zipfile.ZipFile(collection) as source, zipfile.ZipFile(corrupt, "w") as out:
                for name in source.namelist():
                    out.writestr(name, b"broken" if name == f"cells/{b.id}.zip" else source.read(name))
            with self.assertRaises(CellArchiveCorruptedError):
                imported.cells.import_collection(corrupt)
            self.assertEqual([c.id for c in imported.cells.list()], [a.id])
        env = {**os.environ, "OMP_NUM_THREADS": "2"}
        subprocess.run([sys.executable, __file__, str(self.root / "child"), str(collection),
                        a.id, b.id], check=True, timeout=45, env=env)

    def test_interrupted_collection_rolls_back_on_reopen(self):
        a = self.mc.cells.create("a")
        b = self.mc.cells.create("b")
        third = self.mc.cells.create("third")
        source = self.mc.cells._store
        destination = self.root / "interrupted"
        destination.mkdir()
        shutil.copyfile(source.path(a.id), destination / source.path(a.id).name)
        shutil.copyfile(source.path(third.id), destination / source.path(third.id).name)
        (destination / ".minicells-import.json").write_text(json.dumps([a.id, b.id]))
        with MiniCells._from_backbone(Fixture(), destination) as recovered:
            self.assertEqual([cell.id for cell in recovered.cells.list()], [third.id])

    def test_web_defaults_to_sdk_auto(self):
        from fastapi.testclient import TestClient
        from server import app_for

        a = self.mc.cells.write(self.mc.cells.create("a").id, "aaa")
        with TestClient(app_for(self.mc)) as web:
            auto = web.post("/api/query", json={"question": "q", "max_new_tokens": 1})
            self.assertEqual(auto.status_code, 200)
            self.assertEqual(auto.json()["selected_cell_ids"], [a.id])
            self.assertEqual(auto.json()["mode"], "auto")
            without = web.post("/api/query", json={"question": "q", "cells": [], "max_new_tokens": 1})
            self.assertEqual(without.json()["fallback_reason"], "explicit_no_cells")

    def test_failed_write_and_model_mismatch(self):
        a = self.mc.cells.create("a")
        path = self.mc.cells._store.path(a.id)
        before = path.read_bytes()
        with patch.object(self.model, "step", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(CellWriteError):
                self.mc.cells.write(a.id, "text")
        self.assertEqual(path.read_bytes(), before)
        file = self.mc.cells.export(a.id, self.root / "one.zip")
        model = Fixture()
        model.identity = {"model": "different"}
        with MiniCells._from_backbone(model, self.root / "incompatible") as target:
            with self.assertRaises(IncompatibleModelError):
                target.cells.import_cell(file)
        with self.assertRaises(ModelLoadError):
            MiniCells.open(storage=self.root / "cpu", device="cpu")


def child(storage, file, cell, fork):
    with fixture_patch(), patch("minicells_native.query.FusionReader", Reader):
        with MiniCells._from_backbone(Fixture(), storage) as mc:
            mc.cells.import_collection(file)
            original = asdict(mc.cells.get(fork))
            result = mc.query("hello", [cell], max_new_tokens=1)
            assert result.text == "1"
            assert mc.cells.append(cell, "more").revision == 2
            assert mc.query("hello", [cell], max_new_tokens=1).text == result.text
            assert asdict(mc.cells.get(fork)) == original
        with MiniCells._from_backbone(Fixture(), storage) as mc:
            assert mc.cells.get(cell).revision == 2
            assert len(mc.cells.revisions(cell)) == 3


if __name__ == "__main__":
    child(*sys.argv[1:])
