"""Exercise the real bridge entry point and SDK codecs with a tiny CPU backbone."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_bridge
from minicells_native.backend import Backbone


class Cache:
    def __init__(self):
        self.layers = []
        self.position = 0

    @classmethod
    def from_legacy_cache(cls, layers, seen_tokens):
        cache = cls()
        cache.layers = layers
        cache.position = seen_tokens
        return cache

    def get_seq_length(self):
        return self.position

    def to_legacy_cache(self):
        return self.layers

    def __len__(self):
        return len(self.layers)


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + int(sha256(word.encode()).hexdigest()[:5], 16) % 251 for word in text.split()]


class CpuBackbone:
    device = "cpu"
    identity = {"model_revision": run_bridge.REVISION_13B}
    cache_type = Cache
    layer_count = 1
    bank_count = 2
    recurrent_shape = (1, 1, 2, 2)
    native_bf16 = False
    tokenizer = Tokenizer()
    stream = Backbone.stream
    persistent = Backbone.persistent

    def __init__(self):
        self.model = SimpleNamespace(lm_head=lambda hidden: hidden,
                                     model=SimpleNamespace(layers=[None]))

    def step(self, ids, caches, weights):
        cache, = caches
        if not ids or len(ids) > run_bridge.CHUNK or weights != [1.0]:
            raise AssertionError("Expected one bounded stream")
        previous, convolution = 0.0, 0.0
        if cache.layers:
            layer = cache.layers[0]
            previous = float(layer["recurrent_state"][0].flatten()[0])
            saved = layer["conv_state"][0][0]
            if saved is not None:
                convolution = float(saved.flatten()[0])
        rec = torch.full(self.recurrent_shape, previous + sum(ids))
        conv = torch.full((1, 1, 4), convolution + sum(ids))
        cache.layers = [dict(recurrent_state=[rec, rec.clone()],
                             conv_state=([conv, conv.clone()], [None, None], [None, None]),
                             attn_state=None, ffn_state=None)]
        values = torch.linspace(-1, 1, 256) * (.1 + previous % 7 + convolution % 3 + cache.position / 32)
        cache.position += len(ids)
        return values.repeat(1, len(ids), 1)


class BridgeEntryPointTest(unittest.TestCase):
    def test_main_writes_all_cases_and_publishable_artifacts(self):
        from contextlib import ExitStack, nullcontext

        with TemporaryDirectory() as temp, ExitStack() as stack:
            out = Path(temp) / "output"
            fake = CpuBackbone()
            stack.enter_context(patch.object(sys, "argv", ["run_bridge.py", "--output", str(out)]))
            stack.enter_context(patch.dict(sys.modules, {
                "fla.modules": SimpleNamespace(convolution=None),
                "mom.layers": SimpleNamespace(mom_gated_deltanet=None),
            }))
            for name in ("interventions", "expert_padding_policy", "expert_delta_policy"):
                stack.enter_context(patch.object(run_bridge, name, side_effect=lambda *_a, **_k: nullcontext()))
            stack.enter_context(patch.object(run_bridge, "version", return_value="cpu-test"))
            stack.enter_context(patch.object(run_bridge, "environment", return_value={}))
            stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="Tesla T4"))
            stack.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)))
            stack.enter_context(patch.object(torch.cuda, "set_device"))
            stack.enter_context(patch.object(run_bridge.MiniCells, "open", side_effect=
                lambda _model, storage, **kw: run_bridge.MiniCells._from_backbone(
                    fake, storage, write_chunk_size=kw["write_chunk_size"])))

            run_bridge.main()

            receipt = json.loads((out / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual((receipt["format"], receipt["status"], receipt["engineering_status"]),
                             (run_bridge.FORMAT, "DEVELOPMENT_ONLY", "PASS"))
            rows = [json.loads(line) for line in (out / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
            result = json.loads((out / "bridge.json").read_text(encoding="utf-8"))
            self.assertEqual(rows, result["rows"])
            self.assertEqual([row["case"]["index"] for row in rows], list(range(9)))
            self.assertTrue(all(result["summary"]["controls"].values()))
            for row in rows:
                self.assertEqual(set(row["arms"]), set(run_bridge.ARMS))
                self.assertEqual(row["arms"]["C_recurrent_exact"]["query_start_cursors"],
                                 {"gold": 0, "distractor": 0})
                self.assertEqual(row["arms"]["B_live_exact"]["candidates"], row["arms"]["D_full_exact"]["candidates"])
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("cases.jsonl", manifest["artifacts_sha256"])
            for name, expected in manifest["artifacts_sha256"].items():
                self.assertEqual(sha256((out / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
