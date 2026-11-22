import tempfile
import unittest

import torch

from backend import fused_attention
from cells import Cells


class Branch(torch.nn.Module):
    def forward(self, hidden_states, past_key_values):
        past_key_values.append(hidden_states.clone())
        return hidden_states * len(past_key_values), None, past_key_values, None


class RuntimeTest(unittest.TestCase):
    def test_fusion_at_native_branch(self):
        from types import SimpleNamespace
        block = SimpleNamespace(attn=Branch())
        a, b = [], [torch.zeros(1)]
        x = torch.tensor([4.])
        with fused_attention([block], [a,b], [.25,.75]):
            received = block.attn(hidden_states=x, past_key_values=a)[0]
        torch.testing.assert_close(received, torch.tensor([7.]))
        self.assertEqual((len(a),len(b)), (1,2))
        self.assertNotIn("forward",block.attn.__dict__)

    def test_bundle_integrity_fork_and_import(self):
        from types import SimpleNamespace
        model=SimpleNamespace(identity={"revision":"test"})
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            store=Cells(a, model)
            meta=store.create("Cell")
            cloned=store.fork(meta["cell_id"])
            self.assertEqual(cloned["parent_cell_id"],meta["cell_id"])
            self.assertEqual(cloned["parent_revision"],0)
            self.assertNotEqual(cloned["cell_id"],meta["cell_id"])
            saved=store.path(meta["cell_id"]).read_bytes()
            restored=Cells(b,model)
            self.assertEqual(restored.import_bundle(saved),meta)
            with self.assertRaises(ValueError):
                restored.import_bundle(saved)
            with self.assertRaises(ValueError):
                store.path("../bad")

    def test_cancelled_write_preserves_revision(self):
        from types import SimpleNamespace
        model = SimpleNamespace(identity={}, tokenizer=SimpleNamespace(encode=lambda *a, **k: [1,2,3]))
        model.stream = lambda tensors: []
        model.step = lambda tokens, caches, weights: caches[0].extend(tokens)
        with tempfile.TemporaryDirectory() as directory:
            store = Cells(directory, model)
            cell = store.create("Cancellation")
            completed = []
            self.assertIsNone(store.write(cell["cell_id"], "text", 1,
                lambda done, total: completed.append(done), lambda: bool(completed)))
            self.assertEqual(completed, [1])
            self.assertEqual(store.read(cell["cell_id"])[0], cell)

    def test_resume_from_saved_stream(self):
        from types import SimpleNamespace

        class Cache:
            def __init__(self, layers=None, position=0):
                self.layers = layers or [dict(recurrent_state=[torch.zeros(1)],
                    conv_state=[[torch.zeros(1), None]], attn_state=None, ffn_state=None)]
                self.position = position

            def to_legacy_cache(self):
                return tuple(self.layers)

            def get_seq_length(self):
                return self.position

            @classmethod
            def from_legacy_cache(cls, layers, seen_tokens):
                return cls(layers, seen_tokens)

        model = SimpleNamespace(identity={}, cache_type=Cache, device="cpu",
                                tokenizer=SimpleNamespace(encode=lambda *a, **k: [1,2,3]))
        model.stream = lambda tensors: Cache()
        tokens_seen = []
        def step(tokens, caches, weights):
            cache = caches[0]
            tokens_seen.extend(tokens)
            cache.position += len(tokens)
            cache.layers[0]["conv_state"][0][0].add_(sum(tokens))
            cache.layers[0]["recurrent_state"][0].add_(sum(tokens))
        model.step = step
        def persistent(cache):
            self.assertEqual(cache.position, 3)
            self.assertEqual(cache.layers[0]["conv_state"][0][0].item(), 6)
            self.assertEqual(cache.layers[0]["recurrent_state"][0].item(), 6)
            return {f"{i}.4": torch.zeros(1,4,256,256) for i in range(24)}
        model.persistent = persistent
        with tempfile.TemporaryDirectory() as directory:
            store = Cells(directory, model)
            cell = store.create("Resume")
            self.assertIsNone(store.write(cell["cell_id"], "text", 1,
                lambda *_: None, lambda: bool(tokens_seen), resume=True))
            self.assertEqual(store.read(cell["cell_id"])[0]["revision"],0)
            reopened = Cells(directory, model)
            updated = reopened.write(cell["cell_id"], "text", 1, lambda *_: None,
                                     lambda: False, resume=True)
            self.assertEqual(tokens_seen, [1,2,3])
            self.assertEqual(updated["revision"],1)
            self.assertFalse(store.path(cell["cell_id"]).with_suffix(".stream").exists())

    def test_web_cell_lifecycle(self):
        from types import SimpleNamespace
        from fastapi.testclient import TestClient
        from server import app_for
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(app_for(Cells(directory, SimpleNamespace(identity={})))) as client:
                self.assertEqual(client.get("/").status_code,200)
                created = client.post("/api/cells",json={"name":"Web"})
                self.assertEqual(created.status_code,200)
                cid = created.json()["cell_id"]
                bundle = client.get(f"/api/cells/{cid}/export").content
                self.assertEqual(client.delete(f"/api/cells/{cid}").status_code,200)
                self.assertEqual(client.post("/api/cells/import",content=bundle).status_code,200)
                self.assertEqual(client.get(f"/api/cells/{cid}").json()["name"],"Web")
