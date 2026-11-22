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
