import unittest
import numpy as np
from benchmark import Attention, Config, Delta, FullKV, GatedDelta, Local, dataset


class BenchmarkTests(unittest.TestCase):
    def test_causality_and_fork(self):
        cfg = Config()
        k, v, _, _ = dataset(42, 8, "random", cfg)
        for cls in (Attention, Delta, FullKV, GatedDelta, Local):
            with self.subTest(cls=cls):
                model = cls(cfg)
                np.testing.assert_array_equal(model.step(k[0], k[0], v[0]), 0)
                snapshot = model.export_state()
                fork = cls(cfg)
                fork.import_state(snapshot)
                np.testing.assert_array_equal(fork.read(k), model.read(k))
                fork.write(k[1], v[1])
                for name, tensor in model.export_state().items():
                    np.testing.assert_array_equal(tensor, snapshot[name])

    def test_delta_equations(self):
        cfg = Config(beta=.7, alpha=.8)
        k, v, _, _ = dataset(43, 8, "random", cfg)
        for cls, alpha in ((Delta, 1.), (GatedDelta, .8)):
            model = cls(cfg)
            model.write(k[0], v[0])
            old = model.matrix.copy()
            model.write(k[1], v[1])
            outer = k[1,:, :,None]*k[1,:,None,:]
            expected = alpha * old @ (np.eye(cfg.dimension) - cfg.beta * outer)
            expected += cfg.beta * v[1,:,:,None]*k[1,:,None,:]
            np.testing.assert_allclose(model.matrix, expected, atol=1e-15)

    def test_generator(self):
        cfg = Config()
        for task in ("random", "correlated", "hard_negative", "overwrite", "nonlinear"):
            for a,b in zip(dataset(42,16,task,cfg), dataset(42,16,task,cfg)):
                np.testing.assert_array_equal(a,b)


class TitansParityTests(unittest.TestCase):
    def test_pinned_adapter(self):
        import os
        root = os.getenv("HAM_TITANS_UPSTREAM")
        if not root:
            self.skipTest("Set HAM_TITANS_UPSTREAM to validate against pinned Titans")
        import torch
        from titans_adapter import Titans, load_upstream
        cfg = Config()
        k, v, q, _ = dataset(42, 8, "random", cfg)
        adapter = Titans(cfg, load_upstream(root), 42)
        native = adapter.fresh()
        with torch.no_grad():
            for key, value in zip(k, v):
                address = adapter.address(key)
                reference, _ = native.memory.read_only(
                    (address, None, None), native.state) if native.state is not None else (None, None)
                expected = np.zeros_like(key) if reference is None else reference.numpy().reshape(key.shape)
                np.testing.assert_array_equal(adapter.step(key, key, value), expected)
                native.state = native.memory.store_only(
                    (address, address, native.tensor(value)), state=native.state)
        np.testing.assert_array_equal(adapter.read(q), native.read(q))
        restored = adapter.fresh()
        restored.import_state(adapter.export_state())
        np.testing.assert_array_equal(restored.read(q), adapter.read(q))
        before = adapter.read(q)
        restored.write(k[0], -v[0])
        np.testing.assert_array_equal(adapter.read(q), before)
