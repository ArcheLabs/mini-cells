from dataclasses import replace
import unittest
import numpy as np
from memory import Config, DeltaMemory, Parallel


class DeltaTests(unittest.TestCase):
    def test_single_overwrite_orthogonal_and_similar(self):
        cfg = Config(heads=1, key_dim=2, value_dim=3)
        m = DeltaMemory(cfg)
        a,b = np.eye(2)[:,None,:]
        v,w = np.eye(3)[:2,None,:]
        np.testing.assert_array_equal(m.step(a,a,v), 0)
        np.testing.assert_array_equal(m.read(a), v)
        m.write(b,w)
        np.testing.assert_array_equal(m.read(b), w)
        m.write(a,-v)
        np.testing.assert_array_equal(m.read(a), -v)
        np.testing.assert_array_equal(m.read(b), w)
        np.testing.assert_allclose(m.read(a+.01*b), (-v+.01*w)/np.sqrt(1.0001))

    def test_reference_and_fork(self):
        cfg = Config(beta=.7, alpha=.92)
        rng = np.random.default_rng(42)
        for alpha in (1., cfg.alpha):
            m = DeltaMemory(replace(cfg, alpha=alpha))
            for _ in range(3):
                k,v = rng.normal(size=(2,cfg.heads,cfg.key_dim))
                k /= np.linalg.norm(k,axis=-1,keepdims=True)
                ref = alpha*m.matrix @ (np.eye(cfg.key_dim)-cfg.beta*k[:,:,None]*k[:,None,:])
                ref += cfg.beta*v[:,:,None]*k[:,None,:]
                m.write(k,v)
                np.testing.assert_allclose(m.matrix, ref, atol=1e-14)
            snapshot = m.export_state()
            fork = m.fresh()
            fork.import_state(snapshot)
            np.testing.assert_array_equal(fork.read(k), m.read(k))
            fork.write(k,-v)
            np.testing.assert_array_equal(m.matrix,snapshot)
