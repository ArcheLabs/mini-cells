import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

import replay


class ReplayTests(unittest.TestCase):
    def test_checkpoint_integrity_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay.save(root / 'decision.json', {'source_commit': replay.SOURCE_COMMIT})
            files = []
            for seed in replay.SEEDS:
                folder = root / f'seed_{seed}'
                folder.mkdir()
                hashes = {}
                for step in replay.STEPS:
                    path = folder / f'step_{step:03}.pt'
                    path.write_bytes(f'{seed}:{step}'.encode())
                    hashes[path.name] = replay.sha(path)
                    files.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                                      sha256=hashes[path.name]))
                replay.save(folder / 'checkpoint_hashes.json', hashes)
            replay.save(root / 'metadata.json', {'files': files})
            self.assertEqual(len(replay.verify(root)), 9)
            path.write_bytes(b'wrong!')
            with self.assertRaisesRegex(ValueError, 'Published artifact mismatch'):
                replay.verify(root)

    def test_restore_names_shapes_and_finiteness(self):
        wrapper = nn.Linear(2, 2, bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.pt'
            expected = torch.ones_like(wrapper.weight)
            for name, value, message in [('wrong', expected, 'names'),
                    ('weight', torch.zeros(3, 2), 'tensor mismatch'),
                    ('weight', expected * float('nan'), 'Nonfinite')]:
                torch.save(dict(step=0, parameters={name: value}), path)
                with self.assertRaisesRegex(ValueError, message):
                    replay.restore(wrapper, path, 0)
            torch.save(dict(step=0, parameters={'weight': expected}), path)
            result = replay.restore(wrapper, path, 0)
            self.assertEqual(result['weight']['sha256'], hashlib.sha256(expected.numpy()).hexdigest())
            torch.testing.assert_close(wrapper.weight, expected)

    def test_measurements(self):
        self.assertAlmostEqual(replay.compare(torch.ones(2), torch.ones(2)*2)['relative_rms'], 1)
        self.assertAlmostEqual(replay.compare(torch.ones(2), torch.ones(2)*2)['cosine'], 1, places=6)
        self.assertEqual(replay.tensor_stats((torch.eye(2),))[0]['head_norms'], [1, 1])
        self.assertEqual(replay.describe(torch.tensor([0., 1.]))['median'], .5)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            replay.describe(torch.tensor([float('inf')]))

    def test_read_capture_is_removed_on_error(self):
        class Memory(nn.Module):
            def __init__(self):
                super().__init__()
                self.history_mix_proj = nn.Identity()
                self.history_mix = nn.Parameter(torch.zeros(2048))

            def read_only(self, tensor):
                return tensor * 2, None

        m = Memory()
        x = torch.ones(1, 1, 2048)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with replay.capture(m, write=False) as obs:
                replay.same_tree(m.read_only(x)[0], x * 2)
                replay.same_tree(obs['read'], x * 2)
                replay.same_tree(obs.get('gate'), None)
                m.history_mix_proj(x)
                self.assertEqual(obs['gate'].shape, x.shape)
                raise RuntimeError('fixture')
        self.assertNotIn('read_only', vars(m))
        self.assertEqual(len(m.history_mix_proj._forward_hooks), 0)
        original = (1, x)
        copy = replay.snapshot(original)
        replay.same_tree(original, copy)
        copy[1].zero_()
        self.assertEqual(x.sum().item(), 2048)

class ProtocolTests(unittest.TestCase):
    def test_frozen_pairing_and_corpus_rebuild(self):
        pairs = replay.donor_assignments(64, 3)
        self.assertEqual(len(pairs), 64)
        self.assertTrue(all(len(set(v)) == 3 and k not in v for k, v in pairs.items()))
        from unittest.mock import patch
        rows = [f'Document {i}\nExample' for i in range(5)]
        with patch('corpus.articles', side_effect=lambda x: x):
            one = replay.select(rows[:4], rows[4:], lambda t: [int(t[9])] * 2,
                                tokens=2, train_size=1, holdout_size=2, compat_size=1)
            two = replay.select(rows[:4], rows[4:], lambda t: [int(t[9])] * 2,
                                tokens=2, train_size=1, holdout_size=2, compat_size=1)
        self.assertEqual(one, two)
        hashes = [d['document_sha256'] for d in one[1]['splits']['primary_holdout']['documents']]
        self.assertEqual(hashes, sorted(hashes))


if __name__ == '__main__':
    unittest.main()
