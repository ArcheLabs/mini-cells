"""Three tiny CPU sampler checks. Synthetic evidence is never historical evidence."""
import ast
import copy
import unittest

import numpy as np
from replay_sampling import SAMPLER, canonical, drawer, replay, sha


class SamplingChecks(unittest.TestCase):
    def test_draws_match_original_cpu_batches(self):
        import torch
        tree = ast.parse(SAMPLER.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'batches')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SAMPLER), 'exec'), namespace)
        c = {'training': {'effective_batch': 4, 'microbatch': 2, 'sequence_length': 4}}
        data = {'windows': {'train': {'starts': [0, 8, 16]}}, 'train': np.arange(32)}
        a, b = np.random.default_rng(1234), np.random.default_rng(1234)
        for _ in range(4):
            batches = namespace['batches'](data, c, a, 'cpu')
            indices = drawer()(c, data['windows']['train']['starts'], b)
            for (x, _), chosen in zip(batches, indices):
                expected = [data['windows']['train']['starts'][i] for i in chosen]
                self.assertEqual(x[:, 0].tolist(), expected)
            self.assertEqual(canonical(a.bit_generator.state), canonical(b.bit_generator.state))
        self.assertFalse(torch.cuda.is_initialized())

    def test_missing_checkpoint_never_certifies(self):
        c, w = fixture()
        _, report = replay(c, w, 1)
        self.assertTrue(all(i['status'] == 'NOT_CERTIFIED' for i in report['intervals']))

    def test_endpoint_version_and_cursor_required(self):
        c, w = fixture()
        _, predicted = replay(c, w, 1)
        boundaries = [dict(step=s, data_rng=state, data_cursor=s * 4,
                          checkpoint_bytes_verified=True, receipt_identity_verified=True,
                          full_state_fingerprint_verified=True, checkpoint_sha256='synthetic', receipt_sha256='synthetic')
                      for s, state in predicted['predicted_boundary_states'].items()]
        evidence = dict(seed=1, config_sha256=canonical(c), sampler_source_sha256=sha(SAMPLER),
                        historical_numpy_version=np.__version__, boundaries=boundaries)
        _, good = replay(c, w, 1, evidence)
        self.assertEqual(good['intervals'][0]['status'], 'CHECKPOINT_BOUNDED_SAMPLER_REPLAY_VERIFIED')
        changed = copy.deepcopy(evidence)
        changed['boundaries'][-1]['data_rng']['has_uint32'] ^= 1
        _, bad = replay(c, w, 1, changed)
        self.assertEqual(bad['intervals'][0]['status'], 'NOT_CERTIFIED')
        evidence['historical_numpy_version'] = None
        _, unknown = replay(c, w, 1, evidence)
        self.assertEqual(unknown['intervals'][0]['status'], 'NOT_CERTIFIED')


def fixture():
    c = dict(training=dict(data_seed_offset=7, steps=2, tokens_per_step=4),
             engine=dict(training=dict(sequence_length=4, effective_batch=1, microbatch=1)))
    w = {'train': dict(starts=[0, 8], groups=[0, 0], raw_story_ids=[0], story_offsets=[0, 32])}
    return c, w


if __name__ == '__main__':
    unittest.main()
