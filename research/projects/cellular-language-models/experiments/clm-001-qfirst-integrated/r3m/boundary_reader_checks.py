"""Exercise the real Store save/copy/verify path with tiny synthetic checkpoints, no model."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import recover_sampling_boundaries as reader


class FakeTrainer:
    def __init__(self, step):
        self.step = step

    def state(self):
        return dict(step=self.step, data_cursor=self.step * 512,
                    data_rng=dict(bit_generator='PCG64', state=dict(state=1, inc=3), has_uint32=0, uinteger=0))


class Backend:
    def __init__(self):
        self.paths = {}

    def get(self, name, revision):
        return self.paths[name]


class BoundaryReaderChecks(unittest.TestCase):
    def run_reader(self, corrupt=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = reader.read(reader.BASE.parent / 'r3i/config/frozen.json')['baseline']
            identity = {k: baseline[k] for k in ('experiment_id', 'source_sha', 'config_sha256', 'run_id')}
            c = reader.read(reader.BASE.parent / 'r3h/config/frozen.json')
            backend, files = Backend(), {}
            for seed in c['seeds']:
                prefix = 'jobs/seed-%s-arm-B-H/' % seed
                store = reader.Store(root / prefix, dict(identity, seed=seed, architecture='B', training_router='H'))
                for step in (0, 100, 200, 300, 400, 500):
                    row = store.save(FakeTrainer(step))
                    for path in (store.receipt_file(step), store.root / row['file']):
                        name = path.relative_to(root).as_posix()
                        key = 'objects/' + reader.digest(path)
                        backend.paths[key] = path
                        files[name] = dict(object=key, sha256=reader.digest(path), bytes=path.stat().st_size)
            if corrupt:
                name = next(k for k in backend.paths if str(backend.paths[k]).endswith('.pt'))
                replacement = root / 'wrong-bytes'
                replacement.write_bytes(b'changed')
                backend.paths[name] = replacement
            manifest = dict(identity=identity, files=files)
            with patch.object(reader, 'verified_manifest', return_value=copy.deepcopy(manifest)), redirect_stdout(io.StringIO()):
                result = reader.recover(backend, root / 'evidence.json')
            # Return JSON-roundtripped metadata only, never synthetic checkpoint bytes.
            return json.loads(json.dumps(result))

    def test_real_copy_then_store_verify(self):
        result = self.run_reader()
        self.assertEqual(len(result['failures']), 0)
        self.assertEqual(sum(len(j['boundaries']) for j in result['jobs']), 18)
        self.assertTrue(all(b['full_state_fingerprint_verified'] for j in result['jobs'] for b in j['boundaries']))
        self.assertEqual(result['optimizer_updates'], 0)

    def test_changed_checkpoint_is_not_verified(self):
        result = self.run_reader(corrupt=True)
        self.assertGreater(len(result['failures']), 0)
        self.assertLess(sum(len(j['boundaries']) for j in result['jobs']), 18)


if __name__ == '__main__':
    unittest.main()
