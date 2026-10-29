"""Minimal fake-store tests: exact bytes, metadata-only requests and failure isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recover_original_artifacts as recovery


class FakeBackend:
    def __init__(self, folder):
        self.folder, self.calls = folder, []

    def get(self, path, revision):
        self.calls.append((path, revision))
        return self.folder / path


def save(folder, name, data):
    p = folder / name
    p.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
    return recovery.digest(p)


def fixture(folder):
    tokenizer = save(folder, 'tokenizer-object', b'original tokenizer fixture')
    windows = save(folder, 'windows-object', {r: {'raw_story_ids': [i], 'starts': [0]}
                                            for i, r in enumerate(('train', 'calibration', 'admission', 'development'))})
    manifest = save(folder, 'manifest-object', dict(confirmation_materialized=False,
                    files={'tokenizer.json': tokenizer, 'windows.json': windows}))
    files = {'assets/' + n: dict(object=n.split('.')[0] + '-object', sha256=h)
             for n, h in [('manifest.json', manifest), ('tokenizer.json', tokenizer), ('windows.json', windows)]}
    bundle = save(folder, 'bundle', dict(identity=dict(tokenizer_sha256=tokenizer, assets_manifest_sha256=manifest), files=files))
    ref = dict(hf_repo='owner/private', hf_revision='a' * 40, bundle_path='bundle', bundle_sha256=bundle)
    return ref, manifest, tokenizer


class RecoveryTests(unittest.TestCase):
    def test_pinned_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ref, manifest, tok = fixture(folder)
            backend = FakeBackend(folder)
            with patch.object(recovery, 'TOKENIZER', tok):
                result = recovery.recover(backend, ref, manifest, folder / 'out')
            self.assertEqual(result['tokenizer_gate'], 'PASS_EXACT_ORIGINAL_BYTES')
            self.assertEqual(result['window_gate'], 'BLOCKED_PENDING_INDEPENDENT_AUDIT')
            self.assertEqual([p for p, _ in backend.calls], ['bundle', 'manifest-object', 'tokenizer-object', 'windows-object'])
            self.assertTrue(all(r == ref['hf_revision'] for _, r in backend.calls))
            with self.assertRaises(FileExistsError):
                recovery.recover(backend, ref, manifest, folder / 'out')

    def test_changed_bundle_stops_before_object_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ref, manifest, _ = fixture(folder)
            ref['bundle_sha256'] = '0' * 64
            backend = FakeBackend(folder)
            result = recovery.recover(backend, ref, manifest, folder / 'out')
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(result['files'], {})


if __name__ == '__main__':
    unittest.main()
