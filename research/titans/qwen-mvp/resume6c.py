"""Recover the interrupted Stage 6C checkpoint through the saved Output API."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "qwen-mvp-stage6c-train-5255aa8c407c"


def recover(output, filename='adapter.pt'):
    sys.path.insert(0, str(ROOT / 'research/projects/cellular-language-models/experiments'
                          / 'epclm-002-a0-persistent-state-recall/scripts'))
    from kaggle_output_recovery import _load_backend, _version_number

    api, fetch = _load_backend(ROOT)
    from kaggle_posthoc.kaggle_source import inventory

    name = SOURCE + '/' + filename
    latest = _version_number(api.metadata())
    for version in range(latest, max(0, latest - 30), -1):
        print(f'Inspect saved Kaggle version {version}', flush=True)
        inv = inventory(api, version)
        candidates = [row['name'] for row in inv['files']
                      if row['name'] == name or row['name'].endswith('/' + name)]
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError('Ambiguous checkpoint path in saved version')
        path = output / candidates[0]
        fetch(api, inv, candidates, output)
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        (output / 'recovery.json').write_text(json.dumps(dict(
            version_number=version, checkpoint_sha256=sha, run_id=SOURCE,
            authentication='versioned_output; original run did not publish checkpoint hash'), indent=2))
        return path
    raise FileNotFoundError(f'{name} absent from the last 30 saved versions')


if __name__ == '__main__':
    report = Path(sys.argv[1]).resolve()
    checkpoint = recover(report.parent)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name('train6c.py')), str(report),
         '--resume', str(checkpoint), '--target-updates', '96'],
        check=True,
    )
