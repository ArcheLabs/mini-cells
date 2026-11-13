"""Recover the frozen Stage 3 checkpoint through the saved Output API."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name('stage6-source.json')


def recover(output):
    sys.path.insert(0, str(ROOT / 'research/projects/cellular-language-models/experiments'
                          / 'epclm-002-a0-persistent-state-recall/scripts'))
    from kaggle_output_recovery import _load_backend, _version_number

    api, fetch = _load_backend(ROOT)
    from kaggle_posthoc.kaggle_source import inventory

    proof = json.loads(SOURCE.read_text())
    name = proof['run_id'] + '/adapter.pt'
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
        if sha != proof['checkpoint_sha256']:
            raise ValueError('Saved checkpoint differs from the published Stage 3 SHA256')
        (output / 'recovery.json').write_text(json.dumps(dict(
            version_number=version, script_version_id=inv.get('metadata', {}).get('script_version_id'),
            checkpoint_sha256=sha, stage3_result_commit='aefaf887058675b1564856ed7eb8ef061e05fb75',
            run_id=proof['run_id']), indent=2))
        return path
    raise FileNotFoundError(f'{name} absent from the last 30 saved versions')


if __name__ == '__main__':
    from stage6 import main

    report = Path(sys.argv[1])
    main(report, recover(report.parent), SOURCE)
