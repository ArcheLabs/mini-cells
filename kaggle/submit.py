"""Prepare a bundle for the existing Kaggle kernel and optional API submission.

The existing kernel id and type are retained. Kaggle reads GITHUB_TOKEN at
runtime for the existing GitHub publication path. HF_TOKEN is attempted when
available but is optional for the public, pinned TinyStories dataset. Tokens
are never serialized in the bundle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(commit, seed, destination):
    if len(commit) != 40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('A full source SHA is required')
    if seed not in (42, 123, 456):
        raise ValueError('Choose registered development seed 42, 123 or 456')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = ROOT / 'experiments/003-hard-cell-transition/notebooks/experiment.ipynb'
    notebook = json.loads(path.read_text())
    script = []
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        text = text.replace('__EXPERIMENT_COMMIT__', commit).replace('__EXPERIMENT_SEED__', str(seed))
        cell['source'] = text.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            script.append(text)
        else:
            script.append('\n'.join('# ' + line for line in text.splitlines()))
    (destination / 'experiment.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    (destination / 'experiment.py').write_text('\n\n'.join(script) + '\n')
    metadata = json.loads((ROOT / 'kaggle/kernel-metadata.json').read_text())
    metadata.update(id='bingjiangli/minicells', title='MiniCells', code_file='experiment.py', kernel_type='script')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (destination / 'preparation.json').write_text(json.dumps({
        'source_commit': commit, 'seed': seed, 'kernel_id': metadata['id'],
        'notebook_sha256': hashlib.sha256((destination / 'experiment.ipynb').read_bytes()).hexdigest(),
        'launcher_sha256': hashlib.sha256((destination / 'experiment.py').read_bytes()).hexdigest(),
        'server_timeout_seconds': 3600, 'application_timeout_seconds': 3300,
        'required_runtime_secret_labels': ['GITHUB_TOKEN'],
        'optional_runtime_secret_labels': ['HF_TOKEN'],
        'secret_scope': 'GITHUB_TOKEN enables the existing Kaggle-to-GitHub publisher; '
                        'HF_TOKEN is optional for public pinned TinyStories'}, indent=2) + '\n')
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--seed', type=int, required=True, choices=[42, 123, 456])
    parser.add_argument('--output', required=True)
    parser.add_argument('--push', action='store_true', help='Explicitly update and run the original Kaggle kernel')
    args = parser.parse_args()
    directory = prepare(args.commit, args.seed, args.output)
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        result = api.kernels_push(str(directory), timeout=3600)
        # Server-returned identity/version, never inferred from title or leaderboard.
        if result is None or result.error:
            raise RuntimeError('Kaggle push failed: ' + (result.error if result else 'empty response'))
        print(json.dumps({'url': result.url, 'version': result.versionNumber}, default=str))
    else:
        print('Prepared exact-commit notebook and compatible script for bingjiangli/minicells.')


if __name__ == '__main__':
    main()
