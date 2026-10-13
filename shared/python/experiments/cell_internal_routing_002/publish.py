"""Append small immutable seed evidence using the repository's existing auth tools."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from experiments.common.train import write_json
from experiments.cell_internal_routing_002.report import PROTOCOL, CONFIG, ROOT, digest, matrix_report

# Existing publisher modules use sibling imports; retain their supported import path.
sys.path.insert(0, str(ROOT / 'scripts/research'))
from publish_core_validation_007 import preflight_push, _push
from publish_experiment_results import run_git, validate_origin

ARTIFACT_BASE = 'research/projects/cellular-language-models/experiments/002-cell-internal-router/artifacts'
ALLOWED = {'.json', '.jsonl', '.csv', '.txt', '.log'}


def branch_for(source_commit):
    return 'results/routing-002-' + source_commit[:12]


def artifact_root(source_commit):
    return ROOT / ARTIFACT_BASE / source_commit


def verify_manifest(directory, source_commit, seed):
    directory = Path(directory)
    manifest = json.loads((directory / 'publication-manifest.json').read_text())
    if manifest['source_commit'] != source_commit or manifest['seed'] != seed:
        raise RuntimeError('Published seed identity mismatch')
    if manifest['protocol_sha256'] != digest(PROTOCOL) or manifest['source_config_sha256'] != digest(CONFIG):
        raise RuntimeError('Published protocol/configuration mismatch')
    for relative, expected in manifest['file_sha256'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != expected:
            raise RuntimeError('Published artifact hash mismatch: ' + relative)
    return manifest


def preflight(source_commit, seed, secret_name='GITHUB_TOKEN'):
    validate_origin(ROOT)
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('A full source commit SHA is required')
    changes = run_git(ROOT, 'diff', '--name-only', source_commit, 'HEAD').stdout.splitlines()
    if any(not name.startswith(ARTIFACT_BASE + '/') for name in changes):
        raise RuntimeError('Result branch changed experiment source; use the exact source commit')
    if run_git(ROOT, 'diff', '--name-only', 'HEAD').stdout.strip():
        raise RuntimeError('Tracked source modifications detected before GPU work')
    branch = branch_for(source_commit)
    preflight_push(branch, secret_name)
    existing = artifact_root(source_commit) / ('seed-' + str(seed))
    if existing.exists():
        verify_manifest(existing, source_commit, seed)
        return existing
    return None


def publish_seed(output, source_commit, seed, secret_name='GITHUB_TOKEN'):
    output = Path(output)
    root = artifact_root(source_commit)
    dest = root / ('seed-' + str(seed))
    if dest.exists():
        verify_manifest(dest, source_commit, seed)
        raise FileExistsError('Seed evidence is immutable; an observed seed cannot be overwritten')
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ('seed-' + str(seed) + '.tmp')
    temporary.mkdir(exist_ok=False)
    try:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in ALLOWED:
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError('Oversized small-evidence artifact: ' + path.name)
                relative = path.relative_to(output)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        shutil.copy2(PROTOCOL, temporary / 'protocol.json')
        shutil.copy2(CONFIG, temporary / 'source-config.json')
        manifest = {'source_commit': source_commit, 'seed': seed,
                    'protocol_sha256': digest(PROTOCOL), 'source_config_sha256': digest(CONFIG),
                    'file_sha256': {path.relative_to(temporary).as_posix(): digest(path)
                                    for path in sorted(temporary.rglob('*')) if path.is_file()},
                    'checkpoint_scope': 'checkpoints stay in Kaggle outputs; only their SHA256 is published'}
        write_json(temporary / 'publication-manifest.json', manifest)
        temporary.rename(dest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    seeds = [root / ('seed-' + str(value)) for value in json.loads(PROTOCOL.read_text())['seeds']]
    write_json(root / 'matrix-decision.json', matrix_report(seeds))
    run_git(ROOT, 'config', 'user.name', 'MiniCells Kaggle')
    run_git(ROOT, 'config', 'user.email', 'kaggle@minicells.local')
    run_git(ROOT, 'add', '--', root.relative_to(ROOT).as_posix())
    staged = run_git(ROOT, 'diff', '--cached', '--name-only').stdout.splitlines()
    if any(not name.startswith(root.relative_to(ROOT).as_posix() + '/') for name in staged):
        raise RuntimeError('Unrelated staged changes detected')
    run_git(ROOT, 'commit', '-m', 'research: record routing 002 development seed ' + str(seed))
    _push(branch_for(source_commit), secret_name)
    write_json(output / 'publication.json', {'status': 'published',
               'branch': branch_for(source_commit), 'path': dest.relative_to(ROOT).as_posix(),
               'publication_commit': run_git(ROOT, 'rev-parse', 'HEAD').stdout.strip()})
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-commit', default=os.environ.get('EXPERIMENT_COMMIT'))
    args = parser.parse_args()
    if not args.source_commit:
        raise ValueError('Pass --source-commit or set EXPERIMENT_COMMIT')
    publish_seed(args.output, args.source_commit, args.seed)


if __name__ == '__main__':
    main()
