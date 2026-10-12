"""Immutable small evidence publication using existing repository authentication."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from experiments.common.train import write_json
from minicells.utils.research_report import ROOT, CONFIG, PROTOCOL

sys.path.insert(0, str(ROOT / 'scripts/research'))
from publish_core_validation_007 import preflight_push, _push
from publish_experiment_results import run_git, validate_origin

BASE = 'artifacts/experiments/003-hard-cell-transition'
ALLOWED = {'.json', '.jsonl', '.csv', '.txt', '.log', '.md', '.png'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def branch(source):
    return 'results/routing-003-' + source[:12]


def verify(directory, source, seed):
    manifest = json.loads((directory / 'publication-manifest.json').read_text())
    if (manifest['source_commit'], manifest['seed'], manifest['config_sha256'], manifest['protocol_sha256']) != (
        source, seed, digest(CONFIG), digest(PROTOCOL)):
        raise RuntimeError('Existing seed identity differs from frozen source/configuration')
    for name, expected in manifest['file_sha256'].items():
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents or digest(path) != expected:
            raise RuntimeError('Immutable artifact hash mismatch: ' + name)


def preflight(source, seed):
    if len(source) != 40 or any(char not in '0123456789abcdef' for char in source):
        raise ValueError('EXPERIMENT_COMMIT must be a full source SHA')
    validate_origin(ROOT)
    changed = run_git(ROOT, 'diff', '--name-only', source, 'HEAD').stdout.splitlines()
    if any(not name.startswith(BASE + '/' + source + '/') for name in changed):
        raise RuntimeError('Results branch changed source outside its own evidence directory')
    if run_git(ROOT, 'diff', '--name-only', 'HEAD').stdout.strip():
        raise RuntimeError('Modified tracked source before training')
    preflight_push(branch(source), 'GITHUB_TOKEN')
    directory = ROOT / BASE / source / ('seed-' + str(seed))
    if directory.exists():
        verify(directory, source, seed)
        return directory
    return None


def publish(output, source, seed):
    root = ROOT / BASE / source
    destination = root / ('seed-' + str(seed))
    if destination.exists():
        raise FileExistsError('Observed seed evidence cannot be overwritten')
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ('seed-' + str(seed) + '.tmp')
    temporary.mkdir(exist_ok=False)
    try:
        for path in sorted(Path(output).rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in ALLOWED:
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError('Oversized evidence artifact: ' + path.name)
                target = temporary / path.relative_to(output)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        shutil.copy2(CONFIG, temporary / 'source-config.yaml')
        shutil.copy2(PROTOCOL, temporary / 'protocol.json')
        write_json(temporary / 'publication-manifest.json', {
            'source_commit': source, 'seed': seed, 'config_sha256': digest(CONFIG),
            'protocol_sha256': digest(PROTOCOL),
            'file_sha256': {path.relative_to(temporary).as_posix(): digest(path)
                            for path in sorted(temporary.rglob('*')) if path.is_file()},
            'checkpoint_scope': 'weights stay in Kaggle outputs; only their SHA256 is published'})
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    config = json.loads(CONFIG.read_text())
    observed = {}
    for value in config['seeds']:
        path = root / ('seed-' + str(value))
        if path.exists():
            verify(path, source, value)
            observed[str(value)] = json.loads((path / 'decision.json').read_text())
    complete = len(observed) == len(config['seeds']) and all(
        row['comparable_at_full_budget'] for row in observed.values())
    write_json(root / 'matrix-decision.json', {'status': 'DEVELOPMENT_COMPLETE' if complete else 'DEVELOPMENT_INCOMPLETE',
        'scientific_decision': False, 'seeds': observed,
        'all_seed_minimum_gate': complete and all(row['gates']['minimum'] for row in observed.values())})
    run_git(ROOT, 'config', 'user.name', 'MiniCells Kaggle')
    run_git(ROOT, 'config', 'user.email', 'kaggle@minicells.local')
    run_git(ROOT, 'add', '--', root.relative_to(ROOT).as_posix())
    staged = run_git(ROOT, 'diff', '--cached', '--name-only').stdout.splitlines()
    if any(not name.startswith(root.relative_to(ROOT).as_posix() + '/') for name in staged):
        raise RuntimeError('Unrelated staged source detected')
    run_git(ROOT, 'commit', '-m', 'research: record routing 003 development seed ' + str(seed))
    _push(branch(source), 'GITHUB_TOKEN')
    write_json(Path(output) / 'publication.json', {'status': 'published', 'branch': branch(source),
        'path': destination.relative_to(ROOT).as_posix(),
        'publication_commit': run_git(ROOT, 'rev-parse', 'HEAD').stdout.strip()})
