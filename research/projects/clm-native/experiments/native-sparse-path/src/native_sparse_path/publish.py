"""Thin native archive adapter over the existing Askpass GitHub publisher."""
import json
import shutil
import sys
from pathlib import Path
from .reuse import ROOT, BASE, canonical_sha256, file_sha256, write_json

for path in (
    ROOT / 'research/projects/cellular-language-models/context/08-hybrid-clm/scripts/research',
    ROOT / 'research/projects/continual-learning/context/04-continual-learning/scripts/research',
):
    sys.path.insert(0, str(path))
from publish_experiment_results import run_git, validate_origin
from publish_core_validation_007 import preflight_push, push_branch


def prepare(source, secret='GITHUB_TOKEN'):
    validate_origin(ROOT)
    branch = 'results/clm-native-' + source[:12]
    current = run_git(ROOT, 'branch', '--show-current').stdout.strip()
    if current != branch:
        if run_git(ROOT, 'status', '--porcelain', '--untracked-files=no').stdout.strip():
            raise RuntimeError('tracked source changes must be committed before publication')
        remote = run_git(ROOT, 'ls-remote', '--exit-code', '--heads', 'origin', branch, check=False)
        if remote.returncode == 0:
            run_git(ROOT, 'fetch', 'origin', f'refs/heads/{branch}')
            run_git(ROOT, 'checkout', '-B', branch, 'FETCH_HEAD')
            if run_git(ROOT, 'merge-base', '--is-ancestor', source, 'HEAD', check=False).returncode:
                raise ValueError('result branch does not descend from frozen source')
        elif remote.returncode == 2:
            run_git(ROOT, 'checkout', '-b', branch, source)
        else:
            raise RuntimeError('cannot inspect results branch')
    protected = [str(BASE.relative_to(ROOT) / part) for part in ('src', 'configs', 'registration',
                                                              'run.py', 'PREREGISTRATION.md')]
    if run_git(ROOT, 'diff', '--quiet', source, 'HEAD', '--', *protected, check=False).returncode:
        raise ValueError('frozen source differs on results branch')
    preflight_push(branch, secret)
    return branch


def publish(output, source, secret='GITHUB_TOKEN', preflight_only=False):
    branch = prepare(source, secret)
    if preflight_only:
        return {'branch': branch, 'status': 'PUBLISH_PREFLIGHT_PASSED'}
    output = Path(output)
    files = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.suffix in ('.json', '.jsonl', '.log', '.md', '.png', '.csv')}
    if not files or 'decision.json' not in files:
        raise ValueError('complete screening decision required; raw runs remain local')
    for path in output.glob('*/*/metrics.json'):
        result = json.loads(path.read_text())
        if result['identity']['source_commit'] != source:
            raise ValueError('mixed source identities')
    target = BASE / 'artifacts' / source / canonical_sha256(files)
    if target.exists():
        if json.loads((target / 'evidence.json').read_text())['files'] != files:
            raise FileExistsError('immutable evidence changed')
    else:
        target.mkdir(parents=True)
        for name in files:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / name, destination)
        write_json(target / 'evidence.json', {'source_commit': source, 'files': files,
            'scientific_decision': False, 'checkpoints': 'retained in Kaggle output; only hashes published'})
    run_git(ROOT, 'config', 'user.name', 'MiniCells Kaggle')
    run_git(ROOT, 'config', 'user.email', 'research@minicells.invalid')
    run_git(ROOT, 'add', '--', str(target.relative_to(ROOT)))
    if run_git(ROOT, 'diff', '--cached', '--quiet', check=False).returncode:
        run_git(ROOT, 'commit', '-m', 'research: archive CLM Native screening evidence')
    push_branch(branch, secret)
    return {'branch': branch, 'archive': str(target), 'status': 'PUBLISHED'}
