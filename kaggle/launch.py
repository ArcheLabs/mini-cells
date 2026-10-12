"""Pinned checkout and bounded process, reused by the canonical notebook."""
from __future__ import annotations
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def prepare_source(commit, seed):
    if len(commit) != 40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('Prepare the notebook for an exact source SHA through experiment-run')
    if seed not in (42, 123, 456):
        raise ValueError('Unregistered development seed')
    started = time.time()
    deadline = started + 3300
    os.environ.update(EXPERIMENT_STARTED_AT=str(started), EXPERIMENT_COMMIT=commit,
                      TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for name in ('HF_TOKEN', 'GITHUB_TOKEN'):
        try:
            token = os.environ.get(name, '').strip() or secrets.get_secret(name).strip()
        except Exception as exc:
            raise RuntimeError('Cannot read Kaggle Secret ' + name + '. Open the original '
                'bingjiangli/minicells editor, enable this existing secret, and run there. '
                'API authentication does not grant runtime secret access.') from exc
        if not token:
            raise RuntimeError('Empty Kaggle Secret: ' + name)
        os.environ[name] = token
    source = Path('/kaggle/working/experiment-source')
    source.mkdir(exist_ok=False)
    def git(*args, check=True):
        return subprocess.run(['git', *args], cwd=source, check=check, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=max(1, min(120, deadline - time.time() - 180)))
    git('init')
    git('remote', 'add', 'origin', 'https://github.com/ArcheLabs/mini-cells.git')
    git('sparse-checkout', 'init', '--no-cone')
    git('sparse-checkout', 'set', '--no-cone', '/src/minicells/', '/experiments/__init__.py',
        '/experiments/common/', '/experiments/cell_internal_routing_002/',
        '/experiments/003-hard-cell-transition/', '/scripts/research/publish_experiment_results.py',
        '/scripts/research/publish_core_validation_007.py', '/artifacts/experiments/003-hard-cell-transition/')
    git('fetch', '--depth=1', '--filter=blob:none', 'origin', commit)
    branch = 'results/routing-003-' + commit[:12]
    remote = git('ls-remote', '--exit-code', '--heads', 'origin', branch, check=False)
    if remote.returncode == 0:
        git('fetch', '--depth=1', '--filter=blob:none', 'origin', 'refs/heads/' + branch)
        git('checkout', '-b', branch, 'FETCH_HEAD')
        # Ensure the original source object is retained for preflight comparison.
    elif remote.returncode == 2:
        git('checkout', '-b', branch, commit)
    else:
        raise RuntimeError('Could not inspect existing immutable results before GPU work')
    # Explicit exact dependency versions: an already-installed datasets 5 cannot bypass pins.
    requirements = {'datasets': '4.0.0', 'tokenizers': '0.21.4', 'matplotlib': '3.10.6'}
    missing = []
    for name, version in requirements.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            missing.append(name + '==' + version)
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *missing],
                       check=True, timeout=max(1, min(420, deadline - time.time() - 180)))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(source / 'src'), str(source)])
    config = json.loads((source / 'experiments/003-hard-cell-transition/config.yaml').read_text())
    return source, deadline, config


def run_experiment(source, deadline, seed):
    process = subprocess.Popen([sys.executable, '-m', 'minicells.training.runner',
                               '--seed', str(seed), '--output', '/kaggle/working/results-003'],
                               cwd=source, start_new_session=True)
    try:
        result = process.wait(timeout=max(1, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError('55-minute application limit reached; partial evidence remains in outputs')
    if result:
        raise RuntimeError('Experiment worker failed; inspect results-003 before rerunning an observed seed')
