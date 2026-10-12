"""Pinned sparse Git checkout, existing secret/publication path, one-hour server cap."""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

COMMIT = '__EXPERIMENT_COMMIT__'
SEED = '__EXPERIMENT_SEED__'


def main():
    started = time.time()
    deadline = started + 3300
    os.environ.update(EXPERIMENT_STARTED_AT=str(started), EXPERIMENT_COMMIT=COMMIT,
                      TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')
    if COMMIT.startswith('__') or SEED.startswith('__'):
        raise RuntimeError('Prepare this launcher through the manual GitHub Actions workflow')
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for name in ('HF_TOKEN', 'GITHUB_TOKEN'):
        token = os.environ.get(name, '').strip() or secrets.get_secret(name).strip()
        if not token:
            raise RuntimeError('Missing Kaggle Secret: ' + name)
        os.environ[name] = token
    source = Path('/kaggle/working/experiment-source')
    source.mkdir(parents=True, exist_ok=False)
    def git(*args, check=True):
        return subprocess.run(['git', *args], cwd=source, check=check, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=max(1, min(300, deadline - time.time() - 180)))
    git('init')
    git('remote', 'add', 'origin', 'https://github.com/ArcheLabs/mini-cells.git')
    git('config', 'remote.origin.promisor', 'true')
    git('config', 'remote.origin.partialclonefilter', 'blob:none')
    git('sparse-checkout', 'init', '--no-cone')
    git('sparse-checkout', 'set', '--no-cone', '/experiments/__init__.py',
        '/experiments/cell_internal_routing_002/', '/experiments/common/',
        '/scripts/research/publish_experiment_results.py',
        '/scripts/research/publish_core_validation_007.py',
        '/research/validations/routing-002-cell-internal-routing/',
        '/artifacts/experiments/002-cell-internal-routing/',
        '/research/08-hybrid-clm/experiments/002-cell-internal-router/',
        '/research/08-hybrid-clm/experiments/_shared/',
        '/research/08-hybrid-clm/scripts/research/publish_experiment_results.py',
        '/research/04-continual-learning/scripts/research/publish_core_validation_007.py',
        '/research/08-hybrid-clm/experiments/002-cell-internal-router/validation/',
        '/research/08-hybrid-clm/experiments/002-cell-internal-router/artifacts/')
    git('fetch', '--depth=1', '--filter=blob:none', 'origin', COMMIT)
    branch = 'results/routing-002-' + COMMIT[:12]
    remote = git('ls-remote', '--exit-code', '--heads', 'origin', branch, check=False)
    if remote.returncode == 0:
        git('fetch', '--depth=1', '--filter=blob:none', 'origin', 'refs/heads/' + branch)
        git('checkout', '-b', branch, 'FETCH_HEAD')
    elif remote.returncode == 2:
        git('checkout', '-b', branch, COMMIT)
    else:
        raise RuntimeError('Cannot inspect the existing result branch before GPU work')
    if any(importlib.util.find_spec(name) is None for name in ('datasets', 'tokenizers')):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        'datasets>=2.18,<5', 'tokenizers>=0.15,<1'], check=True,
                       timeout=max(1, min(600, deadline - time.time() - 180)))
    process = subprocess.Popen([sys.executable, '-m', 'experiments.cell_internal_routing_002.train',
                                '--seed', SEED, '--output', '/kaggle/working/results'],
                                cwd=source, start_new_session=True)
    try:
        result = process.wait(timeout=max(1, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError('Session reached its 55-minute application limit')
    if result:
        raise SystemExit(result)


if __name__ == '__main__':
    main()
