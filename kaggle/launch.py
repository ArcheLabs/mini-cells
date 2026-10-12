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
    try:
        secrets = UserSecretsClient()
    except Exception as exc:
        secrets = None
        secrets_error = exc
    else:
        secrets_error = None

    hf_token = os.environ.get('HF_TOKEN', '').strip()
    if not hf_token and secrets is not None:
        try:
            hf_token = secrets.get_secret('HF_TOKEN').strip()
        except Exception as exc:
            # TinyStories is public; a missing or inaccessible token need not stop this run.
            print('Optional Kaggle Secret HF_TOKEN unavailable (' + type(exc).__name__ +
                  '); continuing with anonymous access to the public dataset.', flush=True)
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
    else:
        os.environ.pop('HF_TOKEN', None)
        if secrets_error is not None:
            print('Kaggle Secrets client unavailable for optional HF_TOKEN (' +
                  type(secrets_error).__name__ + '); using anonymous dataset access.', flush=True)

    github_token = os.environ.get('GITHUB_TOKEN', '').strip()
    if not github_token:
        if secrets is None:
            raise RuntimeError('Required Kaggle Secret GITHUB_TOKEN is unavailable; '
                               'the existing GitHub preflight/publisher cannot run.') from secrets_error
        try:
            github_token = secrets.get_secret('GITHUB_TOKEN').strip()
        except Exception as exc:
            raise RuntimeError('Required Kaggle Secret GITHUB_TOKEN is unavailable; '
                               'the existing GitHub preflight/publisher cannot run.') from exc
    if not github_token:
        raise RuntimeError('Required Kaggle Secret GITHUB_TOKEN is empty.')
    os.environ['GITHUB_TOKEN'] = github_token
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
        '/experiments/003-hard-cell-transition/',
        '/scripts/research/publish_experiment_results.py',
        '/research/08-hybrid-clm/experiments/003-hard-cell-transition/',
        '/research/08-hybrid-clm/experiments/_shared/',
        '/research/08-hybrid-clm/scripts/research/publish_experiment_results.py',
        '/research/04-continual-learning/scripts/research/publish_core_validation_007.py',
        '/scripts/research/publish_core_validation_007.py',
        '/artifacts/experiments/003-hard-cell-transition/',
        '/research/08-hybrid-clm/experiments/003-hard-cell-transition/artifacts/')
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
