"""Kaggle launcher. Actions replaces the payload with this commit's experiment files."""
from __future__ import annotations

import base64
import importlib.util
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PAYLOAD = "__EXPERIMENT_PAYLOAD__"
COMMIT = "__EXPERIMENT_COMMIT__"


def main():
    os.environ['EXPERIMENT_STARTED_AT'] = str(time.time())
    os.environ['EXPERIMENT_COMMIT'] = COMMIT
    if PAYLOAD.startswith('__'):
        raise RuntimeError('Prepare the Kaggle payload through the GitHub Actions workflow.')
    working = Path('/shared/kaggle/working')
    working.mkdir(parents=True, exist_ok=True)
    archive_path = working / 'experiment-source.zip'
    archive_path.write_bytes(base64.b64decode(PAYLOAD, validate=True))
    source = working / 'experiment-source'
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(source)
    if any(importlib.util.find_spec(name) is None for name in ('datasets', 'tokenizers')):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        'datasets>=2.18,<5', 'tokenizers>=0.15,<1'], check=True, timeout=600)
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.argv = ['train', '--output', str(working / 'results')]
    from experiments.runners.train import main as train
    train()


if __name__ == '__main__':
    main()
