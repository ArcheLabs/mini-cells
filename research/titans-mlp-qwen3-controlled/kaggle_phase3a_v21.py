"""Paste this Python script into Kaggle with one T4, Internet and GITHUB_TOKEN enabled."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

REMOTE = 'https://github.com/ArcheLabs/mini-cells.git'
BRANCH = 'codex/phase3a-v21-template-smoke'
MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
output = Path(tempfile.mkdtemp(prefix='phase3a-v21-', dir='/kaggle/working'))


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


with tempfile.TemporaryDirectory() as tmp:
    repo = Path(tmp) / 'repo'
    run('git', 'clone', '--depth=1', '--branch', BRANCH, REMOTE, str(repo))
    project = repo / 'research/titans-mlp-qwen3-controlled'
    sys.path.insert(0, str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results,
    )

    load_github_token('GITHUB_TOKEN')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    run_id = output.name
    branch = 'kaggle/' + run_id
    decision = dict(format='minicells.phase3a-v21.v1', source_commit=source, run_id=run_id, status='ERROR')
    error = None
    try:
        run(sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.3')
        run(sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio')
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(MODEL, revision=REVISION,
                                     allow_patterns=['*.json', '*.safetensors', 'merges.txt'])
        smoke = output / 'smoke'
        run(sys.executable, '-u', str(project / 'phase3a_v21.py'), snapshot,
            str(smoke), '--device', 'cuda', cwd=repo)
        data = json.loads((smoke / 'results.json').read_text())
        assert data['device'].startswith('cuda') and 'T4' in data['gpu'] and data['dtype'] == 'torch.float32'
        decision.update(status=data['status'], test=data['test'], validation_recall1=data['validation_recall1'])
    except Exception as exc:
        error = exc
        decision['error'] = str(exc)
    (output / 'decision.json').write_text(json.dumps(decision, indent=2))
    stage = repo / 'results/phase3a'
    shutil.copytree(output, stage)
    names = tuple(p.relative_to(stage).as_posix() for p in sorted(stage.rglob('*')) if p.is_file())
    spec = ExperimentSpec('results/phase3a', f'artifacts/experiments/titans-mlp-qwen3-controlled/{run_id}',
                          names, branch, decision['format'])
    receipt = dict(status='PUBLICATION_FAILED', branch=branch)
    try:
        destination = prepare_artifacts(repo, 'phase3a', spec, os.getenv('KAGGLE_KERNEL_VERSION'))
        push_results(repo, destination, 'phase3a', branch, 'GITHUB_TOKEN', 'origin/main')
        receipt.update(status='REMOTE_SHA_VERIFIED', commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip())
    finally:
        (output / 'publication.json').write_text(json.dumps(receipt, indent=2))
        print(decision, receipt, flush=True)
    if error is not None:
        raise error
