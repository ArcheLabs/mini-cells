"""Paste this Python script into Kaggle with one T4, Internet and GITHUB_TOKEN enabled."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

REMOTE = 'https://github.com/ArcheLabs/mini-cells.git'
BRANCH = 'codex/titans-mlp-qwen3-controlled'
MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
output = Path(tempfile.mkdtemp(prefix='phase3a-', dir='/kaggle/working'))


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
    decision = dict(format='minicells.phase3a.v1', source_commit=source, run_id=run_id, status='ERROR')
    error = None
    try:
        run(sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.3')
        run(sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio')
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(MODEL, revision=REVISION,
                                     allow_patterns=['*.json', '*.safetensors', 'merges.txt'])
        entry = str(project / 'phase3a.py')
        smoke = output / 'smoke'
        run(sys.executable, '-u', entry, snapshot, str(smoke), '--smoke', '--device', 'cuda', cwd=repo)
        cpu = json.loads((project / 'phase3-results/smoke/results.json').read_text())
        gpu = json.loads((smoke / 'results.json').read_text())
        assert cpu['dataset_sha256'] == gpu['dataset_sha256']
        differences = []
        for a, b in zip(cpu['rows'], gpu['rows'], strict=True):
            assert (a['case'], a['pooling'], a['seed']) == (b['case'], b['pooling'], b['seed'])
            for method in ('raw', 'probe'):
                left, right = a['methods'][method][0], b['methods'][method][0]
                differences.append(dict(case=a['case'], pooling=a['pooling'], method=method,
                                        rank_changes=sum(x != y for x, y in zip(left['ranks'], right['ranks'], strict=True)),
                                        max_metric_delta=max(abs(left[k]-right[k])
                                                             for k in ('positive', 'hardest_negative', 'margin'))))
        (output / 'device_comparison.json').write_text(json.dumps(differences, indent=2))
        if any(row['max_metric_delta'] > .001 or row['rank_changes'] > 2 for row in differences):
            raise RuntimeError(f'CPU/T4 smoke differs. Inspect {output}')
        seconds = gpu['elapsed_seconds'] - gpu['loading_seconds']
        print(f'T4 smoke {seconds:.1f}s. Rough full estimate {seconds*95/60:.1f} minutes.', flush=True)
        full = output / 'formal'
        run(sys.executable, '-u', entry, snapshot, str(full), '--device', 'cuda', cwd=repo)
        with (output / 'summary.json').open('w') as handle:
            run(sys.executable, str(project / 'summarize_phase3a.py'), str(full/'results.json'), stdout=handle)
        decision["status"] = json.loads((output / "summary.json").read_text())["decision"]
    except Exception as exc:
        error = exc
        decision['error'] = str(exc)
    (output / 'decision.json').write_text(json.dumps(decision, indent=2))
    stage = repo / 'results/phase3a'
    shutil.copytree(output, stage)
    names = tuple(p.relative_to(stage).as_posix() for p in sorted(stage.rglob('*.json')))
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
