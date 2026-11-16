"""Paste this Python script into Kaggle with one T4 and Internet enabled."""
import json
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
    print(output, json.loads((output / 'summary.json').read_text())['decision'], flush=True)
