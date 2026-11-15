# %% [markdown]
# # Portable Cell v0.2
# Enable GPU, Internet and the GITHUB_TOKEN secret. Use Save & Run All.

# %%
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from kaggle_secrets import UserSecretsClient

remote = 'https://github.com/ArcheLabs/mini-cells.git'
upstream_url = 'https://github.com/sarahpannn/titans-pytorch.git'
upstream_commit = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
model = 'unsloth/Llama-3.2-1B-Instruct'
working = Path(tempfile.mkdtemp(prefix='portable-cell-e2e-', dir='/kaggle/working'))
output = working / 'results'
run_id = working.name
branch = f'kaggle/{run_id}'
format_name = 'minicells.portable-cell-e2e.v2'


def run(*args):
    subprocess.run(args, check=True)


def digest(path):
    with path.open('rb') as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


with tempfile.TemporaryDirectory(prefix='portable-source-') as directory:
    root = Path(directory)
    repo, upstream = root / 'mini-cells', root / 'titans'
    run('git', 'clone', '--depth', '1', '--branch', 'research/portable-cell-v01', remote, str(repo))
    source = repo / 'research/portable-cell-v01'
    source_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                            text=True).strip()
    decision = dict(format=format_name, run_id=run_id, source_commit=source_commit, status='ERROR')
    try:
        run('git', 'init', str(upstream))
        run('git', '-C', str(upstream), 'fetch', '--depth', '1', upstream_url, upstream_commit)
        run('git', '-C', str(upstream), 'checkout', '--detach', 'FETCH_HEAD')
        run('git', '-C', str(upstream), 'apply', str(source / 'upstream-position.patch'))
        run(sys.executable, '-m', 'pip', 'install', '-q', str(upstream) + '[research]')
        from huggingface_hub import model_info, snapshot_download

        revision = model_info(model).sha
        model_path = Path(snapshot_download(model, revision=revision,
                                            allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja']))
        model_files = {p.name: digest(p) for p in sorted(model_path.iterdir()) if p.is_file()}
        fingerprint = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
        identity = dict(model_id=model, revision=revision, backbone_fingerprint=fingerprint,
                        titan_source_commit=upstream_commit,
                        upstream_config_sha256=digest(upstream / 'configs/seg512_pretrain/training_config.json'),
                        runtime_precision='float32', use_flex_attn=False, use_flash_attn=False)
        candidates = []
        outer = []
        for path in sorted(Path('/kaggle/input').rglob('checkpoint_manifest.json')):
            try:
                record = json.loads(path.read_text())
                weights = path.with_name('outer.safetensors')
                compatible = all(record.get(k) == v for k, v in identity.items())
                compatible = compatible and weights.is_file() and digest(weights) == record.get('outer_sha256')
                if compatible and not outer:
                    outer = ['--outer-checkpoint', str(weights), '--outer-sha256', record['outer_sha256']]
                candidates.append(dict(path=str(path), identity_and_hash_match=compatible))
            except (ValueError, OSError):
                candidates.append(dict(path=str(path), identity_and_hash_match=False))
        (working / 'source_audit.json').write_text(json.dumps(
            dict(identity=identity, model_files=model_files, mounted_candidates=candidates), indent=2))
        print(f'Results directory: {output}', flush=True)
        with (working / 'runtime.log').open('w') as log:
            command = [sys.executable, str(source / 'run_runtime.py'), '--upstream', str(upstream),
                       '--output', str(output), '--revision', revision, '--model-path', str(model_path),
                       '--backbone-fingerprint', fingerprint, *outer]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f'Runtime worker failed with exit code {process.returncode}')
        decision.update(json.loads((output / 'runtime-report.json').read_text()))
    except Exception as error:
        decision['error'] = str(error)
        raise
    finally:
        output.mkdir(exist_ok=True)
        if (output / 'runtime-report.json').exists():
            decision.update(json.loads((output / 'runtime-report.json').read_text()))
        for name in ('source_audit.json', 'runtime.log'):
            if (working / name).exists():
                shutil.move(working / name, output / name)
        (output / 'decision.json').write_text(json.dumps(decision, indent=2))
        (output / 'REPORT.md').write_text(
            f'# Portable Cell v0.2\n\nStatus: {decision["status"]}\n\n'
            'See decision.json, runtime-report.json and semantic_recall.json for evidence.\n')
        manifest = {p.relative_to(output).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                    for p in sorted(output.rglob('*')) if p.is_file()}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        publication = {'status': 'ERROR', 'branch': branch}
        try:
            os.environ['GITHUB_TOKEN'] = UserSecretsClient().get_secret('GITHUB_TOKEN')
            sys.path.insert(0, str(repo))
            from scripts.research.publish_experiment_results import ExperimentSpec, prepare_artifacts, push_results

            stage = repo / 'portable-results'
            selected = []
            for path in sorted(output.rglob('*')):
                if not path.is_file() or path.suffix not in ('.json', '.md', '.log', '.safetensors'):
                    continue
                if path.stat().st_size >= 95 * 1024 * 1024:
                    continue
                relative = path.relative_to(output)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                selected.append(relative.as_posix())
            spec = ExperimentSpec('portable-results', f'artifacts/experiments/minicells08/{run_id}',
                                  tuple(selected), branch, format_name)
            destination = prepare_artifacts(repo, 'minicells08', spec, os.getenv('KAGGLE_KERNEL_VERSION'))
            push_results(repo, destination, 'minicells08', branch, 'GITHUB_TOKEN', 'origin/main')
            publication.update(status='PCC_ARTIFACT_PUBLISH_PASS', commit=subprocess.check_output(
                ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
        except Exception as error:
            publication['error_type'] = type(error).__name__
        (output / 'publication.json').write_text(json.dumps(publication, indent=2))
        print(publication, flush=True)
