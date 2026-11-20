"""Paste into a Kaggle Python Script; enable Internet, T4, and GITHUB_TOKEN."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid

SOURCE = 'https://github.com/ArcheLabs/mini-cells.git'
BRANCH = 'codex/qwen3-titans-reproduction'
TITANS = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
DIRECTORY = 'research/titans-qwen3-06b/qwen3-titans-reproduction'


def main():
    import tomllib
    from importlib.metadata import version

    os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', USE_TF='0',
                      HF_HOME='/tmp/qwen-reproduction-hf')
    working = Path('/kaggle/working')
    scratch = Path(tempfile.mkdtemp(prefix='qwen-repro-'))
    repo, upstream = scratch / 'source', scratch / 'titans'
    subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
                    '--branch', BRANCH, SOURCE, str(repo)], check=True, timeout=180)
    subprocess.run(['git', 'sparse-checkout', 'set', DIRECTORY, 'scripts/research'],
                   cwd=repo, check=True, timeout=180)
    sys.path.insert(0, str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results,
    )
    load_github_token('GITHUB_TOKEN')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=repo).strip()
    name = 'qwen3-titans-smoke-' + uuid.uuid4().hex[:12]
    output = working / name
    output.mkdir()
    env = os.environ | {'PYTHONPATH': str(upstream), 'TITANS_SOURCE': str(upstream)}
    phase = 'setup'
    success = False
    with (output / 'run.log').open('w') as log:
        def run(*args, cwd=repo):
            print(f'[{phase}] {args[0]}', flush=True)
            with subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as proc:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                if proc.wait():
                    raise subprocess.CalledProcessError(proc.returncode, args)
        try:
            run('git', 'init', str(upstream))
            run('git', 'fetch', '--depth', '1', 'https://github.com/sarahpannn/titans-pytorch.git',
                TITANS, cwd=upstream)
            run('git', 'checkout', '--detach', 'FETCH_HEAD', cwd=upstream)
            deps = tomllib.loads((upstream / 'pyproject.toml').read_text())['project']['dependencies']
            constraints = scratch / 'constraints.txt'
            constraints.write_text('torch==' + version('torch') + '\n')
            run(sys.executable, '-m', 'pip', 'install', '-c', str(constraints),
                'transformers==4.57.6', 'datasets==4.4.1', 'assoc-scan==0.0.6',
                'accelerated-scan==0.3.1', 'wandb', 'accelerate',
                *(d for d in deps if not d.startswith('torch')))
            phase = 'deterministic-tests'
            run(sys.executable, '-m', 'unittest', '-v', 'test_run', 'test_model', cwd=repo / DIRECTORY)
            phase = 'real-data-smoke-and-restore'
            run(sys.executable, 'run.py', str(upstream), str(output / 'smoke'),
                '--smoke', cwd=repo / DIRECTORY)
            success = True
        except Exception:
            traceback.print_exc(file=log)
            traceback.print_exc()

    kind = 'minicells.qwen3-titans-kaggle-smoke.v1'
    branch = 'kaggle/' + name
    script_version = os.getenv('KAGGLE_SCRIPT_VERSION_ID')
    decision = dict(format=kind, source_commit=source, run_id=name,
                    kaggle_script_version_id=script_version,
                    status='ENGINEERING_SMOKE_PASS' if success else 'FAIL', phase=phase,
                    scientific_status='NOT_EVALUATED')
    (output / 'decision.json').write_text(json.dumps(decision, indent=2))
    relative = f'results/{name}'
    shutil.copytree(output, repo / relative, ignore=shutil.ignore_patterns('*.pt'))
    files = tuple(sorted(p.relative_to(repo / relative).as_posix()
                         for p in (repo / relative).rglob('*') if p.is_file()))
    artifact = f'artifacts/experiments/{name}'
    subprocess.run(['git', 'sparse-checkout', 'add', artifact], cwd=repo, check=True)
    spec = ExperimentSpec(relative, artifact, files, branch, kind)
    folder = prepare_artifacts(repo, name, spec, script_version)
    push_results(repo, folder, name, branch, 'GITHUB_TOKEN', 'origin/main')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    print(dict(status='REMOTE_SHA_VERIFIED', branch=branch, commit=commit), flush=True)
    if not success:
        raise RuntimeError(f'{phase} failed; see published run.log')


if __name__ == '__main__':
    main()
