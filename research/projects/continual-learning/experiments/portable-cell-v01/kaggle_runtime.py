"""Portable Cell v0.2 Kaggle launcher. Enable GPU, Internet and GITHUB_TOKEN."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

remote = 'https://github.com/ArcheLabs/mini-cells.git'
branch = 'research/portable-cell-v01'
run_id = 'portable-cell-e2e-' + uuid.uuid4().hex[:12]
work = Path('/kaggle/working') / run_id
work.mkdir()
output = work / 'results'
log = work / 'raw.log'
result_branch = 'kaggle/' + run_id
format_name = 'minicells.portable-cell-e2e.v2'

with tempfile.TemporaryDirectory() as tmp:
    repo = Path(tmp) / 'repo'
    subprocess.run(['git', 'clone', '--depth=1', '--branch', branch, remote, str(repo)],
                   check=True, timeout=300)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    sys.path.insert(0, str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results,
    )

    load_github_token('GITHUB_TOKEN')
    command = [sys.executable, '-u', str(repo / 'research/portable-cell-v01/kaggle_experiment.py'),
               str(output)]
    with log.open('w') as stream, subprocess.Popen(
        command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        exit_code = process.wait()

    output.mkdir(exist_ok=True)
    report = output / 'runtime-report.json'
    decision = json.loads(report.read_text()) if report.exists() else {'status': 'ERROR'}
    if exit_code:
        decision.update(status='ERROR', exit_code=exit_code)
    decision.update(format=format_name, source_commit=source, run_id=run_id)
    (output / 'decision.json').write_text(json.dumps(decision, indent=2))
    (output / 'REPORT.md').write_text(f'# Portable Cell v0.2\n\nStatus: {decision["status"]}\n')
    shutil.move(log, output / log.name)
    log = output / log.name
    audit = work / 'source_audit.json'
    if audit.exists():
        shutil.move(audit, output / audit.name)

    manifest = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                manifest[path.relative_to(output).as_posix()] = {
                    'sha256': hashlib.file_digest(handle, 'sha256').hexdigest(),
                    'bytes': path.stat().st_size,
                }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    stage = repo / 'results/portable-cell-e2e'
    selected = []
    for path in sorted(output.rglob('*')):
        if (path.is_file() and path.suffix in ('.json', '.md', '.log', '.safetensors')
                and path.stat().st_size < 95 * 1024 * 1024):
            relative = path.relative_to(output)
            (stage / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, stage / relative)
            selected.append(relative.as_posix())

    receipt = {'status': 'PUBLICATION_FAILED', 'branch': result_branch}
    try:
        spec = ExperimentSpec('results/portable-cell-e2e', 'artifacts/experiments/minicells08/' + run_id,
                              tuple(selected), result_branch, format_name)
        destination = prepare_artifacts(repo, 'minicells08', spec, os.getenv('KAGGLE_KERNEL_VERSION'))
        push_results(repo, destination, 'minicells08', result_branch, 'GITHUB_TOKEN', 'origin/main')
        receipt.update(status='REMOTE_SHA_VERIFIED', acceptance='PCC_ARTIFACT_PUBLISH_PASS',
                       commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip())
    except Exception as error:
        receipt['error_type'] = type(error).__name__
    (output / 'publication.json').write_text(json.dumps(receipt, indent=2))
    print(decision['status'], receipt, flush=True)
    if exit_code or receipt['status'] != 'REMOTE_SHA_VERIFIED':
        raise RuntimeError(f'Inspect {log} and {output / "publication.json"}')
