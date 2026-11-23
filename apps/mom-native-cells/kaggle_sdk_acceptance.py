"""Kaggle Python Script: GPU, Internet, existing GITHUB_TOKEN secret."""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import os
import uuid
from pathlib import Path

SOURCE = 'https://github.com/ArcheLabs/mini-cells.git'
BRANCH = 'codex/minicells-python-sdk-001'
APP = 'apps/mom-native-cells'
REV = 'f4081422ea2cf01b1b5d646cac05cad62a5d5438'
MOM = '125038087b71b635c171cec857bd0f21d10955b6'
FLA = '0.2.2'
FLA_SHA256 = 'b6a62b140209d1350861d768e9acb3786548a47960cc324aca39de940ca6b83f'


def main():
    started = time.monotonic()
    scratch = Path(tempfile.mkdtemp(prefix='mom-preflight-'))
    repo = scratch / 'source'
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    '--branch',BRANCH,SOURCE,str(repo)],check=True,timeout=180)
    subprocess.run(['git','sparse-checkout','set','scripts/research', 'src',
                    APP],cwd=repo,check=True)
    sys.path.insert(0,str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results)
    load_github_token('GITHUB_TOKEN')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    resume = os.getenv("MINICELLS_SDK_PUBLISH_ID")
    if resume and not re.fullmatch(r"minicells-python-sdk-001-[0-9a-f]{12}", resume):
        raise ValueError("Invalid SDK acceptance run ID")
    name = resume or 'minicells-python-sdk-001-' + uuid.uuid4().hex[:12]
    output = Path('/kaggle/working') / name
    receipt = output / "experiment" / "decision.json"
    identity = output / "run-identity.json"
    version_id = os.getenv('KAGGLE_SCRIPT_VERSION_ID')
    if resume:
        if not receipt.is_file():
            raise FileNotFoundError("A saved completed SDK smoke is required for publication retry")
        recorded = json.loads(identity.read_text())
        if source != recorded["source_commit"]:
            raise ValueError("Saved SDK smoke requires the original source revision")
        version_id = recorded["script_version_id"]
    else:
        output.mkdir()
        identity.write_text(json.dumps(dict(source_commit=source, script_version_id=version_id)))
    code = 1
    reason = None
    try:
        if not resume:
            execute(scratch, repo, output)
        code = 0
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        reason = str(exc)
        print(reason, flush=True)

    kind = 'minicells.python-sdk.001.v1'
    result = json.loads(receipt.read_text()) if receipt.exists() else {}
    status = 'GPU_SMOKE_COMPLETE' if code == 0 and result.get('status') == 'GPU_SMOKE_COMPLETE' else 'BLOCKED'
    code = int(status == 'BLOCKED')
    diagnosis = result.get('diagnosis') if not code else (reason or 'Missing SDK smoke receipt')
    (output/'decision.json').write_text(json.dumps(dict(
        format=kind, source_commit=source, model_revision=REV, mom_revision=MOM,
        tokenizer_revision='46b15820a4df269e99aed9d709e017677c15d24b',
        status=status, fla_version=FLA, fla_wheel_sha256=FLA_SHA256,
        diagnosis=diagnosis, generation_quality=result.get('generation_quality'),
        elapsed_seconds=time.monotonic()-started, script_version_id=version_id,
        returncode=code, scientific_status='DEVELOPMENT_ONLY'), indent=2))
    relative='results/'+name
    shutil.copytree(output,repo/relative,ignore=shutil.ignore_patterns('*.pt','*.sqlite','*.zip','original','restored'))
    artifact='artifacts/experiments/'+name
    branch='kaggle/'+name
    subprocess.run(['git','sparse-checkout','add',artifact],cwd=repo,check=True)
    spec=ExperimentSpec(relative,artifact,tuple(sorted(f.relative_to(repo/relative).as_posix() for f in (repo/relative).rglob('*') if f.is_file())),branch,kind)
    folder=prepare_artifacts(repo,name,spec,version_id)
    push_results(repo,folder,name,branch,'GITHUB_TOKEN','origin/main')
    print('Published:',branch,flush=True)
    if code:
        raise RuntimeError('MiniCells Python SDK smoke incomplete; see published decision')


def execute(scratch, repo, output):
    from importlib.metadata import version
    def run(args, **kw):
        subprocess.run(args, check=True, **kw)
    run(["git", "checkout", "--ignore-skip-worktree-bits", "HEAD", "--",
         "pyproject.toml", "README.md", "LICENSE"], cwd=repo, timeout=120)
    constraints=repo.parent/'constraints.txt'
    constraints.write_text('torch=='+version('torch')+'\ntriton=='+version('triton')+'\n')
    import hashlib
    wheel_dir=scratch/'wheels'
    wheel_dir.mkdir()
    run([sys.executable,'-m','pip','download','--only-binary=:all:','--no-deps',
         '--timeout','15','--retries','2','--dest',str(wheel_dir),
         'flash-linear-attention=='+FLA],timeout=90)
    wheel=wheel_dir/'flash_linear_attention-0.2.2-py3-none-any.whl'
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=FLA_SHA256:
        raise ValueError('FLA wheel SHA-256 mismatch')
    run([sys.executable,'-m','pip','install','--only-binary=:all:',
         '--timeout','15','--retries','2','-c',str(constraints),
         'transformers==4.51.3','sentencepiece','fastapi>=0.115,<1','httpx',str(wheel)],timeout=180)
    run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(repo)], timeout=180)
    run([sys.executable, '-m', 'unittest', '-v', 'test_runtime', 'test_recall',
         'test_compositional_recall', 'test_multicell_fusion', 'test_gdr_diagnostics',
         'test_multicell_residual', 'test_sdk'], cwd=repo/APP, timeout=180)
    run([sys.executable, '-u', str(repo/APP/'sdk_acceptance.py'), str(output/'experiment')],
        timeout=1800)

if __name__=='__main__':
    main()
