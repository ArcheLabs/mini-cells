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
BRANCH = 'codex/mom-multicell-residual-001'
APP = 'apps/mom-native-cells'
REV = 'f4081422ea2cf01b1b5d646cac05cad62a5d5438'
MOM = '125038087b71b635c171cec857bd0f21d10955b6'
FLA = '0.2.2'
FLA_SHA256 = 'b6a62b140209d1350861d768e9acb3786548a47960cc324aca39de940ca6b83f'


def main():
    started = time.monotonic()
    scratch = Path(tempfile.mkdtemp(prefix='mom-multicell-residual-'))
    repo = scratch / 'source'
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    '--branch',BRANCH,SOURCE,str(repo)],check=True,timeout=180)
    subprocess.run(['git','sparse-checkout','set','scripts/research',
                    APP],cwd=repo,check=True)
    sys.path.insert(0,str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results)
    load_github_token('GITHUB_TOKEN')
    resume = os.getenv("MINICELLS_RESIDUAL_PUBLISH_ID")
    if resume is not None and not re.fullmatch(r"mom-multicell-residual-001-[0-9a-f]{12}", resume):
        raise ValueError("Invalid residual run ID")
    name = resume or 'mom-multicell-residual-001-' + uuid.uuid4().hex[:12]
    output = Path('/kaggle/working') / name
    if resume is not None and not output.is_dir():
        raise FileNotFoundError(f"Saved run not found: {output}")
    output.mkdir(exist_ok=bool(resume))
    print("Residual run:", name, flush=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    version_id = os.getenv('KAGGLE_SCRIPT_VERSION_ID')
    identity_path = output / 'run-identity.json'
    if resume:
        identity = json.loads(identity_path.read_text())
        if identity['source_commit'] != source:
            raise ValueError("Saved run requires its original source commit")
        version_id = identity['script_version_id']
    else:
        identity_path.write_text(json.dumps(dict(source_commit=source, script_version_id=version_id)))
    finished = output / 'experiment' / 'decision.json'
    code=1
    reason=None
    try:
        if resume and not finished.exists():
            raise ValueError("Only a completed residual experiment can be republished")
        if not finished.exists():
            execute(scratch, repo, output)
        code = 0
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        reason = str(exc)
        print(reason, flush=True)

    kind = 'minicells.mom-multicell-residual-001.v1'
    experiment = output/'experiment'
    receipt = experiment/'decision.json'
    result = json.loads(receipt.read_text()) if receipt.exists() else {}
    status = 'DIAGNOSTIC_COMPLETE' if code == 0 and result.get('status') == 'DIAGNOSTIC_COMPLETE' else 'BLOCKED'
    code = int(status == 'BLOCKED')
    accuracy = {}
    if not code:
        outcome = json.loads((experiment/'retrieval-metrics.json').read_text())['final']
        scores = outcome['methods'][outcome['primary']]
        accuracy = {key: scores[key] for key in ('auroc', 'auprc', 'pairwise', 'recall_at_1', 'recall_at_2', 'mrr')}
        diagnosis = result['research']
    else:
        diagnosis = reason or 'Experiment did not emit a completion receipt'
        failure = experiment / 'failure.json'
        if failure.exists():
            detail = json.loads(failure.read_text())
            diagnosis = f"{detail['stage']}: {detail['type']}: {detail['error']}"
    version_id = identity['script_version_id'] if resume else version_id
    (output/'decision.json').write_text(json.dumps(dict(
        format=kind, source_commit=source, model_revision=REV, mom_revision=MOM,
        tokenizer_revision='46b15820a4df269e99aed9d709e017677c15d24b',
        status=status, fla_version=FLA, fla_wheel_sha256=FLA_SHA256,
        diagnosis=diagnosis, accuracy=accuracy, dataset_sha256=result.get('dataset_hash'), research=result.get('research'),
        engineering=result.get('engineering', 'BLOCKED'), next_step=result.get('next_step'),
        qa=result.get('qa'),
        elapsed_seconds=result.get("elapsed_seconds", time.monotonic()-started), script_version_id=version_id,
        returncode=code, scientific_status='DEVELOPMENT_ONLY'), indent=2))
    relative='results/'+name
    shutil.copytree(output,repo/relative,ignore=shutil.ignore_patterns('residual-records'))
    artifact='artifacts/experiments/'+name
    branch='kaggle/'+name
    subprocess.run(['git','sparse-checkout','add',artifact],cwd=repo,check=True)
    spec=ExperimentSpec(relative,artifact,tuple(sorted(f.relative_to(repo/relative).as_posix() for f in (repo/relative).rglob('*') if f.is_file())),branch,kind)
    folder=prepare_artifacts(repo,name,spec,version_id)
    push_results(repo,folder,name,branch,'GITHUB_TOKEN','origin/main')
    print('Published:',branch,flush=True)
    if code:
        raise RuntimeError('MoM residual experiment incomplete; see published decision')


def execute(scratch, repo, output):
    from importlib.metadata import version

    upstream=repo.parent/'mom'
    def run(args, **kw):
        subprocess.run(args,check=True,**kw)
    run(['git','init',str(upstream)])
    run(['git','fetch','--depth','1','https://github.com/OpenSparseLLMs/MoM.git',MOM],
        cwd=upstream,timeout=180)
    run(['git','checkout','--detach','FETCH_HEAD'],cwd=upstream)
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
    def download(model,revision,patterns):
        return subprocess.check_output([sys.executable,'-c',
            'from huggingface_hub import snapshot_download; import sys; '
            'print(snapshot_download(sys.argv[1],revision=sys.argv[2],allow_patterns=sys.argv[3:]))',
            model,revision,*patterns],text=True,timeout=420).strip()
    run([sys.executable,'-m','unittest','-v','test_runtime','test_recall'],cwd=repo/APP,timeout=120)
    run([sys.executable, '-m', 'unittest', '-v', 'test_compositional_recall', 'test_multicell_fusion',
         'test_gdr_diagnostics', 'test_multicell_residual'],
        cwd=repo/APP, timeout=120)
    run([sys.executable, '-c',
         'import torch; print(torch.__version__, torch.cuda.get_device_name(), flush=True)'],
        timeout=120)
    snapshot=download('linear-moe-hub/MoM-Gated-Deltanet-340M',REV,
                      ['config.json','model.safetensors'])
    tokenizer=download('fla-hub/gla-1.3B-100B','46b15820a4df269e99aed9d709e017677c15d24b',
                       ['*.json','tokenizer.model'])
    run([sys.executable, '-u', str(repo/APP/'run_multicell_residual.py'),
         str(upstream), snapshot, tokenizer, str(output/'experiment')], timeout=7200)


if __name__=='__main__':
    main()
