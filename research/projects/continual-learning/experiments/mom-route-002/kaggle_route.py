"""Kaggle Python Script: GPU, Internet, existing GITHUB_TOKEN secret."""
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

SOURCE = 'https://github.com/ArcheLabs/mini-cells.git'
BRANCH = 'codex/ma-001'
EXPERIMENTS = 'research/projects/continual-learning/experiments'
REV = 'f4081422ea2cf01b1b5d646cac05cad62a5d5438'
MOM = '125038087b71b635c171cec857bd0f21d10955b6'
FLA = '0.2.2'
FLA_SHA256 = 'b6a62b140209d1350861d768e9acb3786548a47960cc324aca39de940ca6b83f'


def main():
    scratch = Path(tempfile.mkdtemp(prefix='mom-preflight-'))
    repo = scratch / 'source'
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    '--branch',BRANCH,SOURCE,str(repo)],check=True,timeout=180)
    subprocess.run(['git','sparse-checkout','set','scripts/research',
                    *(EXPERIMENTS+'/'+n for n in ('mom-int-001','mom-route-002','ma-001'))],cwd=repo,check=True)
    sys.path.insert(0,str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results)
    load_github_token('GITHUB_TOKEN')
    name='mom-route-002-'+uuid.uuid4().hex[:12]
    output=Path('/kaggle/working')/name
    output.mkdir()
    code=1
    reason=None
    try:
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
             'transformers==4.51.3',str(wheel)],timeout=180)
        def download(model,revision,patterns):
            return subprocess.check_output([sys.executable,'-c',
                'from huggingface_hub import snapshot_download; import sys; '
                'print(snapshot_download(sys.argv[1],revision=sys.argv[2],allow_patterns=sys.argv[3:]))',
                model,revision,*patterns],text=True,timeout=420).strip()
        snapshot=download('linear-moe-hub/MoM-Gated-Deltanet-340M',REV,
                          ['config.json','model.safetensors'])
        tokenizer=download('fla-hub/gla-1.3B-100B','46b15820a4df269e99aed9d709e017677c15d24b',
                           ['*.json','tokenizer.model'])
        e5=download('intfloat/multilingual-e5-small','614241f622f53c4eeff9890bdc4f31cfecc418b3',
                    ['*.json','model.safetensors','sentencepiece.bpe.model'])
        code=subprocess.run([sys.executable,'-u',str(repo/EXPERIMENTS/'mom-route-002/run.py'),
            str(upstream),snapshot,tokenizer,e5,str(output/'experiment')],timeout=1200).returncode
    except (subprocess.SubprocessError,OSError,ValueError) as exc:
        reason=str(exc)
        print(reason,flush=True)
    kind='minicells.mom-route-002.v1'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    receipt=output/'experiment'/'decision.json'
    status=json.loads(receipt.read_text())['status'] if receipt.exists() else 'BLOCKED'
    if code==0 and status not in ('GO','NO_GO','PARTIAL_PASS'):
        code=1
    (output/'decision.json').write_text(json.dumps(dict(format=kind,source_commit=source,
        model_revision=REV,status=status, fla_version=FLA,fla_wheel_sha256=FLA_SHA256,
        reason=reason,returncode=code,scientific_status='DEVELOPMENT_ONLY'),indent=2))
    relative='results/'+name
    shutil.copytree(output,repo/relative,ignore=shutil.ignore_patterns('*.pt','*.sqlite'))
    artifact='artifacts/experiments/'+name
    branch='kaggle/'+name
    subprocess.run(['git','sparse-checkout','add',artifact],cwd=repo,check=True)
    spec=ExperimentSpec(relative,artifact,tuple(sorted(f.relative_to(repo/relative).as_posix() for f in (repo/relative).rglob('*') if f.is_file())),branch,kind)
    folder=prepare_artifacts(repo,name,spec,None)
    push_results(repo,folder,name,branch,'GITHUB_TOKEN','origin/main')
    print('Published:',branch,flush=True)
    if code:
        raise RuntimeError('MoM routed-memory evaluation incomplete; see published decision')


if __name__=='__main__':
    main()
