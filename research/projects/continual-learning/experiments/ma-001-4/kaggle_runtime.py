"""Kaggle Python Script: CPU, Internet, existing GITHUB_TOKEN secret."""
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
REV = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
E5 = '614241f622f53c4eeff9890bdc4f31cfecc418b3'


def main():
    repo = Path(tempfile.mkdtemp(prefix='ma001-semantic-')) / 'source'
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    '--branch',BRANCH,SOURCE,str(repo)],check=True,timeout=180)
    subprocess.run(['git','sparse-checkout','set','scripts/research',
                    *(EXPERIMENTS+'/'+n for n in ('ma-001','ma-001-1','ma-001-4','ham-001'))],cwd=repo,check=True)
    sys.path.insert(0,str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results)
    load_github_token('GITHUB_TOKEN')
    name='ma-001-4-runtime-'+uuid.uuid4().hex[:12]
    output=Path('/kaggle/working')/name
    output.mkdir()
    code=1
    reason=None
    try:
        import tomllib
        from importlib.metadata import version
        scratch=repo.parent
        upstream=scratch/'titans'
        subprocess.run(['git','init',str(upstream)],check=True)
        subprocess.run(['git','fetch','--depth','1',
            'https://github.com/sarahpannn/titans-pytorch.git',REV],cwd=upstream,check=True,timeout=180)
        subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=upstream,check=True)
        deps=tomllib.loads((upstream/'pyproject.toml').read_text())['project']['dependencies']
        constraints=scratch/'constraints.txt'
        constraints.write_text('torch=='+version('torch')+'\n')
        subprocess.run([sys.executable,'-m','pip','install','-c',str(constraints),
            'transformers==4.57.3','sentencepiece','assoc-scan==0.0.6','accelerated-scan==0.3.1',
            *(d for d in deps if not d.startswith('torch'))],check=True,timeout=240)
        snapshot=subprocess.check_output([sys.executable,'-c',
            "from huggingface_hub import snapshot_download; import sys; print(snapshot_download('intfloat/multilingual-e5-small',revision=sys.argv[1],allow_patterns=['*.json','model.safetensors','sentencepiece.bpe.model']))",
            E5],text=True,timeout=420).strip()
        index=scratch/'index'
        index.mkdir()
        subprocess.run([sys.executable,'-u',str(repo/EXPERIMENTS/'ma-001-1/run.py'),
                        snapshot,str(index),'--phase','write'],check=True,timeout=240)
        runtime=repo/EXPERIMENTS/'ma-001-4'
        code=subprocess.run([sys.executable,'-u',str(runtime/'run_experiment.py'),
            str(upstream),str(output/'runtime'),'--mode','all','--index',str(index),
            '--e5',snapshot],cwd=runtime,timeout=420).returncode
    except (subprocess.SubprocessError,OSError) as exc:
        reason=str(exc)
    kind='minicells.ma-001-4.runtime.v1'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    receipt=output/'runtime'/'decision.json'
    status=json.loads(receipt.read_text())['status'] if receipt.exists() else 'BLOCKED_OR_ERROR'
    (output/'decision.json').write_text(json.dumps(dict(format=kind,source_commit=source,
        model_revision=REV,status=status, e5_revision=E5,
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
        raise RuntimeError('Runtime acceptance incomplete; see published decision')


if __name__=='__main__':
    main()
