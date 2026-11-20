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
REV = 'fa2804872c3b4bd748f38c0185cc85775361e735'


def main():
    repo = Path(tempfile.mkdtemp(prefix='ma001-semantic-')) / 'source'
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    '--branch',BRANCH,SOURCE,str(repo)],check=True,timeout=180)
    subprocess.run(['git','sparse-checkout','set','scripts/research',
                    EXPERIMENTS+'/ma-001',EXPERIMENTS+'/ma-001-2'],cwd=repo,check=True)
    sys.path.insert(0,str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results)
    load_github_token('GITHUB_TOKEN')
    name='ma-001-2-semantic-'+uuid.uuid4().hex[:12]
    output=Path('/kaggle/working')/name
    output.mkdir()
    code=1
    reason=None
    try:
        subprocess.run([sys.executable,'-m','pip','install','transformers==4.57.3','sentencepiece'],
                       check=True,timeout=180)
        code=subprocess.run([sys.executable,'-u',str(repo/EXPERIMENTS/'ma-001-2/semantic_audit.py'),
                             REV,str(output/'semantic-audit.json'),'--download'],timeout=420).returncode
    except (subprocess.SubprocessError,OSError) as exc:
        reason=str(exc)
    kind='minicells.ma-001-2.semantic.v1'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    (output/'decision.json').write_text(json.dumps(dict(format=kind,source_commit=source,
        model_revision=REV,status='SCORED' if code==0 else 'BLOCKED_OR_ERROR',
        reason=reason,returncode=code,scientific_status='DEVELOPMENT_ONLY'),indent=2))
    relative='results/'+name
    shutil.copytree(output,repo/relative)
    artifact='artifacts/experiments/'+name
    branch='kaggle/'+name
    subprocess.run(['git','sparse-checkout','add',artifact],cwd=repo,check=True)
    spec=ExperimentSpec(relative,artifact,tuple(sorted(f.name for f in output.iterdir())),branch,kind)
    folder=prepare_artifacts(repo,name,spec,None)
    push_results(repo,folder,name,branch,'GITHUB_TOKEN','origin/main')
    print('Published:',branch,flush=True)
    if code:
        raise RuntimeError('Semantic scoring incomplete; see published decision')


if __name__=='__main__':
    main()
