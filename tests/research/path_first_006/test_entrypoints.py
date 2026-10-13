import json,subprocess,sys
import pytest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]

@pytest.mark.parametrize('mode',['train','profile'])
def test_unregistered_cli_refuses_before_missing_source_and_gpu(mode):
    result=subprocess.run([sys.executable,'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/run_experiment_006.py','--mode',mode,
        '--assets','not-a-bundle','--device','cuda:999'],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode and 'NOT REGISTERED' in result.stderr
    assert 'CUDA' not in result.stderr and 'FileNotFoundError' not in result.stderr

def test_unregistered_publish_refuses_before_git_secrets():
    result=subprocess.run([sys.executable,'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/publish_experiment_006.py',
        '--seed','606','--source-commit','0'*40,'--preflight-only'],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode and 'NOT REGISTERED' in result.stderr
    assert 'GitHub write preflight' not in result.stderr

def test_notebook_defaults_and_parse():
    p=ROOT/'research/projects/cellular-language-models/experiments/006-path-first-clm/notebook.ipynb'
    n=json.loads(p.read_text());code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    import ast
    ast.parse(code)
    assert "MODE = 'preflight'" in code and code.index('authorize(c,p,SEED,ROLE)')<code.index('from kaggle_secrets')
