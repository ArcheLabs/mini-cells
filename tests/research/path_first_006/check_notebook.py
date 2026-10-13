import ast,json,re
from pathlib import Path
p=Path('research/08-hybrid-clm/experiments/006-path-first-clm/notebook.ipynb')
n=json.loads(p.read_text());code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
for c in n['cells']:
    if c['cell_type']=='code':
        assert c['execution_count'] is None and c['outputs']==[]
        ast.parse(''.join(c['source']))
assert "MODE = 'preflight'" in code
assert re.search(r"SOURCE_COMMIT = '[0-9a-f]{40}'",code)
assert code.index('authorize(c,p,SEED,ROLE)')<code.index('from kaggle_secrets')
assert 'SETUP_DEADLINE' not in code and 'time.time()' not in code
assert 'CUDA_VISIBLE_DEVICES' in code
assert 'minicells-exp005' not in code and 'confirmation_holdout.npy' not in code
print('Notebook cells compile; CPU default, authorization before Secrets, pinned source and isolated workers verified.')
