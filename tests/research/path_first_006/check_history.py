"""Assert this engineering branch preserves every historical tracked file."""
import json,subprocess
BASE='0c1886052973547be1a8beded4c44c9dac1538c2'
allowed=('research/08-hybrid-clm/experiments/006-path-first-clm/',
         'tests/research/path_first_006/', 'scripts/research/run_experiment_006.py',
         'scripts/research/publish_experiment_006.py','scripts/research/analyze_experiment_006.py',
         '.github/workflows/experiment-006-cpu.yml','research/registry.yaml')
changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
assert all(any(n==p or n.startswith(p) for p in allowed) for n in changed),changed
before=json.loads(subprocess.check_output(['git','show',BASE+':research/registry.yaml'],text=True))
with open('research/registry.yaml') as f:after=json.load(f)
assert [r for r in after['experiments'] if r['id']!='006']==before['experiments']
assert len([r for r in after['experiments'] if r['id']=='006'])==1
print('Historical tracked source/protocol/results and previous registry entries are unchanged.')
