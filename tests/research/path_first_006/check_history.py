"""Assert this engineering branch preserves every historical tracked file."""
import json,subprocess
BASE='0c1886052973547be1a8beded4c44c9dac1538c2'
allowed=('research/projects/cellular-language-models/experiments/006-path-first-clm/',
         'tests/research/path_first_006/', 'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/run_experiment_006.py',
         'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/publish_experiment_006.py','research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/analyze_experiment_006.py',
         '.github/workflows/experiment-006-cpu.yml','research/governance/registry.yaml')
changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
assert all(any(n==p or n.startswith(p) for p in allowed) for n in changed),changed
before=json.loads(subprocess.check_output(['git','show',BASE+':research/governance/registry.yaml'],text=True))
with open('research/governance/registry.yaml') as f:after=json.load(f)
assert [r for r in after['experiments'] if r['id']!='006']==before['experiments']
assert len([r for r in after['experiments'] if r['id']=='006'])==1
print('Historical tracked source/protocol/results and previous registry entries are unchanged.')
