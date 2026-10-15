"""Static Kaggle gate: syntax, pinned source, modes, no inherited deadlines."""
import ast
import json
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
notebook = json.loads((BASE / 'notebook.ipynb').read_text())
assert notebook['nbformat'] == 4
sources = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        ast.parse(code)
        sources.append(code)
code = '\n'.join(sources)
assert 'SOURCE_COMMIT' in code and 'push_existing' in code
assert 'preflight-only' in code and '--devices' in code
assert 'SETUP_DEADLINE' not in code and '3300' not in code
assert "MODE = 'screen'" in code
assert code.index("command('preflight')") < code.index("['--preflight-only']") < code.index("command('screen')") < code.index("command('publish'),")
assert "subprocess.run(command('preflight'), cwd=ROOT, check=True)" in code
assert "subprocess.run(command('publish') + ['--preflight-only'], cwd=ROOT, check=True)" in code
assert "subprocess.run(command('screen') + ['--devices', *devices], cwd=ROOT, check=True)" in code
print('Notebook cells compile; manual execution has no inherited setup/session deadline.')
