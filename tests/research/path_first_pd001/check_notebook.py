import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT
    / "research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth"
)
notebook = json.loads((BASE / "notebook.ipynb").read_text())
code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
for c in notebook["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
assert "MODE = 'auto'" in code and "'--device','cpu'" in code
assert "UserSecretsClient" not in code and "get_secret" not in code
assert "CUDA_VISIBLE_DEVICES" in code and "'train'" not in code
assert "check=True" in code and "pd001-development-" in code
print("PD notebook uses one automatic development flow; no embedded token values")
performance = json.loads((BASE / 'performance-notebook.ipynb').read_text())
performance_code = '\n'.join(''.join(c['source']) for c in performance['cells'] if c['cell_type']=='code')
for c in performance['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
assert "'perf-audit'" in performance_code and "'resume-audit'" in performance_code
assert '--allow-gpu-microbenchmark' in performance_code
assert "'train'" not in performance_code and 'get_secret' not in performance_code
assert 'RESTORE_FROM' in performance_code and 'BASELINE_COMMIT' in performance_code
print('Performance notebook has bounded same-GPU synthetic measurement and portable restore only')
