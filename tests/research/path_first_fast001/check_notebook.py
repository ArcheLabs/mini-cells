import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT
    / "research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/exploration-fast-001"
)
n = json.loads((BASE / "notebook.ipynb").read_text())
code = "\n".join("".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code")
for cell in n["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
tree = ast.parse(code)
values = {
    target.id: node.value.value
    for node in tree.body
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    for target in node.targets
    if isinstance(target, ast.Name)
}
assert values["MODE"] == "preflight"
assert values["ALLOW_GPU_PILOT"] is False and values["APPROVE_DEVELOPMENT"] is False
literals = {
    node.value
    for node in ast.walk(tree)
    if isinstance(node, ast.Constant) and isinstance(node.value, str)
}
assert {"CLM-S16", "CLM-S8", "M0-TINY", "--approve-development", "CUDA_VISIBLE_DEVICES"} <= literals
assert "killpg" not in code and "timeout=" not in code
print(
    "FAST Notebook defaults to CPU-only preflight; explicit reviewed GPU stages, no hard time cap"
)
