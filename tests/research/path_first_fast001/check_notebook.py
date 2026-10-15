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
literals = {
    n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
}
assert "MODE" not in values and "APPROVE_DEVELOPMENT" not in values
assert {"auto", "--devices", "cuda:0", "cuda:1"} <= literals
assert "PILOT_FILE" not in code and "ACCEPT_SLOWER_THAN_TARGET" not in code
assert "killpg" not in code and "timeout=" not in code
print("FAST Run All: automatic checks then development; no manual staging or hard deadline")
