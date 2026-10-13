import ast,json,re
from pathlib import Path
p=Path("research/08-hybrid-clm/experiments/006-path-first-clm/notebook-dev-pilot.ipynb")
notebook=json.loads(p.read_text())
code="\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code")
for c in notebook["cells"]:
    if c["cell_type"]=="code":
        assert c["execution_count"] is None and c["outputs"]==[]
        ast.parse("".join(c["source"]))
assert "UNREGISTERED DEVELOPMENT ONLY" in "".join(notebook["cells"][0]["source"])
assert re.search(r'SOURCE_COMMIT = "[0-9a-f]{40}"',code)
assert 'MODE = "train_pair"' in code
assert '"-m", "exp006.development_pilot"' in code
assert '"--mode", "train"' not in code or "exp006.development_pilot" in code
assert "confirmation" not in code.lower() or "formal_confirmation" in code
assert "CUDA_VISIBLE_DEVICES" in code
print("006 pilot Notebook source pin, clean cells and Python syntax PASS")
