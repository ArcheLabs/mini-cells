from pathlib import Path
import ast
import json
import subprocess


def test_notebook_is_cpu_only_pinned_and_syntax_valid():
    base = Path(__file__).resolve().parents[1]
    nb = json.loads((base / "notebooks/kaggle-smoke.ipynb").read_text())
    code = "".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
    ast.parse(code)
    assert "CUDA_VISIBLE_DEVICES=''" in code
    assert "'--mode','engineering'" in code
    assert "pytest" not in code and "gpu-smoke" not in code
    assert "HF_TOKEN" not in code and "GITHUB_TOKEN" not in code
    assert "0b04e89ef6f14f7c9ad3aa6da5692df8daef9b9b" in code
    root = next(p for p in base.parents if (p / "pyproject.toml").exists())
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "0b04e89ef6f14f7c9ad3aa6da5692df8daef9b9b", "HEAD"],
        cwd=root,
        check=True,
    )
    assert all(not cell.get("outputs") for cell in nb["cells"])
