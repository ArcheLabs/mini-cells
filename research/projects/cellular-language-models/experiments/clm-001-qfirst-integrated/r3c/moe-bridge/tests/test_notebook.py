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


def test_pilot_notebook_one_launch_pins_authorized_implementation():
    base = Path(__file__).resolve().parents[1]
    nb = json.loads((base / "notebooks/kaggle-pilot.ipynb").read_text())
    code = "".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(code)
    assigns = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    sha = ast.literal_eval(assigns["SOURCE_SHA"])
    assert sha == "ff24f8c9952064c22eab6b6c6acdf88376531e57"
    assert "'--mode','pilot'" in code and "CUDA_VISIBLE_DEVICES" not in code
    assert "pytest" not in code and "KAGGLE_API_TOKEN" not in code and "--authorization" not in code
    assert "max 40 optimizer updates" in code
    assert all(not cell.get("outputs") for cell in nb["cells"])
    root = next(p for p in base.parents if (p / "pyproject.toml").exists())
    subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=root, check=True)
