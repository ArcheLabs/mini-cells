"""A cold sparse tree must contain the non-code files read by the runtime."""

import ast
import json
import subprocess
from pathlib import Path

from minicells_fast_r1.common import BASE, ROOT


def test_sparse_delivery_includes_required_cpu_evidence_and_repairs_cached_checkout(tmp_path):
    notebook = json.loads((BASE / "notebooks/kaggle-r1-cpu-gate0.ipynb").read_text())
    code = "".join(notebook["cells"][1]["source"])
    parsed = ast.parse(code)
    patterns = next(
        ast.literal_eval(node.value)
        for node in parsed.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PATTERNS" for t in node.targets)
    )
    relative = BASE.relative_to(ROOT) / "verification/CPU_VALIDATION.json"
    assert "/" + relative.as_posix() in patterns
    assert code.index("if not required.is_file():") > code.index("if actual!=SOURCE_SHA")
    assert code.index("if not required.is_file():") < code.index("'--inspect-cloud-parents'")
    assert "fast-r1-cpu-output-evidence-fix001" in code
    assert "ASSETS=Path('/kaggle/working/fast-r1-cpu-assets')" in code
    target = tmp_path / "cold"
    subprocess.run(["git", "init", "--quiet", str(target)], check=True)
    objects = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--git-path", "objects"], cwd=ROOT, text=True
        ).strip()
    )
    if not objects.is_absolute():
        objects = ROOT / objects
    (target / ".git/objects/info/alternates").write_text(str(objects.resolve()) + "\n")
    subprocess.run(["git", "-C", str(target), "sparse-checkout", "init", "--no-cone"], check=True)
    subprocess.run(
        ["git", "-C", str(target), "sparse-checkout", "set", "--no-cone", *patterns], check=True
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--quiet", "--detach", head], check=True, timeout=60
    )
    assert (target / relative).is_file()
    assert json.loads((target / relative).read_text())["status"] == "PASS_CPU_ENGINEERING_ONLY"


def test_gpu_delivery_uses_reviewed_scope_and_return_only_retry():
    notebook = json.loads((BASE / "notebooks/kaggle-r1-gpu-gate1.ipynb").read_text())
    code = "".join(notebook["cells"][1]["source"])
    compile(code, "gate1_delivery", "exec")
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"]
    assert "'--run-diagnostic','--publish-github'" in code
    assert "'--publish-existing'" in code
    assert "GATE1_DIAGNOSTIC_COMPLETE_INDEPENDENT_REVIEW_REQUIRED" in code
    assert "fast-r1-gate1-output" in code
    assert "ASSETS=Path('/kaggle/working/fast-r1-cpu-assets')" in code
    assert "gate1-authorization" in (BASE / "implementation/minicells_fast_r1/diagnostic.py").read_text()
