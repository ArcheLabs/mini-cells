"""Verify committed runtime inputs before preflight, execution and resume."""

import hashlib
import json
import subprocess
from pathlib import Path


def verify_source_tree(repo):
    repo = Path(repo)
    modified = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--"], cwd=repo, text=True
    ).splitlines()
    if modified:
        raise RuntimeError("D1_UNCOMMITTED_TRACKED_SOURCE_CHANGES:" + ",".join(modified))
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True
    ).splitlines()
    if any(Path(name).suffix == ".py" for name in untracked):
        raise RuntimeError("D1_UNTRACKED_PYTHON_SOURCE")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def code_sha(base, repo):
    base, repo = Path(base), Path(repo)
    files = sorted((base / "implementation/minicells_r3mf_d1").rglob("*.py")) + [base / "run.py"]
    rows = [
        (p.relative_to(repo).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
