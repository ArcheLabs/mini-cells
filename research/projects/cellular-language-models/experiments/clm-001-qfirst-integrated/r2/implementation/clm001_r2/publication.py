"""Thin metadata adapter over existing authenticated publisher; never push experiment code or weights."""

from pathlib import Path
import shutil
import subprocess
import sys
from .common import ROOT, BASE, canonical_sha256, write_json, file_sha256

ALLOWED = {
    "run-summary.json",
    "arm-QF-R2-metrics-summary.json",
    "arm-J0-R2-metrics-summary.json",
    "pre-admission-contrast.json",
    "final-system-contrast.json",
    "q-calibration-summary.json",
    "stop-ctx-path-summary.json",
    "shadow-candidate-summary.json",
    "failure-index.json",
    "resource-ledger.json",
    "combined-report.json",
    "cpu-gate.json",
    "parameter-audit.json",
}


def helper():
    paths = [
        ROOT
        / "research/projects/continual-learning/context/04-continual-learning/scripts/research",
        ROOT / "research/projects/cellular-language-models/context/08-hybrid-clm/scripts/research",
    ]
    for path in paths:
        sys.path.insert(0, str(path))
    import publish_core_validation_007

    return publish_core_validation_007


def git(checkout, *args):
    return subprocess.run(
        ["git", *args],
        cwd=checkout,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )


def prepare(checkout, source, namespace):
    if len(source) != 40 or any(v not in "0123456789abcdef" for v in source):
        raise ValueError("full source SHA required")
    checkout = Path(checkout)
    branch = "results/clm001-r2-" + source[:12] + "-" + canonical_sha256(namespace)[:8]
    if not checkout.exists():
        checkout.mkdir(parents=True)
        git(checkout, "init", "--quiet")
        git(checkout, "checkout", "-b", branch)
        git(checkout, "remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
    git(checkout, "config", "user.name", "MiniCells Kaggle")
    git(checkout, "config", "user.email", "minicells-kaggle@users.noreply.github.com")
    git(checkout, "fetch", "--depth=128", "origin", "main:refs/remotes/origin/main")
    remote = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if remote.returncode == 0:
        git(checkout, "fetch", "--depth=128", "origin", branch + ":refs/remotes/origin/" + branch)
        git(checkout, "checkout", "-B", branch, "refs/remotes/origin/" + branch)
        # Preserve prior metadata commits, then follow current main. No historical source commits are published.
        git(checkout, "merge", "--no-edit", "refs/remotes/origin/main")
    elif remote.returncode == 2:
        git(checkout, "checkout", "-B", branch, "refs/remotes/origin/main")
    else:
        raise RuntimeError("cannot inspect result branch before GPU work")
    p = helper()
    p.ROOT = checkout
    p.preflight_push(branch, "GITHUB_TOKEN")
    return checkout, branch


def publish(output, checkout, source, namespace, *, preflight_only=False):
    checkout, branch = prepare(checkout, source, namespace)
    if preflight_only:
        return {"status": "PUBLISH_PREFLIGHT_PASSED", "branch": branch}
    output = Path(output)
    destination = (
        checkout / BASE.relative_to(ROOT) / "artifacts" / source / canonical_sha256(namespace)[:12]
    )
    destination.mkdir(parents=True, exist_ok=True)
    staged = []
    for name in ALLOWED:
        path = output / name
        if not path.is_file():
            continue
        if path.is_symlink() or path.stat().st_size > 2_000_000:
            raise ValueError("metadata path/size rejected")
        from .report import verified

        value = verified(path)
        if not isinstance(value, dict):
            raise ValueError("metadata object required")
        # Only aggregate evidence is whitelisted: checkpoints, tokens, target-labelled Oracle rows and raw traces stay Kaggle Output.
        shutil.copy2(path, destination / name)
        staged.append(str((destination / name).relative_to(checkout)))
    if not staged:
        raise ValueError("no whitelisted result metadata")
    write_json(
        destination / "publication.json",
        {
            "source_commit": source,
            "namespace": namespace,
            "experiment_id": "CLM-001-R2",
            "scientific_decision": False,
            "files": sorted(staged),
            "file_sha256": {str(Path(name).name): file_sha256(checkout / name) for name in staged},
        },
    )
    staged.append(str((destination / "publication.json").relative_to(checkout)))
    git(checkout, "config", "user.name", "MiniCells Kaggle")
    git(checkout, "config", "user.email", "minicells-kaggle@users.noreply.github.com")
    git(checkout, "add", "--", *staged)
    names = git(checkout, "diff", "--cached", "--name-only").stdout.splitlines()
    if set(names) - set(staged) or any(name.startswith(".github/") for name in names):
        raise ValueError("non-result files were staged")
    if names:
        git(checkout, "commit", "-m", "results: CLM-001-R2 " + source[:12])
    p = helper()
    p.ROOT = checkout
    p.push_branch(branch, "GITHUB_TOKEN")
    return {
        "status": "GITHUB_RESULTS_PUSHED",
        "publication_status": "PUSHED",
        "branch": branch,
        "commit": git(checkout, "rev-parse", "HEAD").stdout.strip(),
    }
