"""Sparse blobless metadata publisher, reusing existing Git/authentication helpers."""

from pathlib import Path
import time
from minicells_runtime.store import atomic, canonical, digest, ContinuationError
from clm001_r3 import publication as existing
from clm001_r3b.publication import reject_private_payload


def prepare(parent, run):
    root = Path(parent) / "cf001-publication"
    branch = "results/r3b-cf001-" + canonical(run)[:16]
    timings = {}
    start = time.perf_counter()
    if not (root / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        existing.git(root, "init", "--quiet")
        existing.git(root, "remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
        existing.git(root, "config", "remote.origin.promisor", "true")
        existing.git(root, "config", "remote.origin.partialclonefilter", "blob:none")
        existing.git(root, "sparse-checkout", "init", "--no-cone")
        existing.git(
            root,
            "sparse-checkout",
            "set",
            "--no-cone",
            "/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/counterfactual-stop-route/reports/",
        )
    existing.git(root, "config", "user.name", "MiniCells Kaggle")
    existing.git(root, "config", "user.email", "minicells-kaggle@users.noreply.github.com")
    before = time.perf_counter()
    existing.git(
        root,
        "fetch",
        "--filter=blob:none",
        "--depth=128",
        "origin",
        "main:refs/remotes/origin/main",
    )
    timings["fetch_main_seconds"] = time.perf_counter() - before
    import subprocess

    remote = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    before = time.perf_counter()
    if remote.returncode == 0:
        existing.git(
            root,
            "fetch",
            "--filter=blob:none",
            "--depth=128",
            "origin",
            branch + ":refs/remotes/origin/" + branch,
        )
        existing.git(root, "checkout", "-B", branch, "refs/remotes/origin/" + branch)
        existing.git(root, "merge", "--no-edit", "refs/remotes/origin/main")
    elif remote.returncode == 2:
        existing.git(root, "checkout", "-B", branch, "refs/remotes/origin/main")
    else:
        raise ContinuationError("CF_PUBLICATION_BRANCH_LOOKUP_FAILED")
    timings["result_branch_prepare_seconds"] = time.perf_counter() - before
    helper = existing.helper()
    helper.ROOT = root
    before = time.perf_counter()
    helper.preflight_push(branch, "GITHUB_TOKEN")
    timings["write_permission_check_seconds"] = time.perf_counter() - before
    timings["total_preflight_seconds"] = time.perf_counter() - start
    print({"stage": "CF_GITHUB_PREFLIGHT_PASS", **timings}, flush=True)
    return root, branch, timings


def publish(prepared, values, run, output):
    root, branch, _ = prepared
    start = time.perf_counter()
    directory = (
        root
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/counterfactual-stop-route/reports"
        / ("cf-" + canonical(run)[:16])
    )
    staged = []
    allowed = {
        "experiment-manifest.json",
        "checkpoint-validation.json",
        "old-analysis-preservation.json",
        "baseline-parity.json",
        "cf1-stop-value-calibration.json",
        "cf2-forced-stop.json",
        "cf3-repeat-constraint.json",
        "resource-ledger.json",
        "coverage-ledger.json",
        "sampling-summary.json",
    }
    for name, value in values.items():
        if name not in allowed:
            raise ContinuationError("CF_NON_PUBLIC_REPORT_REJECTED")
        reject_private_payload(value)
        atomic(directory / name, value)
        staged.append((directory / name).relative_to(root).as_posix())
    import shutil

    figures = Path(output) / "public/figures"
    for name in ("cf1-stop-misranking.png", "cf2-paired-effects.png", "cf3-paired-effects.png"):
        if (figures / name).exists():
            target = directory / "figures" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(figures / name, target)
            staged.append(target.relative_to(root).as_posix())
    from .reporting import findings

    path = directory / "findings.md"
    path.write_text(findings(values), encoding="utf-8")
    staged.append(path.relative_to(root).as_posix())
    existing.git(root, "add", "--", *staged)
    changed = existing.git(root, "diff", "--cached", "--name-only").stdout.splitlines()
    if set(changed) - set(staged):
        raise ContinuationError("CF_NON_METADATA_STAGE_REJECTED")
    if changed:
        existing.git(root, "commit", "-m", "results: read-only CF-001 aggregate evidence")
    helper = existing.helper()
    helper.ROOT = root
    helper.push_branch(branch, "GITHUB_TOKEN")
    sha = existing.git(root, "rev-parse", "HEAD").stdout.strip()
    if existing.git(root, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()[0] != sha:
        raise ContinuationError("CF_PUBLICATION_READBACK_FAILED")
    receipt = {
        "status": "PUBLISHED",
        "branch": branch,
        "content_commit_sha": sha,
        "files": {p: digest(root / p) for p in staged},
        "publication_seconds": time.perf_counter() - start,
        "scientific_decision": False,
    }
    atomic(Path(output) / "publication.json", receipt)
    # Publish a separate receipt without inventing a circular self hash.
    atomic(directory / "publication.json", receipt)
    rel = (directory / "publication.json").relative_to(root).as_posix()
    existing.git(root, "add", "--", rel)
    if existing.git(root, "diff", "--cached", "--name-only").stdout.strip():
        existing.git(root, "commit", "-m", "results: CF-001 publication receipt")
    helper.push_branch(branch, "GITHUB_TOKEN")
    head = existing.git(root, "rev-parse", "HEAD").stdout.strip()
    if existing.git(root, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()[0] != head:
        raise ContinuationError("CF_PUBLICATION_READBACK_FAILED")
    atomic(
        Path(output) / "publication.json",
        {**receipt, "receipt_commit_sha": head, "publication_seconds": time.perf_counter() - start},
    )
    return receipt
