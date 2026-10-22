"""Public aggregates only, current-main parent, existing auth helper."""

from pathlib import Path
import shutil
from .common import read


def publish(output):
    summary = read(Path(output) / "reports/METRICS.json")
    # Reuse exact result-only main-descended publishing discipline; no pre-training GitHub handshake.
    from minicells_r3c_moe_bridge.publication import git, ROOT
    from .common import BASE, canonical

    prepared = Path(output) / "github-metadata"
    prepared.mkdir(parents=True, exist_ok=True)
    if not (prepared / ".git").exists():
        git(prepared, "init", "--quiet")
        git(prepared, "remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
        git(prepared, "config", "remote.origin.promisor", "true")
        git(prepared, "config", "remote.origin.partialclonefilter", "blob:none")
    git(prepared, "fetch", "--depth=1", "--filter=blob:none", "origin", "main")
    parent = git(prepared, "rev-parse", "FETCH_HEAD").stdout.strip()
    key = canonical(summary)
    branch = "results/r3f-" + key[:16] + "-" + parent[:12]
    if git(prepared, "ls-remote", "origin", "refs/heads/" + branch).stdout.strip():
        return {"status": "ALREADY_PUBLISHED", "branch": branch}
    dest = BASE.relative_to(ROOT) / "reports/published" / key
    git(prepared, "sparse-checkout", "init", "--no-cone")
    git(prepared, "sparse-checkout", "set", "--no-cone", "/" + str(dest.parent) + "/")
    existing = git(prepared, "branch", "--list", branch).stdout.strip()
    if existing:
        git(prepared, "checkout", branch)
    else:
        git(prepared, "checkout", "-b", branch, parent)
    git(prepared, "config", "user.name", "MiniCells Kaggle")
    git(prepared, "config", "user.email", "minicells@users.noreply.github.com")
    allowed = (
        "RUN_MANIFEST.json",
        "METRICS.json",
        "POLICY_STABILITY.json",
        "ADAPTATION_RESULTS.json",
        "GRAPH_SUMMARY.json",
        "TRAINING_COST.json",
        "SCIENTIFIC-FINDINGS.md",
        "ROUTING_DIAGNOSTICS.json",
        "TRUST_REGION_TRACE.jsonl",
        "SINGLE_SEED_TRIAL.json",
        "MATRIX_RELEASE.json",
        "R3E_COMPARISON.json",
    )
    for leaf in allowed:
        src = Path(output) / "reports" / leaf
        if src.exists():
            target = prepared / dest / leaf
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
    git(prepared, "add", "--", str(dest))
    if git(prepared, "diff", "--cached", "--name-only").stdout.strip():
        git(prepared, "commit", "-m", "Publish R3f aggregate results only")
    changed = git(prepared, "diff", "--name-only", parent, "HEAD").stdout.splitlines()
    if not changed or any(
        not p.startswith(str(dest) + "/") or Path(p).name not in allowed for p in changed
    ):
        raise RuntimeError("RESULT_ONLY_PUBLICATION_REQUIRED")
    push_results(prepared, branch)
    local_commit = git(prepared, "rev-parse", "HEAD").stdout.strip()
    remote = git(prepared, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()
    if not remote or remote[0] != local_commit:
        raise RuntimeError("REMOTE_RESULT_COMMIT_NOT_VERIFIED")
    return {
        "status": "PUBLISHED",
        "remote_commit_sha": local_commit,
        "branch": branch,
        "parent_main_sha": parent,
        "report_directory": str(dest),
    }


def push_results(prepared, branch):
    """Use the historical auth helper with its supported call signature."""
    from minicells_r3c_moe_bridge.publication import helper

    auth = helper()
    old_root = auth.ROOT
    try:
        auth.ROOT = prepared
        auth.push_branch(branch, "GITHUB_TOKEN")
    finally:
        auth.ROOT = old_root
