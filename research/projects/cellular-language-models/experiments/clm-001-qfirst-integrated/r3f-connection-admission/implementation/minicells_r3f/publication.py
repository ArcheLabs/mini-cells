"""Public aggregates only, current-main parent, existing auth helper."""

from pathlib import Path
import shutil
from .common import read


def publish(output):
    summary = read(Path(output) / "reports/METRICS.json")
    # Reuse exact result-only main-descended publishing discipline; no pre-training GitHub handshake.
    from minicells_r3c_moe_bridge.publication import git, helper, ROOT
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
    git(prepared, "commit", "-m", "Publish R3f aggregate results only")
    changed = git(prepared, "diff", "--name-only", parent, "HEAD").stdout.splitlines()
    if not changed or any(
        not p.startswith(str(dest) + "/") or Path(p).name not in allowed for p in changed
    ):
        raise RuntimeError("RESULT_ONLY_PUBLICATION_REQUIRED")
    auth = helper()
    old_root = auth.ROOT
    old_git = auth.run_git

    def bounded(*args, **kwargs):
        kwargs.setdefault("timeout", 45)
        return old_git(*args, **kwargs)

    try:
        auth.ROOT = prepared
        auth.run_git = bounded
        auth.push_branch(branch, "GITHUB_TOKEN")
    finally:
        auth.ROOT = old_root
        auth.run_git = old_git
    return {
        "status": "PUBLISHED",
        "branch": branch,
        "parent_main_sha": parent,
        "report_directory": str(dest),
    }
