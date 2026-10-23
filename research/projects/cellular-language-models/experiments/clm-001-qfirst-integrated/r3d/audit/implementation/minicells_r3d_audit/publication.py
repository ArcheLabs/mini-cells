"""Optional aggregate-only return, reusing repository Git auth; never a read prerequisite."""

from pathlib import Path
import uuid
import subprocess
from minicells_r3c_moe_bridge.publication import metadata_checkout, git, helper
from .common import BASE, ROOT, canonical


def publish(output, summary):
    output = Path(output)
    checkout, parent = metadata_checkout(output / ("metadata-" + uuid.uuid4().hex[:8]))
    destination = BASE.relative_to(ROOT) / "reports/published" / canonical(summary)
    git(
        checkout,
        "sparse-checkout",
        "set",
        "--no-cone",
        "/README.md",
        "/" + destination.as_posix() + "/",
    )
    branch = "results/r3d-audit-" + canonical(summary)[:16] + "-" + parent[:12]
    if git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.strip():
        return {"status": "ALREADY_PUBLISHED", "branch": branch}
    git(checkout, "checkout", "-b", branch, "FETCH_HEAD")
    git(checkout, "config", "user.name", "MiniCells Kaggle")
    git(checkout, "config", "user.email", "minicells-kaggle@users.noreply.github.com")
    import shutil

    sources = [
        "AUDIT_SUMMARY.json",
        "COUNTERFACTUAL_VALUES.json",
        "POLICY_RANKING.json",
        "POLICY_INTERPOLATION.json",
        "CHECKPOINT_ATTRIBUTION.json",
        "SCIENTIFIC-FINDINGS.md",
    ]
    sources += ["figures/" + p.name for p in (output / "reports/figures").glob("*.png")]
    for relative in sources:
        target = checkout / destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "reports" / relative, target)
    git(checkout, "add", "--", destination.as_posix())
    git(checkout, "commit", "-m", "Publish read-only R3d-A aggregates and three figures")
    changed = git(checkout, "diff", "--name-only", parent, "HEAD").stdout.splitlines()
    if set(changed) != {str(destination / p) for p in sources}:
        raise RuntimeError("PUBLICATION_CONTAINS_NON_AUDIT_FILES")
    auth = helper()
    old = auth.ROOT
    previous_run_git = auth.run_git

    def bounded_git(root, *args, check=True, env=None):
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            check=check,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
        )

    try:
        auth.ROOT = checkout
        auth.run_git = bounded_git
        auth.push_branch(branch, "GITHUB_TOKEN")
    finally:
        auth.ROOT = old
        auth.run_git = previous_run_git
    return {
        "status": "PUBLISHED",
        "branch": branch,
        "parent_main_sha": parent,
        "report_directory": destination.as_posix(),
    }
