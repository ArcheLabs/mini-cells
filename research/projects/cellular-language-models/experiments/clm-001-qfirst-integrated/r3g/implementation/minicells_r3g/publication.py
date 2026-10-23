"""Use the existing git/auth components; publish only the named R3g aggregate files."""

from pathlib import Path
import shutil
from minicells_r3c_moe_bridge.publication import git, ROOT
from minicells_r3f.publication import push_results
from .common import BASE, canonical, read

ALLOWED = (
    "INPUT_VERIFICATION.json",
    "BRANCH_RESTORE.json",
    "CROSSOVER_MATRIX.json",
    "CELL_PARAMETER_DIFF.json",
    "ROUTING_DISTRIBUTION.json",
    "TRAINING_TRAJECTORY_AUDIT.json",
    "ATTRIBUTION_SUMMARY.json",
    "SCIENTIFIC-FINDINGS.md",
    "RUN_MANIFEST.json",
    "METRICS.json",
    "ENVIRONMENT.json",
    "COMPUTE_COST.json",
)


def publish(output):
    output = Path(output)
    key = canonical(read(output / "reports/METRICS.json"))
    checkout = output / "github-metadata"
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        git(checkout, "init", "--quiet")
        git(checkout, "remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
        git(checkout, "config", "remote.origin.promisor", "true")
        git(checkout, "config", "remote.origin.partialclonefilter", "blob:none")
    git(checkout, "fetch", "--depth=1", "--filter=blob:none", "origin", "main")
    parent = git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
    branch = "results/r3g-" + key[:16] + "-" + parent[:12]
    dest = BASE.relative_to(ROOT) / "reports/published" / key
    git(checkout, "sparse-checkout", "init", "--no-cone")
    git(checkout, "sparse-checkout", "set", "--no-cone", "/" + str(dest.parent) + "/")
    git(checkout, "checkout", "-B", branch, parent)
    git(checkout, "config", "user.name", "MiniCells Kaggle")
    git(checkout, "config", "user.email", "minicells@users.noreply.github.com")
    for name in ALLOWED:
        src = output / "reports" / name
        if src.exists():
            target = checkout / dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
    git(checkout, "add", "--", str(dest))
    staged = git(checkout, "diff", "--cached", "--name-only").stdout.splitlines()
    if not staged or any(
        not p.startswith(str(dest) + "/") or Path(p).name not in ALLOWED for p in staged
    ):
        raise RuntimeError("AGGREGATE_ONLY_PUBLICATION_REQUIRED")
    git(checkout, "commit", "-m", "Publish R3g read-only attribution aggregates")
    remote = git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()
    if remote:
        git(checkout, "fetch", "--depth=1", "origin", branch)
        if (
            git(checkout, "rev-parse", "FETCH_HEAD^{tree}").stdout
            != git(checkout, "rev-parse", "HEAD^{tree}").stdout
        ):
            raise RuntimeError("REMOTE_RESULT_CONTENT_CONFLICT")
        git(checkout, "reset", "--hard", "FETCH_HEAD")
    else:
        push_results(checkout, branch)
    local = git(checkout, "rev-parse", "HEAD").stdout.strip()
    remote = git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()
    if not remote or remote[0] != local:
        raise RuntimeError("REMOTE_RESULT_COMMIT_NOT_VERIFIED")
    return {"status": "PUBLISHED", "branch": branch, "remote_commit_sha": local}
