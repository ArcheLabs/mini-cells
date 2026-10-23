"""Read the original aggregate report from its independently pinned result commit."""

from pathlib import Path
from minicells_r3c_moe_bridge.publication import git
from .common import atomic, canonical, parent_identity


def expected_summary(c):
    return {
        "identity": parent_identity(c),
        "status": "LOCAL_COADAPTATION_MECHANISM_VERIFIED",
        "hypotheses": {k: "SUPPORTED_EXPLORATORY" for k in ("H1", "H2", "H3")},
        "scientific_decision": False,
        "full_matrix_authorized": False,
        "arms": {
            arm: {
                "valid": True,
                "failure": None,
                "chosen_scale": c["chosen_scale"],
                "adapt_steps": 10,
                "final_quality_accepted": c["selected_branches"][arm] == "new",
            }
            for arm in c["arms"]
        },
    }


def read_trajectory(c, output, local=None):
    import json
    from .common import read, digest

    name = "TRAINING_ROUTE_DIFFERENCE.json"
    if local is not None and (Path(local) / "reports" / name).is_file():
        manifest = read(Path(local) / "reports/RUN_MANIFEST.json")
        if any(manifest.get(k) != v for k, v in parent_identity(c).items()):
            raise RuntimeError("PUBLIC_PARENT_MANIFEST_IDENTITY_MISMATCH")
        path = Path(local) / "reports" / name
        return read(path), {"origin": "LOCAL_ORIGINAL_REPORT", "sha256": digest(path)}
    summary = expected_summary(c)
    key = canonical(summary)
    root = Path(output) / "public-input"
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        git(root, "init", "--quiet")
        git(root, "remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
        git(root, "config", "remote.origin.promisor", "true")
        git(root, "config", "remote.origin.partialclonefilter", "blob:none")
    if c["parent"].get("public_commit_sha"):
        sha = c["parent"]["public_commit_sha"]
        branch = "PINNED_MERGED_RESULT_COMMIT"
        merged = True
    else:
        matches = git(
            root, "ls-remote", "origin", "refs/heads/results/r3fv2-" + key[:16] + "-*"
        ).stdout.splitlines()
        merged = False
        if not matches:
            matches = git(root, "ls-remote", "origin", "refs/heads/main").stdout.splitlines()
            merged = True
        if not matches:
            raise RuntimeError("ORIGINAL_PUBLIC_REPORT_NOT_FOUND")
        sha, branch = matches[0].split()
    git(root, "fetch", "--depth=1", "--filter=blob:none", "origin", sha)
    prefix = (
        "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-v2/reports/published/"
        + key
        + "/"
    )
    actual = json.loads(git(root, "show", sha + ":" + prefix + "METRICS.json").stdout)
    if actual != summary:
        raise RuntimeError("ORIGINAL_PUBLIC_SUMMARY_MISMATCH")
    report = json.loads(git(root, "show", sha + ":" + prefix + name).stdout)
    path = Path(output) / "original-public-report.json"
    atomic(path, report)
    return report, {
        "origin": "VERIFIED_GITHUB_MERGED_RESULT" if merged else "VERIFIED_GITHUB_RESULT",
        "branch": branch,
        "commit_sha": sha,
        "path": prefix + name,
        "sha256": digest(path),
    }
