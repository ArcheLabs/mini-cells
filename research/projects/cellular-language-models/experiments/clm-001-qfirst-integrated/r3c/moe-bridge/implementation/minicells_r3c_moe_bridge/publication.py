"""Publish only constructed aggregate JSON through the existing authentication helper."""

from pathlib import Path
from clm001_r3.publication import helper, git
from .common import BASE, ROOT, ID, canonical, atomic


def public_summary(report):
    if report.get("schema") != "minicells.r3c.engineering.v1":
        raise ValueError("known aggregate schema required")
    return {
        "schema": "minicells.r3c.public-engineering.v1",
        "experiment_id": ID,
        "status": report["status"],
        "source_sha": report["provenance"]["source_sha"],
        "config_sha256": report["provenance"]["config_sha256"],
        "scope": "SYNTHETIC_CPU_ONLY",
        "arms": [
            {
                "arm": r["arm"],
                "parameters": r["parameters"],
                "cost": r["cost"],
                "gate_task_grad_norm": r["updates"][-1]["gate_task_grad_norm"],
                "gate_balance_grad_norm": r["updates"][-1]["gate_balance_grad_norm"],
                "ce": r["updates"][-1]["ce"],
            }
            for r in report["arms"]
        ],
        "resume": report["resume"],
        "gpu_pilot_completed": False,
        "actual_kaggle_persistence_verified": False,
        "full_engineering_pass": False,
        "scientific_decision": False,
        "confirmation": "SEALED",
    }


def publish(summary, checkout):
    if (
        summary.get("schema") != "minicells.r3c.public-engineering.v1"
        or summary.get("confirmation") != "SEALED"
    ):
        raise ValueError("public schema required")
    checkout = Path(checkout)
    if checkout.exists():
        raise ValueError("use a fresh metadata checkout; existing result stays intact")
    checkout.mkdir(parents=True)
    git(checkout, "init", "--quiet")
    git(checkout, "remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
    git(checkout, "config", "core.repositoryformatversion", "1")
    git(checkout, "config", "extensions.partialClone", "origin")
    git(checkout, "config", "remote.origin.promisor", "true")
    git(checkout, "config", "remote.origin.partialclonefilter", "blob:none")
    git(checkout, "sparse-checkout", "init", "--no-cone")
    git(
        checkout,
        "sparse-checkout",
        "set",
        "--no-cone",
        "/README.md",
        "/" + str(BASE.relative_to(ROOT) / "reports" / "published") + "/",
    )
    git(checkout, "fetch", "--depth=1", "--filter=blob:none", "origin", "main")
    parent = git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
    branch = "results/r3c-bridge-" + canonical(summary)[:12] + "-" + parent[:12]
    git(checkout, "checkout", "-b", branch, "FETCH_HEAD")
    git(checkout, "config", "user.name", "MiniCells Kaggle")
    git(checkout, "config", "user.email", "minicells-kaggle@users.noreply.github.com")
    destination = BASE.relative_to(ROOT) / "reports/published" / (canonical(summary) + ".json")
    atomic(checkout / destination, summary)
    git(checkout, "add", "--", str(destination))
    git(checkout, "commit", "-m", "Publish R3c synthetic CPU aggregate only")
    changed = git(checkout, "diff", "--name-only", parent, "HEAD").stdout.splitlines()
    if changed != [str(destination)]:
        raise RuntimeError("publication contains non-result change")
    auth = helper()
    old = auth.ROOT
    try:
        auth.ROOT = checkout
        auth.push_branch(branch, "GITHUB_TOKEN")
    finally:
        auth.ROOT = old
    return {
        "status": "PUBLISHED",
        "branch": branch,
        "parent_main_sha": parent,
        "scientific_decision": False,
    }
