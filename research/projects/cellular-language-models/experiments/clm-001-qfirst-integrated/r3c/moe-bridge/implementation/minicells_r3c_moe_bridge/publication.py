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
    checkout, parent = metadata_checkout(checkout)
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


def metadata_checkout(checkout):
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
    return checkout, git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()


def prepare_pilot_publication(checkout, run):
    checkout, parent = metadata_checkout(checkout)
    branch = "results/r3c-pilot-preflight-" + run[:16]
    git(checkout, "checkout", "-b", branch, "FETCH_HEAD")
    git(checkout, "config", "user.name", "MiniCells Kaggle")
    git(checkout, "config", "user.email", "minicells-kaggle@users.noreply.github.com")
    auth = helper()
    old = auth.ROOT
    try:
        auth.ROOT = checkout
        auth.preflight_push(branch, "GITHUB_TOKEN")
    finally:
        auth.ROOT = old
    return {"checkout": checkout, "parent": parent}


def pilot_public_summary(report):
    if report.get("schema") != "minicells.r3c.pilot.v1" or report.get("confirmation") != "SEALED":
        raise ValueError("known pilot aggregate required")

    def job(row):
        fields = (
            "step",
            "ce",
            "balance_loss",
            "gate_task_grad_norm",
            "gate_balance_grad_norm",
            "gate_balance_weighted_grad_norm",
            "cell_grad_norms",
            "stage_task_grad_norms",
            "stage_balance_grad_norms",
            "clip_ratios",
            "stage_cell_counts",
            "routing_microbatches",
            "wall_seconds",
            "cuda_time_ms",
            "tokens_per_second",
            "peak_gpu_bytes",
            "checkpoint_save_seconds",
        )
        return {
            "job": row["job"],
            "arm": row["arm"],
            "coefficient": row["coefficient"],
            "steps": row["steps"],
            "calibration_mean_nll": row["calibration_mean_nll"],
            "cell_shares": row["cell_shares"],
            "parameters": row["parameters"],
            "cost": row["inference_matmul_cost"],
            "checkpoint_sha256": row["checkpoint"]["sha256"],
            "updates": [{k: log.get(k, "NOT_MEASURED") for k in fields} for log in row["logs"]],
            "projected_500_update_seconds": row["projected_500_update_seconds"],
            "projection_scope": row["projection_scope"],
        }

    return {
        "schema": "minicells.r3c.public-pilot.v1",
        "experiment_id": ID,
        "status": report["status"],
        "source_sha": report["source_sha"],
        "config_sha256": report["config_sha256"],
        "selection": report["selection"],
        "candidates": [job(row) for row in report["candidates"]],
        "arms": [job(row) for row in report["arms"]],
        "logical_optimizer_updates": report["logical_optimizer_updates"],
        "logical_train_tokens": report["logical_train_tokens"],
        "checkpoint_storage_status": report.get("checkpoint_storage_status", "NOT_MEASURED"),
        "hf_checkpoint_reference": report.get("hf_checkpoint_reference", {}),
        "storage_error": report.get("storage_error", {}),
        "failure": report.get("failure", {}),
        "gpu_pilot_completed": report["gpu_pilot_completed"],
        "development_evaluated": False,
        "actual_kaggle_persistence_verified": False,
        "confirmation": "SEALED",
        "formal_500_step_authorized": False,
        "scientific_decision": False,
    }


def publish_pilot_result(report, prepared):
    summary = pilot_public_summary(report)
    checkout = prepared["checkout"]
    parent = prepared["parent"]
    branch = "results/r3c-pilot-" + canonical(summary)[:16] + "-" + parent[:12]
    destination = BASE.relative_to(ROOT) / "reports/published" / (canonical(summary) + ".json")
    existing = git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.strip()
    if existing:
        import json

        git(checkout, "fetch", "--depth=1", "--filter=blob:none", "origin", branch)
        old = json.loads(git(checkout, "show", "FETCH_HEAD:" + str(destination)).stdout)
        if old != summary:
            raise RuntimeError("published pilot identity conflict")
        return {"status": "ALREADY_PUBLISHED", "branch": branch, "parent_main_sha": parent}
    git(checkout, "checkout", "-b", branch, parent)
    atomic(checkout / destination, summary)
    git(checkout, "add", "--", str(destination))
    git(checkout, "commit", "-m", "Publish short R3c pilot aggregates only")
    if git(checkout, "diff", "--name-only", parent, "HEAD").stdout.splitlines() != [
        str(destination)
    ]:
        raise RuntimeError("non-result publication change")
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
