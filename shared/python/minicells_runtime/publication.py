"""Metadata-only engineering publication using the existing authenticated Git helpers."""

from pathlib import Path
from .store import atomic, canonical, digest
from .r3 import verify
from clm001_r3 import publication as existing


def summary(root):
    meta = verify(root)
    return {
        "schema_version": 1,
        "experiment_id": meta["experiment_id"],
        "run_id": meta["run_id"],
        "source_sha": meta["source_sha"],
        "status": meta["status"],
        "complete": meta["complete"],
        "scientific_decision": False,
        "formal_confirmation": False,
        "KAGGLE_CROSS_SESSION_RESUME": "NOT_VERIFIED",
        "runtime_config_sha256": meta["runtime_config_sha256"],
        "continuation_sha256": digest(Path(root) / "continuation.json"),
        "arms": {
            a: {
                k: row[k]
                for k in (
                    "last_committed_step",
                    "checkpoint_sha256",
                    "probe_bank_sha256",
                    "event_state_sha256",
                    "training_state_sha256",
                )
            }
            for a, row in meta["arms"].items()
        },
    }


def publish(root):
    root = Path(root)
    value = summary(root)
    source = value["source_sha"]
    namespace = {"engineering": "kaggle-continuation", "run_id": value["run_id"]}
    # Existing helper follows main, authenticates from Kaggle Secrets and stages
    # only explicit metadata paths. Scientific source commits are never pushed.
    checkout, branch = existing.prepare(
        root.parent / "engineering-result-checkout", source, namespace
    )
    destination = (
        checkout
        / "research/projects/cellular-language-models/engineering/kaggle-continuation/artifacts"
        / source
        / canonical(namespace)[:12]
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic(destination / "engineering-progress.json", value)
    relative = (destination / "engineering-progress.json").relative_to(checkout).as_posix()
    existing.git(checkout, "add", "--", relative)
    staged = existing.git(checkout, "diff", "--cached", "--name-only").stdout.splitlines()
    if set(staged) - {relative}:
        raise RuntimeError("NON_METADATA_PUBLICATION_REJECTED")
    if staged:
        existing.git(
            checkout, "commit", "-m", "results: Kaggle continuation engineering " + source[:12]
        )
    helper = existing.helper()
    helper.ROOT = checkout
    helper.push_branch(branch, "GITHUB_TOKEN")
    return {
        "status": "PUBLISHED",
        "branch": branch,
        "run_id": value["run_id"],
        "scientific_decision": False,
    }
